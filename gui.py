import sys
import os
import time
from typing import Optional, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QFrame, QGroupBox,
    QProgressBar, QTextEdit, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt, QSize
from PyQt6.QtGui import QPixmap, QImage, QFont, QPalette, QColor

import cv2
import numpy as np

# Import your pushup counter module (now safe even if ultralytics is missing)
from pushup_counter import (
    PushUpCounter, pick_best_side, angle_three_points,
    draw_pose_overlay, draw_skeleton_lines, draw_angle_indicator,
    MIN_KP_CONF, MIN_ANGLE_CONF, EMA_ALPHA, STALE_TTL
)


class VideoProcessorThread(QThread):
    """Thread for processing video in background"""
    frame_ready = pyqtSignal(np.ndarray)
    progress_update = pyqtSignal(int, int)  # current_frame, total_frames
    stats_update = pyqtSignal(dict)  # statistics dictionary
    processing_complete = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.video_path = None
        self.model_path = "yolo11m-pose.pt"
        self.output_path = None
        self.counter = None
        self.is_running = False
        self.cap = None
        self.writer = None
        
        # Threshold values
        self.down_angle = 90.0
        self.up_angle = 110.0
        
    def set_video_path(self, path: str):
        self.video_path = path
        
    def set_output_path(self, path: str):
        self.output_path = path
        
    def set_thresholds(self, down_angle: float, up_angle: float):
        self.down_angle = down_angle
        self.up_angle = up_angle
        if self.counter and hasattr(self.counter, "down_thresh"):
            self.counter.down_thresh = down_angle
            self.counter.up_thresh = up_angle
    
    def stop_processing(self):
        self.is_running = False
        
    def run(self):
        try:
            if not self.video_path or not os.path.exists(self.video_path):
                self.error_occurred.emit("Video file not found")
                return
                
            # Lazy import of ultralytics YOLO so GUI import never fails if package missing
            try:
                from ultralytics import YOLO
            except Exception as e:
                self.error_occurred.emit(f"Ultralytics YOLO not available: {e}")
                return
            
            # Initialize model
            try:
                model = YOLO(self.model_path)
            except Exception as e:
                self.error_occurred.emit(f"Error loading model: {str(e)}")
                return
            
            # Get video properties
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                self.error_occurred.emit("Cannot open video file")
                return
                
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.cap.release()
            
            # Initialize video writer if output path is set
            if self.output_path:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.writer = cv2.VideoWriter(self.output_path, fourcc, fps, (width, height))
            
            # Initialize counter
            self.counter = PushUpCounter(self.down_angle, self.up_angle, EMA_ALPHA, MIN_ANGLE_CONF)
            
            self.is_running = True
            frame_count = 0
            
            # Run tracking
            results_generator = model.track(
                source=self.video_path,
                stream=True,
                conf=0.5,
                iou=0.5,
                verbose=False,
                tracker="bytetrack.yaml",
                pose=True,
            )
            
            start_time = time.time()
            
            # Initialize stats dictionary
            stats = {
                'total_reps': 0,
                'current_frame': 0,
                'total_frames': total_frames,
                'fps': 0,
                'people_tracked': 0,
                'down_threshold': self.down_angle,
                'up_threshold': self.up_angle,
                'stage': '--',
                'angle': 0,
                'side': '--'
            }
            
            for result in results_generator:
                if not self.is_running:
                    break
                    
                frame_count += 1
                
                # Get the original frame
                frame = result.orig_img.copy()
                
                # Extract detection data
                keypoints = getattr(result, "keypoints", None)
                boxes = getattr(result, "boxes", None)
                person_ids = None
                
                # Get tracking IDs if available
                if boxes is not None and hasattr(boxes, 'id') and boxes.id is not None:
                    try:
                        person_ids = boxes.id.cpu().numpy().astype(int)
                    except Exception:
                        person_ids = None
                
                current_fps = fps if frame_count < 2 else frame_count / (time.time() - start_time)
                
                # Process each detected person
                if keypoints is not None and getattr(keypoints, "xy", None) is not None:
                    try:
                        kps_xy = keypoints.xy.cpu().numpy()
                        kps_conf = keypoints.conf.cpu().numpy() if getattr(keypoints, "conf", None) is not None else None
                    except Exception:
                        kps_xy = None
                        kps_conf = None
                    
                    if kps_xy is not None:
                        for i, person_keypoints in enumerate(kps_xy):
                            # Get person ID
                            person_id = int(person_ids[i]) if person_ids is not None and i < len(person_ids) else i
                            
                            # Get confidence values for this person
                            person_conf = kps_conf[i] if kps_conf is not None else None
                            
                            # Select best arm side for tracking
                            side, pts, avg_confidence = pick_best_side(person_keypoints, person_conf, MIN_KP_CONF)
                            if side == "none" or avg_confidence == 0.0:
                                continue
                            
                            shoulder, elbow, wrist = pts
                            
                            # Calculate elbow angle
                            angle = angle_three_points(shoulder, elbow, wrist)
                            
                            # Update counter and get stats
                            count, stage, smooth_angle = self.counter.update(person_id, angle, side, avg_confidence)
                            
                            # Update stats for the first person for GUI display
                            if i == 0:
                                stats['stage'] = stage
                                stats['angle'] = smooth_angle
                                stats['side'] = side
                            
                            # Draw skeleton and angle
                            draw_skeleton_lines(frame, shoulder, elbow, wrist)
                            draw_angle_indicator(frame, elbow, smooth_angle)
                            
                            # Draw pose overlay
                            try:
                                anchor_x = max(10, min(width - 460, int(elbow[0])))
                                anchor_y = max(55, int(elbow[1]) - 10)
                                # This function is for drawing all keypoints, but we are drawing them manually
                                # draw_pose_overlay(frame, person_keypoints, person_conf, MIN_KP_CONF)
                            except Exception:
                                pass
                
                # Draw total counter
                total_reps = self.counter.get_total_reps()
                
                cv2.rectangle(frame, (10, 5), (300, 45), (0, 0, 0), -1)
                cv2.putText(frame, f"TOTAL REPS: {total_reps}", (15, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 
                           3, cv2.LINE_AA)
                
                # Write frame to output video if enabled
                if self.writer:
                    self.writer.write(frame)
                
                # Emit frame for display
                self.frame_ready.emit(frame)
                
                # Emit progress update
                self.progress_update.emit(frame_count, total_frames)
                
                # Emit statistics
                people_tracked = len(self.counter.state) if hasattr(self.counter, "state") else 0
                
                stats['total_reps'] = total_reps
                stats['current_frame'] = frame_count
                stats['fps'] = current_fps
                stats['people_tracked'] = people_tracked
                
                self.stats_update.emit(stats)
                
                # Clean up stale tracks if supported
                try:
                    if hasattr(self.counter, "cleanup_stale_tracks"):
                        self.counter.cleanup_stale_tracks(STALE_TTL)
                except Exception:
                    pass
            
            self.processing_complete.emit()
            
        except Exception as e:
            self.error_occurred.emit(f"Processing error: {str(e)}")
        finally:
            if self.writer:
                self.writer.release()
            if self.cap:
                self.cap.release()


class PushUpCounterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Push-up Counter - AI Analysis Tool")
        self.setGeometry(100, 100, 1400, 900)
        
        # Apply dark theme
        self.setStyleSheet(self.get_dark_stylesheet())
        
        # Initialize variables
        self.video_thread = VideoProcessorThread()
        self.current_video_path = None
        self.is_processing = False
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        # Setup update timer for real-time stats
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        
    def get_dark_stylesheet(self):
        return """
        QMainWindow {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
        }
        
        QPushButton {
            background-color: #3c3c3c;
            border: 2px solid #555555;
            border-radius: 8px;
            padding: 8px 16px;
            font-weight: bold;
            font-size: 12px;
        }
        
        QPushButton:hover {
            background-color: #4a4a4a;
            border-color: #777777;
        }
        
        QPushButton:pressed {
            background-color: #2a2a2a;
        }
        
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #666666;
            border-color: #333333;
        }
        
        QPushButton#startButton {
            background-color: #0d7377;
            border-color: #14a085;
        }
        
        QPushButton#startButton:hover {
            background-color: #14a085;
        }
        
        QPushButton#stopButton {
            background-color: #d63031;
            border-color: #e17055;
        }
        
        QPushButton#stopButton:hover {
            background-color: #e17055;
        }
        
        QPushButton#exportButton {
            background-color: #6c5ce7;
            border-color: #a29bfe;
        }
        
        QPushButton#exportButton:hover {
            background-color: #a29bfe;
        }
        
        QGroupBox {
            font-weight: bold;
            border: 2px solid #555555;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 10px;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
        }
        
        QLabel {
            color: #ffffff;
        }
        
        QLabel#repsCount {
            color: #00ff88;
            font-size: 48px;
            font-weight: bold;
        }
        
        QLabel#videoDisplay {
            border: 2px solid #555555;
            border-radius: 8px;
            background-color: #2a2a2a;
        }
        
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 8px;
            background: #3c3c3c;
            margin: 2px 0;
            border-radius: 4px;
        }
        
        QSlider::handle:horizontal {
            background: #14a085;
            border: 1px solid #0d7377;
            width: 18px;
            margin: -2px 0;
            border-radius: 9px;
        }
        
        QSlider::handle:horizontal:hover {
            background: #00ff88;
        }
        
        QProgressBar {
            border: 2px solid #555555;
            border-radius: 8px;
            text-align: center;
            background-color: #2a2a2a;
        }
        
        QProgressBar::chunk {
            background-color: #14a085;
            border-radius: 6px;
        }
        
        QTextEdit {
            background-color: #2a2a2a;
            border: 2px solid #555555;
            border-radius: 8px;
            padding: 8px;
        }
        """
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Left side - Video display
        self.setup_video_section(main_layout)
        
        # Right side - Controls and stats
        self.setup_control_section(main_layout)
        
    def setup_video_section(self, main_layout):
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        video_frame.setFixedWidth(800)
        
        video_layout = QVBoxLayout(video_frame)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setObjectName("videoDisplay")
        self.video_label.setFixedSize(760, 570)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("No video loaded\nSelect a video file to begin")
        self.video_label.setStyleSheet("font-size: 18px; color: #888888;")
        video_layout.addWidget(self.video_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        video_layout.addWidget(self.progress_bar)
        
        main_layout.addWidget(video_frame)
        
    def setup_control_section(self, main_layout):
        control_frame = QFrame()
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_frame.setFixedWidth(500)
        
        control_layout = QVBoxLayout(control_frame)
        
        # Reps counter display
        reps_group = QGroupBox("Push-up Counter")
        reps_layout = QVBoxLayout(reps_group)
        
        self.reps_label = QLabel("0")
        self.reps_label.setObjectName("repsCount")
        self.reps_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        reps_layout.addWidget(self.reps_label)
        
        control_layout.addWidget(reps_group)
        
        # Statistics display
        stats_group = QGroupBox("Statistics")
        stats_layout = QGridLayout(stats_group)
        
        # Create statistics labels
        self.stats_labels = {}
        stats_items = [
            ("FPS:", "fps_label"),
            ("Frame:", "frame_label"),
            ("People:", "people_label"),
            ("Stage:", "stage_label"),
            ("Angle:", "angle_label"),
            ("Side:", "side_label")
        ]
        
        for i, (name, key) in enumerate(stats_items):
            name_label = QLabel(name)
            name_label.setStyleSheet("font-weight: bold;")
            value_label = QLabel("--")
            value_label.setStyleSheet("color: #00ff88;")
            
            stats_layout.addWidget(name_label, i // 2, (i % 2) * 2)
            stats_layout.addWidget(value_label, i // 2, (i % 2) * 2 + 1)
            self.stats_labels[key] = value_label
            
        control_layout.addWidget(stats_group)
        
        # Threshold controls
        threshold_group = QGroupBox("Angle Thresholds")
        threshold_layout = QVBoxLayout(threshold_group)
        
        # Down angle slider
        down_layout = QHBoxLayout()
        down_layout.addWidget(QLabel("Down Angle:"))
        self.down_angle_label = QLabel("90°")
        self.down_angle_label.setStyleSheet("color: #ff6b6b; font-weight: bold;")
        down_layout.addWidget(self.down_angle_label)
        threshold_layout.addLayout(down_layout)
        
        self.down_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.down_angle_slider.setRange(60, 120)
        self.down_angle_slider.setValue(90)
        self.down_angle_slider.valueChanged.connect(self.update_down_angle)
        threshold_layout.addWidget(self.down_angle_slider)
        
        # Up angle slider
        up_layout = QHBoxLayout()
        up_layout.addWidget(QLabel("Up Angle:"))
        self.up_angle_label = QLabel("110°")
        self.up_angle_label.setStyleSheet("color: #4ecdc4; font-weight: bold;")
        up_layout.addWidget(self.up_angle_label)
        threshold_layout.addLayout(up_layout)
        
        self.up_angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.up_angle_slider.setRange(100, 160)
        self.up_angle_slider.setValue(110)
        self.up_angle_slider.valueChanged.connect(self.update_up_angle)
        threshold_layout.addWidget(self.up_angle_slider)
        
        control_layout.addWidget(threshold_group)
        
        # File controls
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        # Video selection
        video_btn_layout = QHBoxLayout()
        self.select_video_btn = QPushButton("Select Video File")
        self.select_video_btn.clicked.connect(self.select_video_file)
        video_btn_layout.addWidget(self.select_video_btn)
        file_layout.addLayout(video_btn_layout)
        
        self.video_path_label = QLabel("No video selected")
        self.video_path_label.setStyleSheet("color: #888888; font-size: 10px;")
        self.video_path_label.setWordWrap(True)
        file_layout.addWidget(self.video_path_label)
        
        control_layout.addWidget(file_group)
        
        # Processing controls
        process_group = QGroupBox("Processing Controls")
        process_layout = QVBoxLayout(process_group)
        
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Analysis")
        self.start_btn.setObjectName("startButton")
        self.start_btn.clicked.connect(self.start_processing)
        self.start_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setObjectName("stopButton")
        self.stop_btn.clicked.connect(self.stop_processing)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)
        
        process_layout.addLayout(button_layout)
        
        self.export_btn = QPushButton("Export Annotated Video")
        self.export_btn.setObjectName("exportButton")
        self.export_btn.clicked.connect(self.export_video)
        self.export_btn.setEnabled(False)
        process_layout.addWidget(self.export_btn)
        
        control_layout.addWidget(process_group)
        
        # Add stretch
        control_layout.addStretch()
        
        main_layout.addWidget(control_frame)
        
    def setup_connections(self):
        self.video_thread.frame_ready.connect(self.update_video_display)
        self.video_thread.progress_update.connect(self.update_progress)
        self.video_thread.stats_update.connect(self.update_stats)
        self.video_thread.processing_complete.connect(self.processing_finished)
        self.video_thread.error_occurred.connect(self.handle_error)
        
    def select_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv);;All files (*.*)"
        )
        
        if file_path:
            self.current_video_path = file_path
            self.video_path_label.setText(f"Selected: {os.path.basename(file_path)}")
            self.start_btn.setEnabled(True)
            self.export_btn.setEnabled(True)
            
    def update_down_angle(self, value):
        self.down_angle_label.setText(f"{value}°")
        if hasattr(self.video_thread, 'counter') and self.video_thread.counter:
            self.video_thread.set_thresholds(value, self.up_angle_slider.value())
            
    def update_up_angle(self, value):
        self.up_angle_label.setText(f"{value}°")
        if hasattr(self.video_thread, 'counter') and self.video_thread.counter:
            self.video_thread.set_thresholds(self.down_angle_slider.value(), value)
            
    def start_processing(self):
        if not self.current_video_path:
            return
            
        self.is_processing = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.select_video_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        
        self.video_thread.set_video_path(self.current_video_path)
        self.video_thread.set_thresholds(
            self.down_angle_slider.value(),
            self.up_angle_slider.value()
        )
        self.video_thread.start()
        
    def stop_processing(self):
        self.video_thread.stop_processing()
        
    def export_video(self):
        if not self.current_video_path:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Annotated Video",
            f"pushup_analysis_{int(time.time())}.mp4",
            "MP4 files (*.mp4);;All files (*.*)"
        )
        
        if save_path:
            self.video_thread.set_output_path(save_path)
            self.start_processing()
            
    def update_video_display(self, frame):
        # Convert frame to QPixmap and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Scale frame to fit display
        display_width = self.video_label.width() - 20
        display_height = self.video_label.height() - 20
        
        # Calculate scaling factor
        scale = min(display_width / w, display_height / h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        scaled_frame = cv2.resize(rgb_frame, (new_width, new_height))
        
        qt_image = QImage(scaled_frame.data, new_width, new_height, 
                         new_width * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        self.video_label.setPixmap(pixmap)
        
    def update_progress(self, current, total):
        if total > 0:
            progress = int((current / total) * 100)
            self.progress_bar.setValue(progress)
            
    def update_stats(self, stats):
        self.reps_label.setText(str(stats.get('total_reps', 0)))
        self.stats_labels['fps_label'].setText(f"{stats.get('fps', 0):.1f}")
        self.stats_labels['frame_label'].setText(
            f"{stats.get('current_frame', 0)}/{stats.get('total_frames', 0)}"
        )
        self.stats_labels['people_label'].setText(str(stats.get('people_tracked', 0)))
        self.stats_labels['stage_label'].setText(stats.get('stage', '--').upper())
        self.stats_labels['angle_label'].setText(f"{stats.get('angle', 0):.1f}°")
        self.stats_labels['side_label'].setText(stats.get('side', '--').upper())
        
    def processing_finished(self):
        self.is_processing = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.select_video_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
    def handle_error(self, error_message):
        print(f"Error: {error_message}")
        self.processing_finished()
        
    def update_display(self):
        # This can be used for any real-time updates if needed
        pass
        
    def closeEvent(self, event):
        if self.is_processing:
            self.video_thread.stop_processing()
            self.video_thread.wait(3000)  # Wait up to 3 seconds
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Push-up Counter")
    app.setApplicationVersion("1.0")
    
    # Create and show the main window
    window = PushUpCounterGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
