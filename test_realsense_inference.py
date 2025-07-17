#!/usr/bin/env python3
"""
Real-time human pose estimation using RealSense camera.
Compares MediaPipe and MMPose 3D pose estimation in real-time.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyrealsense2 as rs
import threading
import time
from collections import deque

# MediaPipe imports
try:
    import mediapipe as mp
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("WARNING: MediaPipe not available")

# H36M skeleton connections for visualization
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),     # Right leg
    (0, 4), (4, 5), (5, 6),     # Left leg  
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13),      # Left arm
    (8, 14), (14, 15), (15, 16),      # Right arm
]

class RealSensePoseEstimator:
    def __init__(self):
        self.pipeline = None
        self.align = None
        self.running = False
        self.current_frame = None
        self.current_depth = None
        self.frame_lock = threading.Lock()
        
        # Initialize MediaPipe
        self.mp_landmarker = None
        if MEDIAPIPE_AVAILABLE:
            self.init_mediapipe()
        
        # Initialize MMPose
        self.mmpose_inferencer = None
        self.init_mmpose()
        
        # For performance (optional)
        self.fps_counter = deque(maxlen=30)
        self.last_time = time.time()
        
    def init_mediapipe(self):
        """Initialize MediaPipe pose landmarker."""
        model_path = 'pose_landmarker.task'
        try:
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE
            )
            self.mp_landmarker = PoseLandmarker.create_from_options(options)
            print("✅ MediaPipe initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            self.mp_landmarker = None
    
    def init_mmpose(self):
        """Initialize MMPose 3D estimator."""
        try:
            from mmpose.apis import MMPoseInferencer
            self.mmpose_inferencer = MMPoseInferencer(
                pose3d='human3d',
                device='cuda:0' if 'cuda' in 'cuda:0' else 'cpu'
            )
            print("✅ MMPose initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MMPose: {e}")
            try:
                self.mmpose_inferencer = MMPoseInferencer(
                    pose3d='human3d',
                    device='cpu'
                )
                print("✅ MMPose initialized on CPU")
            except Exception as e2:
                print(f"❌ Failed to initialize MMPose on CPU: {e2}")
                self.mmpose_inferencer = None
    
    def init_realsense(self):
        """Initialize RealSense camera."""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Configure streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            
            # Start pipeline
            profile = self.pipeline.start(config)
            
            # Create align object
            align_to = rs.stream.color
            self.align = rs.align(align_to)
            
            # Get camera intrinsics
            color_stream = profile.get_stream(rs.stream.color)
            self.color_intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            print("✅ RealSense camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize RealSense: {e}")
            return False
    
    def capture_frames(self):
        """Capture frames from RealSense in a separate thread."""
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                aligned_frames = self.align.process(frames)
                
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()
                
                if color_frame and depth_frame:
                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(depth_frame.get_data())
                    
                    with self.frame_lock:
                        self.current_frame = color_image.copy()
                        self.current_depth = depth_image.copy()
                        
            except Exception as e:
                if self.running:
                    print(f"Frame capture error: {e}")
                continue
    
    def mediapipe_to_h36m(self, mp_keypoints):
        """Convert MediaPipe 33 keypoints to H36M 17 keypoints format."""
        h36m_keypoints = np.zeros((17, 3))
        
        # H36M Index 0: Hip (Root) - Midpoint of left and right hips
        h36m_keypoints[0] = (mp_keypoints[23] + mp_keypoints[24]) / 2
        h36m_keypoints[1] = mp_keypoints[24]  # Right Hip
        h36m_keypoints[2] = mp_keypoints[26]  # Right Knee
        h36m_keypoints[3] = mp_keypoints[28]  # Right Ankle
        h36m_keypoints[4] = mp_keypoints[23]  # Left Hip
        h36m_keypoints[5] = mp_keypoints[25]  # Left Knee
        h36m_keypoints[6] = mp_keypoints[27]  # Left Ankle
        h36m_keypoints[7] = (mp_keypoints[23] + mp_keypoints[24] + mp_keypoints[11] + mp_keypoints[12]) / 4  # Spine
        h36m_keypoints[8] = (mp_keypoints[11] + mp_keypoints[12]) / 2  # Thorax
        h36m_keypoints[9] = mp_keypoints[0]   # Nose
        h36m_keypoints[10] = (mp_keypoints[2] + mp_keypoints[5]) / 2  # Head
        h36m_keypoints[11] = mp_keypoints[11]  # Left Shoulder
        h36m_keypoints[12] = mp_keypoints[13]  # Left Elbow
        h36m_keypoints[13] = mp_keypoints[15]  # Left Wrist
        h36m_keypoints[14] = mp_keypoints[12]  # Right Shoulder
        h36m_keypoints[15] = mp_keypoints[14]  # Right Elbow
        h36m_keypoints[16] = mp_keypoints[16]  # Right Wrist
        
        return h36m_keypoints
    
    def get_mediapipe_pose(self, image):
        """Get MediaPipe pose estimation for an image."""
        if not self.mp_landmarker:
            return None
        
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            result = self.mp_landmarker.detect(mp_image)
            
            if not result.pose_world_landmarks:
                return None
            
            # Extract MediaPipe 3D coordinates and apply coordinate conversion
            mp_keypoints_raw = np.array([[lm.x, -lm.z, -lm.y] for lm in result.pose_world_landmarks[0]])
            
            # Apply coordinate correction: (x, y, z) -> (x, -y, z)
            mp_keypoints_raw[:, 1] = -mp_keypoints_raw[:, 1]
            
            # Convert MediaPipe to H36M format
            mp_converted_3d = self.mediapipe_to_h36m(mp_keypoints_raw)
            
            return mp_converted_3d
            
        except Exception as e:
            print(f"Error in MediaPipe pose estimation: {e}")
            return None
    
    def get_mmpose_pose(self, image):
        """Get MMPose 3D pose estimation for an image."""
        if not self.mmpose_inferencer:
            return None
        
        try:
            results_gen = self.mmpose_inferencer(image, show=False, out_dir=None)
            results_list = list(results_gen)
            
            if len(results_list) > 0:
                result = results_list[0]
                
                if 'predictions' in result:
                    predictions = result['predictions']
                    if len(predictions) > 0:
                        persons_list = predictions[0]
                        if len(persons_list) > 0:
                            person = persons_list[0]
                            
                            if 'keypoints' in person:
                                keypoints_list = person['keypoints']
                                
                                keypoints_3d = np.zeros((len(keypoints_list), 3))
                                for i, kpt in enumerate(keypoints_list):
                                    if len(kpt) >= 3:
                                        keypoints_3d[i, 0] = kpt[0]  # x
                                        keypoints_3d[i, 1] = kpt[1]  # y
                                        keypoints_3d[i, 2] = kpt[2]  # z
                                
                                return keypoints_3d
            
            return None
            
        except Exception as e:
            print(f"Error in MMPose 3D estimation: {e}")
            return None
    
    def plot_3d_skeleton(self, ax, joints_3d, title, color='red', is_mediapipe=False, is_mmpose=False):
        """Plot 3D skeleton on given axis."""
        if joints_3d is None:
            ax.text(0.5, 0.5, 0.5, 'No Pose\nDetected', ha='center', va='center', fontsize=12)
            ax.set_title(title)
            return
        
        # Convert to mm if in meters and center at pelvis
        if np.max(np.abs(joints_3d)) < 10:
            joints_3d = joints_3d * 1000.0
        joints_centered = joints_3d - joints_3d[0]
        
        # Apply coordinate transformations
        if is_mmpose:
            # For MMPose: apply X-axis negation
            joints_centered[:, 0] = -joints_centered[:, 0]
        
        # Clear previous plot
        ax.clear()
        
        # Plot joints
        ax.scatter(joints_centered[:, 0], joints_centered[:, 1], joints_centered[:, 2], 
                   c=color, s=50, alpha=0.8)
        
        # Plot skeleton connections
        for start_idx, end_idx in H36M_SKELETON:
            if start_idx < len(joints_centered) and end_idx < len(joints_centered):
                xs = [joints_centered[start_idx, 0], joints_centered[end_idx, 0]]
                ys = [joints_centered[start_idx, 1], joints_centered[end_idx, 1]]
                zs = [joints_centered[start_idx, 2], joints_centered[end_idx, 2]]
                ax.plot(xs, ys, zs, c=color, linewidth=2, alpha=0.7)
        
        # Set axis labels
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)') 
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
        # Set equal aspect ratio
        if len(joints_centered) > 0:
            xyz_min = np.min(joints_centered, axis=0)
            xyz_max = np.max(joints_centered, axis=0)
            max_range = np.max(xyz_max - xyz_min) / 2.0
            mid_x = (xyz_max[0] + xyz_min[0]) / 2.0
            mid_y = (xyz_max[1] + xyz_min[1]) / 2.0
            mid_z = (xyz_max[2] + xyz_min[2]) / 2.0
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    def calculate_fps(self):
        """Calculate current FPS."""
        current_time = time.time()
        self.fps_counter.append(current_time - self.last_time)
        self.last_time = current_time
        
        if len(self.fps_counter) > 0:
            return len(self.fps_counter) / sum(self.fps_counter)
        return 0
    
    def run_realtime(self):
        """Run real-time pose estimation."""
        if not self.init_realsense():
            return
        
        self.running = True
        
        # Start frame capture thread
        capture_thread = threading.Thread(target=self.capture_frames)
        capture_thread.daemon = True
        capture_thread.start()
        
        # Set up matplotlib for real-time plotting
        plt.ion()
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        
        print("Starting real-time pose estimation... Press 'q' to quit")
        
        try:
            while self.running:
                with self.frame_lock:
                    if self.current_frame is not None:
                        frame = self.current_frame.copy()
                    else:
                        time.sleep(0.01)
                        continue
                
                # Get pose estimations
                mp_3d = self.get_mediapipe_pose(frame)
                mmpose_3d = self.get_mmpose_pose(frame)
                
                # Update plots
                # Original RGB image
                ax1.clear()
                ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax1.set_title("RealSense RGB")
                ax1.axis('off')
                
                # MediaPipe 3D pose
                self.plot_3d_skeleton(ax2, mp_3d, "MediaPipe 3D", color='green', is_mediapipe=True)
                
                # MMPose 3D pose
                self.plot_3d_skeleton(ax3, mmpose_3d, "MMPose 3D", color='purple', is_mmpose=True)
                
                plt.tight_layout()
                plt.draw()
                plt.pause(0.01)
                
                # Check for quit command
                if plt.get_fignums() == []:  # Window closed
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        if self.pipeline:
            self.pipeline.stop()
        if self.mp_landmarker:
            self.mp_landmarker.close()
        plt.ioff()
        plt.close('all')
        print("✅ Cleanup completed")

def main():
    """Main function to run real-time pose estimation."""
    estimator = RealSensePoseEstimator()
    
    print("Real-time Human Pose Estimation with RealSense")
    print("=" * 50)
    print("This application will:")
    print("- Capture RGB and depth streams from RealSense")
    print("- Run MediaPipe and MMPose 3D pose estimation")
    print("- Display real-time 3D pose visualization")
    print("- Show FPS and pose statistics")
    print("")
    print("Requirements:")
    print("- Intel RealSense camera connected")
    print("- MediaPipe pose model (pose_landmarker.task)")
    print("- MMPose installed with 3D pose models")
    print("")
    
    try:
        estimator.run_realtime()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        estimator.cleanup()

if __name__ == "__main__":
    main()