#!/usr/bin/env python3
"""
visualize_mediapipe_h36m_converted.py

Script to visualize MediaPipe 3D pose detection converted to H36M format
alongside H36M ground truth annotations. Uses the conversion mapping from
mediapipe_h36m.txt to transform MediaPipe 33 keypoints to H36M 17 keypoints.

Usage:
    python visualize_mediapipe_h36m_converted.py [--num_samples N] [--split train|valid]

Requirements:
    - pose_landmarker.task (MediaPipe model file)
    - H36M dataset in /home/oliver/Documents/data/h36m/
    - OpenCV, MediaPipe, matplotlib, numpy, h5py
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import random

# MediaPipe imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# H36M dataset paths
DATA_PATH = "/home/oliver/Documents/data/h36m"
ANNOT_DIR = os.path.join(DATA_PATH, "annot")
IMAGES_DIR = os.path.join(DATA_PATH, "images")

# H36M skeleton connectivity (17 joints)
H36M_SKELETON = [
    # Right leg
    (0, 1), (1, 2), (2, 3),
    # Left leg
    (0, 4), (4, 5), (5, 6),
    # Spine to head chain
    (0, 7), (7, 8), (8, 9), (9, 10),
    # Left arm chain
    (11, 12), (12, 13),
    # Right arm chain
    (14, 15), (15, 16),
    # Shoulders to thorax
    (11, 8), (8, 14)
]


def mediapipe_to_h36m(mp_keypoints):
    """
    Convert MediaPipe 33 keypoints to H36M 17 keypoints format.
    
    Args:
        mp_keypoints: numpy array of shape (33, 3) with MediaPipe 3D keypoints
        
    Returns:
        h36m_keypoints: numpy array of shape (17, 3) with H36M format keypoints
    """
    h36m_keypoints = np.zeros((17, 3))
    
    # H36M Index 0: Hip (Root) - Midpoint of left and right hips
    h36m_keypoints[0] = (mp_keypoints[23] + mp_keypoints[24]) / 2
    
    # H36M Index 1: Right Hip - MediaPipe 24
    h36m_keypoints[1] = mp_keypoints[24]
    
    # H36M Index 2: Right Knee - MediaPipe 26
    h36m_keypoints[2] = mp_keypoints[26]
    
    # H36M Index 3: Right Ankle - MediaPipe 28
    h36m_keypoints[3] = mp_keypoints[28]
    
    # H36M Index 4: Left Hip - MediaPipe 23
    h36m_keypoints[4] = mp_keypoints[23]
    
    # H36M Index 5: Left Knee - MediaPipe 25
    h36m_keypoints[5] = mp_keypoints[25]
    
    # H36M Index 6: Left Ankle - MediaPipe 27
    h36m_keypoints[6] = mp_keypoints[27]
    
    # H36M Index 7: Spine - Center between hips and shoulders
    h36m_keypoints[7] = (mp_keypoints[23] + mp_keypoints[24] + mp_keypoints[11] + mp_keypoints[12]) / 4
    
    # H36M Index 8: Thorax - Midpoint between shoulders
    h36m_keypoints[8] = (mp_keypoints[11] + mp_keypoints[12]) / 2
    
    # H36M Index 9: Nose - MediaPipe 0
    h36m_keypoints[9] = mp_keypoints[0]
    
    # H36M Index 10: Head - Midpoint between eyes
    h36m_keypoints[10] = (mp_keypoints[2] + mp_keypoints[5]) / 2
    
    # H36M Index 11: Left Shoulder - MediaPipe 11
    h36m_keypoints[11] = mp_keypoints[11]
    
    # H36M Index 12: Left Elbow - MediaPipe 13
    h36m_keypoints[12] = mp_keypoints[13]
    
    # H36M Index 13: Left Wrist - MediaPipe 15
    h36m_keypoints[13] = mp_keypoints[15]
    
    # H36M Index 14: Right Shoulder - MediaPipe 12
    h36m_keypoints[14] = mp_keypoints[12]
    
    # H36M Index 15: Right Elbow - MediaPipe 14
    h36m_keypoints[15] = mp_keypoints[14]
    
    # H36M Index 16: Right Wrist - MediaPipe 16
    h36m_keypoints[16] = mp_keypoints[16]
    
    return h36m_keypoints


def build_image_lookup(root_dir):
    """Build a dict mapping each image basename to its absolute path."""
    lookup = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[fname] = os.path.join(dirpath, fname)
    return lookup


def load_image_list_and_annotations(split="train"):
    """Load H36M image list and annotations."""
    if split == "train":
        img_list_file = os.path.join(ANNOT_DIR, "train_images.txt")
        h5_file = os.path.join(ANNOT_DIR, "train.h5")
    else:
        img_list_file = os.path.join(ANNOT_DIR, "valid_images.txt")
        h5_file = os.path.join(ANNOT_DIR, "valid.h5")

    with open(img_list_file, "r") as f:
        image_names = [line.strip() for line in f.readlines()]

    with h5py.File(h5_file, "r") as f:
        raw_2d = f["part"][:]
        raw_3d = f["S"][:]

    # Handle different data formats
    if raw_2d.ndim == 3 and raw_2d.shape[1] == 2 and raw_2d.shape[2] == 17:
        joints_2d = raw_2d.transpose(0, 2, 1).copy()
    else:
        joints_2d = raw_2d.copy()

    if raw_3d.ndim == 3 and raw_3d.shape[1] == 3 and raw_3d.shape[2] == 17:
        joints_3d = raw_3d.transpose(0, 2, 1).copy()
    else:
        joints_3d = raw_3d.copy()

    return image_names, joints_2d, joints_3d


def rotate_x_minus90(joints_3d):
    """Rotate 3D points by -90 degrees about X-axis."""
    x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    return np.stack([x, z, -y], axis=1)


class MediaPipeH36MConverter:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize MediaPipe pose landmarker."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def plot_2d_comparison(self, ax, image, h36m_keypoints, mp_converted_2d, title):
        """Plot 2D keypoints comparison between H36M GT and converted MediaPipe."""
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(title)
        
        # Plot H36M ground truth (blue)
        if h36m_keypoints is not None:
            xs, ys = h36m_keypoints[:, 0], h36m_keypoints[:, 1]
            ax.scatter(xs, ys, c="blue", s=40, alpha=0.8, label="H36M GT", marker='o')
            
            # Plot H36M skeleton
            for (p, c) in H36M_SKELETON:
                ax.plot([xs[p], xs[c]], [ys[p], ys[c]], c="blue", linewidth=2, alpha=0.7)
        
        # Plot converted MediaPipe (red)
        if mp_converted_2d is not None:
            xs, ys = mp_converted_2d[:, 0], mp_converted_2d[:, 1]
            ax.scatter(xs, ys, c="red", s=30, alpha=0.8, label="MediaPipe Converted", marker='^')
            
            # Plot converted skeleton
            for (p, c) in H36M_SKELETON:
                ax.plot([xs[p], xs[c]], [ys[p], ys[c]], c="red", linewidth=2, alpha=0.7)
        
        ax.legend()
    
    def plot_3d_comparison(self, ax3d, h36m_keypoints, mp_converted_3d, title):
        """Plot 3D keypoints comparison between H36M GT and converted MediaPipe."""
        ax3d.clear()
        ax3d.set_title(title)
        ax3d.set_xlabel("X (mm)")
        ax3d.set_ylabel("Y (mm)") 
        ax3d.set_zlabel("Z (mm)")
        
        # Plot H36M ground truth (blue)
        if h36m_keypoints is not None:
            # Convert to millimeters and center
            if np.max(np.abs(h36m_keypoints)) < 10:
                h36m_mm = h36m_keypoints * 1000.0
            else:
                h36m_mm = h36m_keypoints
            
            h36m_centered = h36m_mm - h36m_mm[0]  # Center at pelvis
            h36m_rotated = rotate_x_minus90(h36m_centered)  # Apply H36M rotation
            
            # Plot joints
            ax3d.scatter(h36m_rotated[:, 0], h36m_rotated[:, 1], h36m_rotated[:, 2], 
                        c="blue", s=60, alpha=0.8, label="H36M GT", marker='o')
            
            # Plot skeleton
            for (p, c) in H36M_SKELETON:
                ax3d.plot([h36m_rotated[p, 0], h36m_rotated[c, 0]],
                         [h36m_rotated[p, 1], h36m_rotated[c, 1]],
                         [h36m_rotated[p, 2], h36m_rotated[c, 2]], 
                         c="blue", linewidth=3, alpha=0.7)
        
        # Plot converted MediaPipe (red)
        if mp_converted_3d is not None:
            # Scale to similar range as H36M
            mp_scaled = mp_converted_3d * 1000  # Convert to mm scale
            
            # Plot joints
            ax3d.scatter(mp_scaled[:, 0], mp_scaled[:, 1], mp_scaled[:, 2], 
                        c="red", s=40, alpha=0.8, label="MediaPipe Converted", marker='^')
            
            # Plot skeleton
            for (p, c) in H36M_SKELETON:
                ax3d.plot([mp_scaled[p, 0], mp_scaled[c, 0]],
                         [mp_scaled[p, 1], mp_scaled[c, 1]],
                         [mp_scaled[p, 2], mp_scaled[c, 2]], 
                         c="red", linewidth=2, alpha=0.7)
        
        # Set equal aspect ratio
        all_points = []
        if h36m_keypoints is not None:
            all_points.extend(h36m_rotated)
        if mp_converted_3d is not None:
            all_points.extend(mp_scaled)
        
        if all_points:
            all_points = np.array(all_points)
            xyz_min = np.min(all_points, axis=0)
            xyz_max = np.max(all_points, axis=0)
            max_range = np.max(xyz_max - xyz_min) / 2.0
            
            if max_range > 0:
                mid_x = (xyz_max[0] + xyz_min[0]) / 2.0
                mid_y = (xyz_max[1] + xyz_min[1]) / 2.0
                mid_z = (xyz_max[2] + xyz_min[2]) / 2.0
                
                ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
                ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
                ax3d.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax3d.legend()
    
    def visualize_comparison(self, image_path, h36m_2d, h36m_3d, idx):
        """Visualize converted MediaPipe vs H36M comparison for a single image."""
        # Load and process image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"Could not load image: {image_path}")
            return
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Get MediaPipe results
        result = self.landmarker.detect(mp_image)
        
        if not result.pose_world_landmarks:
            print(f"No pose detected in image {idx}")
            return
        
        # Extract MediaPipe 3D coordinates and apply coordinate conversion
        mp_keypoints_raw = np.array([[lm.x, -lm.z, -lm.y] for lm in result.pose_world_landmarks[0]])
        
        # Apply coordinate correction: (x, y, z) -> (x, -y, z)
        mp_keypoints_raw[:, 1] = -mp_keypoints_raw[:, 1]
        
        # Convert MediaPipe to H36M format
        mp_converted_3d = mediapipe_to_h36m(mp_keypoints_raw)
        
        # Project 3D to 2D for visualization (simple orthographic projection)
        h, w = image_rgb.shape[:2]
        if result.pose_landmarks:
            # Get 2D landmarks and convert to H36M format
            mp_2d_raw = np.array([[lm.x * w, lm.y * h, 0] for lm in result.pose_landmarks[0]])
            mp_converted_2d = mediapipe_to_h36m(mp_2d_raw)[:, :2]  # Take only x, y
        else:
            mp_converted_2d = None
        
        # Create figure with 2x2 subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Top left: 2D comparison
        ax2d = fig.add_subplot(2, 2, 1)
        self.plot_2d_comparison(ax2d, image_rgb, h36m_2d, mp_converted_2d, 
                               f"2D Pose Comparison (H36M Format) - Sample {idx}")
        
        # Top right: H36M 3D ground truth only
        ax3d_h36m = fig.add_subplot(2, 2, 2, projection="3d")
        self.plot_3d_comparison(ax3d_h36m, h36m_3d, None, 
                               f"H36M 3D Ground Truth - Sample {idx}")
        
        # Bottom left: MediaPipe converted to H36M format only
        ax3d_mp = fig.add_subplot(2, 2, 3, projection="3d")
        self.plot_3d_comparison(ax3d_mp, None, mp_converted_3d, 
                               f"MediaPipe Converted to H36M - Sample {idx}")
        
        # Bottom right: 3D comparison overlay
        ax3d_both = fig.add_subplot(2, 2, 4, projection="3d")
        self.plot_3d_comparison(ax3d_both, h36m_3d, mp_converted_3d, 
                               f"3D Comparison (H36M Format) - Sample {idx}")
        
        plt.tight_layout()
        plt.show()
    
    def run_comparison(self, num_samples=3, split="valid"):
        """Run comparison visualization on multiple samples."""
        print(f"Loading H36M {split} split...")
        image_names, joints_2d_all, joints_3d_all = load_image_list_and_annotations(split)
        
        print("Building image lookup...")
        image_lookup = build_image_lookup(IMAGES_DIR)
        
        # Select random samples
        num_samples = min(num_samples, len(image_names))
        chosen_indices = random.sample(range(len(image_names)), num_samples)
        
        for idx in chosen_indices:
            relpath = image_names[idx]
            basename = os.path.basename(relpath)
            fullpath = image_lookup.get(basename)
            
            if fullpath is None or not os.path.isfile(fullpath):
                print(f"WARNING: cannot find image for '{relpath}'. Skipping index {idx}.")
                continue
            
            h36m_2d = joints_2d_all[idx].astype(np.float32)
            h36m_3d = joints_3d_all[idx].astype(np.float32)
            
            print(f"Processing sample {idx}: {basename}")
            self.visualize_comparison(fullpath, h36m_2d, h36m_3d, idx)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(description='Compare MediaPipe (converted to H36M) vs H36M ground truth')
    parser.add_argument('--num_samples', type=int, default=3, help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid'], 
                       help='H36M split to use')
    parser.add_argument('--model', type=str, default='pose_landmarker.task', 
                       help='Path to MediaPipe model')
    
    args = parser.parse_args()
    
    try:
        converter = MediaPipeH36MConverter(model_path=args.model)
        converter.run_comparison(num_samples=args.num_samples, split=args.split)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'converter' in locals():
            converter.cleanup()


if __name__ == "__main__":
    main()