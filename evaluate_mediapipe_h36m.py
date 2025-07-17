#!/usr/bin/env python3
"""
evaluate_mediapipe_h36m.py

Script to evaluate MediaPipe 3D pose estimation on 1000 H36M images.
Calculates average error for each keypoint against ground truth using
the latest MediaPipe to H36M conversion mapping.

Usage:
    python evaluate_mediapipe_h36m.py [--num_samples 1000] [--split valid]

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
from tqdm import tqdm

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

# H36M joint names for plotting
H36M_JOINT_NAMES = [
    "Hip (Root)",      # 0
    "Right Hip",       # 1
    "Right Knee",      # 2
    "Right Ankle",     # 3
    "Left Hip",        # 4
    "Left Knee",       # 5
    "Left Ankle",      # 6
    "Spine",           # 7
    "Thorax",          # 8
    "Nose",            # 9
    "Head",            # 10
    "Left Shoulder",   # 11
    "Left Elbow",      # 12
    "Left Wrist",      # 13
    "Right Shoulder",  # 14
    "Right Elbow",     # 15
    "Right Wrist"      # 16
]


def mediapipe_to_h36m(mp_keypoints):
    """
    Convert MediaPipe 33 keypoints to H36M 17 keypoints format.
    Uses the latest conversion mapping from mediapipe_h36m.txt.
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
    """Rotate 3D points by -90 degrees about X-axis (H36M convention)."""
    x, y, z = joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2]
    return np.stack([x, z, -y], axis=1)


def calculate_3d_error(pred_3d, gt_3d):
    """Calculate 3D Euclidean distance error between predicted and ground truth keypoints."""
    return np.sqrt(np.sum((pred_3d - gt_3d) ** 2, axis=1))


class MediaPipeH36MEvaluator:
    def __init__(self, model_path='pose_landmarker.task'):
        """Initialize MediaPipe pose landmarker."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe model file not found: {model_path}")
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
    
    def process_image(self, image_path):
        """Process a single image and return converted MediaPipe keypoints."""
        try:
            # Load and process image
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                return None
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            
            # Get MediaPipe results
            result = self.landmarker.detect(mp_image)
            
            if not result.pose_world_landmarks:
                return None
            
            # Extract MediaPipe 3D coordinates and apply coordinate conversion
            mp_keypoints_raw = np.array([[lm.x, -lm.z, -lm.y] for lm in result.pose_world_landmarks[0]])
            
            # Apply coordinate correction: (x, y, z) -> (x, -y, z)
            mp_keypoints_raw[:, 1] = -mp_keypoints_raw[:, 1]
            
            # Convert MediaPipe to H36M format
            mp_converted_3d = mediapipe_to_h36m(mp_keypoints_raw)
            
            return mp_converted_3d
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def evaluate_dataset(self, num_samples=1000, split="valid"):
        """Evaluate MediaPipe on H36M dataset."""
        print(f"Loading H36M {split} split...")
        image_names, joints_2d_all, joints_3d_all = load_image_list_and_annotations(split)
        
        print("Building image lookup...")
        image_lookup = build_image_lookup(IMAGES_DIR)
        
        # Select random samples
        num_samples = min(num_samples, len(image_names))
        chosen_indices = random.sample(range(len(image_names)), num_samples)
        
        print(f"Evaluating on {num_samples} samples...")
        
        # Storage for results
        all_errors = []
        valid_samples = 0
        
        # Process each image
        for idx in tqdm(chosen_indices, desc="Processing images"):
            relpath = image_names[idx]
            basename = os.path.basename(relpath)
            fullpath = image_lookup.get(basename)
            
            if fullpath is None or not os.path.isfile(fullpath):
                continue
            
            # Get ground truth
            h36m_3d = joints_3d_all[idx].astype(np.float32)
            
            # Process with MediaPipe
            mp_converted_3d = self.process_image(fullpath)
            
            if mp_converted_3d is None:
                continue
            
            # Prepare ground truth (convert to mm and apply H36M rotation)
            if np.max(np.abs(h36m_3d)) < 10:
                h36m_mm = h36m_3d * 1000.0
            else:
                h36m_mm = h36m_3d
            
            h36m_centered = h36m_mm - h36m_mm[0]  # Center at pelvis
            h36m_rotated = rotate_x_minus90(h36m_centered)
            
            # Prepare MediaPipe prediction (convert to mm)
            mp_mm = mp_converted_3d * 1000.0
            
            # Calculate 3D error for each joint
            errors = calculate_3d_error(mp_mm, h36m_rotated)
            all_errors.append(errors)
            valid_samples += 1
        
        if valid_samples == 0:
            print("No valid samples found!")
            return None, None
        
        # Calculate average errors
        all_errors = np.array(all_errors)
        avg_errors = np.mean(all_errors, axis=0)
        std_errors = np.std(all_errors, axis=0)
        
        print(f"Successfully processed {valid_samples} out of {num_samples} samples")
        return avg_errors, std_errors
    
    def plot_results(self, avg_errors, std_errors, save_path=None):
        """Plot the average error for each keypoint."""
        if avg_errors is None:
            return
        
        plt.figure(figsize=(15, 8))
        
        # Create bar plot
        x_pos = np.arange(len(H36M_JOINT_NAMES))
        bars = plt.bar(x_pos, avg_errors, yerr=std_errors, capsize=5, alpha=0.7, color='skyblue')
        
        # Customize plot
        plt.xlabel('H36M Keypoints', fontsize=12)
        plt.ylabel('Average 3D Error (mm)', fontsize=12)
        plt.title('MediaPipe vs H36M Ground Truth: Average 3D Error per Keypoint', fontsize=14)
        plt.xticks(x_pos, H36M_JOINT_NAMES, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, error, std) in enumerate(zip(bars, avg_errors, std_errors)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
                    f'{error:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Add summary statistics
        overall_avg = np.mean(avg_errors)
        plt.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Average: {overall_avg:.1f} mm')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
        
        # Print detailed results
        print("\n" + "="*60)
        print("DETAILED RESULTS:")
        print("="*60)
        for i, (joint_name, error, std) in enumerate(zip(H36M_JOINT_NAMES, avg_errors, std_errors)):
            print(f"{i:2d}. {joint_name:<15}: {error:6.1f} ± {std:5.1f} mm")
        print("-"*60)
        print(f"Overall Average: {overall_avg:.1f} mm")
        print(f"Best Joint: {H36M_JOINT_NAMES[np.argmin(avg_errors)]} ({np.min(avg_errors):.1f} mm)")
        print(f"Worst Joint: {H36M_JOINT_NAMES[np.argmax(avg_errors)]} ({np.max(avg_errors):.1f} mm)")
        print("="*60)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'landmarker'):
            self.landmarker.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate MediaPipe pose estimation on H36M dataset')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to evaluate')
    parser.add_argument('--split', type=str, default='valid', choices=['train', 'valid'], 
                       help='H36M split to use')
    parser.add_argument('--model', type=str, default='pose_landmarker.task', 
                       help='Path to MediaPipe model')
    parser.add_argument('--save_plot', type=str, default='mediapipe_h36m_evaluation.png',
                       help='Path to save the plot')
    
    args = parser.parse_args()
    
    try:
        evaluator = MediaPipeH36MEvaluator(model_path=args.model)
        avg_errors, std_errors = evaluator.evaluate_dataset(num_samples=args.num_samples, split=args.split)
        evaluator.plot_results(avg_errors, std_errors, save_path=args.save_plot)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'evaluator' in locals():
            evaluator.cleanup()


if __name__ == "__main__":
    main()