#!/usr/bin/env python3
"""
Test script to run inference on H36M images using the simplified infer_realsense.py
with the fixed coordinate system (Z-axis pointing up).
Includes MediaPipe H36M converted poses for comparison.
"""

import os
import sys
import random
import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

# Add the current directory to path
sys.path.insert(0, '/home/oliver/Documents/MotionBERT')

# No MotionBERT imports needed

# H36M dataset configuration
DATA_PATH = "/home/oliver/Documents/data/h36m"
ANNOT_DIR = os.path.join(DATA_PATH, "annot")
IMAGES_DIR = os.path.join(DATA_PATH, "images")
SPLIT = "valid"  # Use validation split
NUM_SAMPLES = 3  # Number of images to test

# H36M skeleton connections
H36M_SKELETON = [
    (0, 1), (1, 2), (2, 3),     # Right leg
    (0, 4), (4, 5), (5, 6),     # Left leg  
    (0, 7), (7, 8), (8, 9), (9, 10),  # Spine to head
    (8, 11), (11, 12), (12, 13),      # Left arm
    (8, 14), (14, 15), (15, 16),      # Right arm
]

def build_image_lookup(root_dir):
    """Build a lookup dictionary mapping image basename to full path."""
    lookup = {}
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                lookup[fname] = os.path.join(dirpath, fname)
    return lookup

def mediapipe_to_h36m(mp_keypoints):
    """
    Convert MediaPipe 33 keypoints to H36M 17 keypoints format.
    Based on the conversion from visualize_mediapipe_h36m_converted.py
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

def load_image_list_and_annotations(split="valid"):
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

    # Transpose to (N,17,2) and (N,17,3) format
    if raw_2d.ndim == 3 and raw_2d.shape[1] == 2 and raw_2d.shape[2] == 17:
        joints_2d = raw_2d.transpose(0, 2, 1).copy()
    else:
        joints_2d = raw_2d.copy()

    if raw_3d.ndim == 3 and raw_3d.shape[1] == 3 and raw_3d.shape[2] == 17:
        joints_3d = raw_3d.transpose(0, 2, 1).copy()
    else:
        joints_3d = raw_3d.copy()

    return image_names, joints_2d, joints_3d

def get_mediapipe_pose(image_path):
    """Get MediaPipe pose estimation for an image."""
    if not MEDIAPIPE_AVAILABLE:
        return None
    
    # Check if MediaPipe model exists
    model_path = 'pose_landmarker.task'
    if not os.path.exists(model_path):
        print(f"MediaPipe model not found at {model_path}")
        return None
    
    try:
        # Initialize MediaPipe
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        landmarker = PoseLandmarker.create_from_options(options)
        
        # Load and process image
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Get MediaPipe results
        result = landmarker.detect(mp_image)
        landmarker.close()
        
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
        print(f"Error in MediaPipe pose estimation: {e}")
        return None

def extract_mmpose_3d_keypoints(mmpose_inferencer, image, model_name):
    """Extract 3D keypoints from MMPose inferencer results."""
    if mmpose_inferencer is None:
        return None
    
    try:
        # Run MMPose inference using the new API
        results_gen = mmpose_inferencer(image, show=False, out_dir=None)
        results_list = list(results_gen)
        
        if len(results_list) > 0:
            result = results_list[0]
            
            # Extract predictions
            if 'predictions' in result:
                predictions = result['predictions']
                if len(predictions) > 0:
                    persons_list = predictions[0]
                    if len(persons_list) > 0:
                        person = persons_list[0]
                        
                        # Extract 3D keypoints
                        if 'keypoints' in person:
                            keypoints_list = person['keypoints']
                            
                            # Convert to numpy array (3D keypoints)
                            keypoints_3d = np.zeros((len(keypoints_list), 3))
                            for i, kpt in enumerate(keypoints_list):
                                if len(kpt) >= 3:
                                    keypoints_3d[i, 0] = kpt[0]  # x
                                    keypoints_3d[i, 1] = kpt[1]  # y
                                    keypoints_3d[i, 2] = kpt[2]  # z
                            
                            print(f"✅ {model_name} pose estimated with shape: {keypoints_3d.shape}")
                            return keypoints_3d
        
        print(f"❌ {model_name} pose estimation failed")
        return None
        
    except Exception as e:
        print(f"❌ Error in {model_name} estimation: {e}")
        return None

def plot_comparison(image, gt_3d, mp_3d, mmpose_videopose3d, mmpose_simplebaseline3d, mmpose_motionbert, idx):
    """Plot comparison between ground truth, MediaPipe, and three MMPose 3D models."""
    fig = plt.figure(figsize=(18, 12))
    
    # Original image
    ax1 = fig.add_subplot(3, 2, 1)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f"H36M Image {idx}")
    ax1.axis('off')
    
    # Ground truth 3D pose (using visualize_h36m.py coordinate system)
    ax2 = fig.add_subplot(3, 2, 2, projection='3d')
    plot_3d_skeleton(ax2, gt_3d, "Ground Truth 3D (H36M)", color='blue', is_ground_truth=True)
    
    # MediaPipe converted 3D pose
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')
    if mp_3d is not None:
        plot_3d_skeleton(ax3, mp_3d, "MediaPipe 3D (H36M format)", color='green', is_mediapipe=True)
    else:
        ax3.text(0.5, 0.5, 0.5, 'MediaPipe\nNot Available', ha='center', va='center', fontsize=12)
        ax3.set_title("MediaPipe 3D (H36M format)")
    
    # MMPose Human3D
    ax4 = fig.add_subplot(3, 2, 4, projection='3d')
    if mmpose_videopose3d is not None:
        plot_3d_skeleton(ax4, mmpose_videopose3d, "MMPose Human3D", color='purple', is_mmpose=True)
    else:
        ax4.set_title("MMPose Human3D")
    
    # MMPose SimpleBaseline3D
    ax5 = fig.add_subplot(3, 2, 5, projection='3d')
    if mmpose_simplebaseline3d is not None:
        plot_3d_skeleton(ax5, mmpose_simplebaseline3d, "MMPose SimpleBaseline3D", color='orange', is_mmpose=True)
    else:
        ax5.set_title("MMPose SimpleBaseline3D")
    
    # MMPose MotionBERT
    ax6 = fig.add_subplot(3, 2, 6, projection='3d')
    if mmpose_motionbert is not None:
        plot_3d_skeleton(ax6, mmpose_motionbert, "MMPose MotionBERT", color='red', is_mmpose=True)
    else:
        ax6.set_title("MMPose MotionBERT")
    
    plt.tight_layout()
    plt.show()

def rotate_x_minus90(joints_3d):
    """
    Rotate the 3D points by -90 degrees about the X-axis.
    Rotation: (x, y, z) → (x, z, -y)
    """
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2]
    x_new = x
    y_new = z
    z_new = -y
    return np.stack([x_new, y_new, z_new], axis=1)

def rotate_z_plus90(joints_3d):
    """
    Rotate the 3D points by +90 degrees about the Z-axis.
    Rotation: (x, y, z) → (-y, x, z)
    """
    x = joints_3d[:, 0]
    y = joints_3d[:, 1]
    z = joints_3d[:, 2]
    x_new = -y
    y_new = x
    z_new = z
    return np.stack([x_new, y_new, z_new], axis=1)

def rotate_combined(joints_3d):
    """
    Apply first rotation by -90° about X, then +90° about Z.
    This matches the visualize_h36m.py transformation.
    """
    rotated_x = rotate_x_minus90(joints_3d)
    return rotated_x  # visualize_h36m.py only applies X rotation in rotate_combined

def plot_3d_skeleton(ax, joints_3d, title, color='red', is_ground_truth=False, is_mediapipe=False, is_mmpose=False):
    """Plot 3D skeleton on given axis."""
    # Convert to mm if in meters and center at pelvis
    if np.max(np.abs(joints_3d)) < 10:
        joints_3d = joints_3d * 1000.0
    joints_centered = joints_3d - joints_3d[0]
    
    # Apply coordinate transformations
    if is_ground_truth:
        # Apply visualize_h36m.py coordinate transformation for ground truth
        joints_centered = rotate_combined(joints_centered)
    elif is_mediapipe:
        # For MediaPipe: don't apply H36M rotation, it's already in the right coordinate system
        # Just scale it to similar range as H36M (already done by multiplying by 1000 above)
        pass
    elif is_mmpose:
        # For MMPose: apply X-axis negation (from test_mmpose_h36m.py)
        joints_centered[:, 0] = -joints_centered[:, 0]
    
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
    xyz_min = np.min(joints_centered, axis=0)
    xyz_max = np.max(joints_centered, axis=0)
    max_range = np.max(xyz_max - xyz_min) / 2.0
    mid_x = (xyz_max[0] + xyz_min[0]) / 2.0
    mid_y = (xyz_max[1] + xyz_min[1]) / 2.0
    mid_z = (xyz_max[2] + xyz_min[2]) / 2.0
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

def main():
    # Check if H36M dataset exists
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: H36M dataset not found at {DATA_PATH}")
        return
    
    # No MotionBERT initialization needed
    
    # Initialize MMPose 3D inferencers (without camera)
    from mmpose.apis import MMPoseInferencer
    device = 'cuda:0' if 'cuda' in 'cuda:0' else 'cpu'
    
    mmpose_videopose3d = None
    mmpose_simplebaseline3d = None
    mmpose_motionbert = None
    try:
        mmpose_videopose3d = MMPoseInferencer(
            pose3d='configs/body_3d_keypoint/video_pose_lift/h36m/video-pose-lift_tcn-243frm-supv_8xb128-160e_h36m.py', 
            pose3d_weights='https://download.openmmlab.com/mmpose/body3d/videopose/videopose_h36m_243frames_fullconv_supervised-880bea25_20210527.pth',
            device=device
        )
        print("✅ MMPose VideoPose3D inferencer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize MMPose VideoPose3D: {e}")
    
    try:
        mmpose_simplebaseline3d = MMPoseInferencer(
            pose3d='configs/body_3d_keypoint/image_pose_lift/h36m/image-pose-lift_tcn_8xb64-200e_h36m.py', 
            pose3d_weights='https://download.openmmlab.com/mmpose/body3d/simple_baseline/simple3Dbaseline_h36m-f0ad73a4_20210419.pth',
            device=device
        )
        print("✅ MMPose SimpleBaseline3D inferencer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MMPose SimpleBaseline3D: {e}")
    
    try:
        mmpose_motionbert = MMPoseInferencer(
            pose3d='configs/body_3d_keypoint/motionbert/h36m/motionbert_dstformer-ft-243frm_8xb32-120e_h36m.py',
            pose3d_weights='https://download.openmmlab.com/mmpose/v1/body_3d_keypoint/pose_lift/h36m/motionbert_ft_h36m-d80af323_20230531.pth',
            device=device
        )
        print("✅ MMPose MotionBERT inferencer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize MMPose MotionBERT: {e}")
    
    # Load H36M data
    print("Loading H36M dataset...")
    image_lookup = build_image_lookup(IMAGES_DIR)
    image_names, joints_2d_all, joints_3d_all = load_image_list_and_annotations(SPLIT)
    
    # Select random samples
    num_samples = min(NUM_SAMPLES, len(image_names))
    chosen_indices = random.sample(range(len(image_names)), num_samples)
    
    print(f"Testing inference on {num_samples} H36M images...")
    
    for i, idx in enumerate(chosen_indices):
        relpath = image_names[idx]
        basename = os.path.basename(relpath)
        fullpath = image_lookup.get(basename)
        
        if fullpath is None or not os.path.isfile(fullpath):
            print(f"WARNING: Image not found for '{relpath}'. Skipping...")
            continue
        
        print(f"\nProcessing image {i+1}/{num_samples}: {basename}")
        
        # Load image
        image = cv2.imread(fullpath)
        if image is None:
            print(f"Failed to load image: {fullpath}")
            continue
        
        # Get ground truth 3D pose
        gt_3d = joints_3d_all[idx].astype(np.float32)
        
        # Get MediaPipe pose estimation
        print("Running MediaPipe pose estimation...")
        mp_3d = get_mediapipe_pose(fullpath)
        if mp_3d is not None:
            print(f"✅ MediaPipe pose estimated with shape: {mp_3d.shape}")
        else:
            print("❌ MediaPipe pose estimation failed")
        
        print("Running MMPose 3D pose estimations...")
        mmpose_vp3d_result = extract_mmpose_3d_keypoints(mmpose_videopose3d, image, "Human3D (VideoPose3D)")
        mmpose_sb3d_result = extract_mmpose_3d_keypoints(mmpose_simplebaseline3d, image, "Human3D (SimpleBaseline3D)")
        mmpose_mb_result = extract_mmpose_3d_keypoints(mmpose_motionbert, image, "Human3D (MotionBERT)")
        
        # Plot comparison
        plot_comparison(image, gt_3d, mp_3d, mmpose_vp3d_result, mmpose_sb3d_result, mmpose_mb_result, idx)
        
        # Print some statistics
        print(f"Ground truth range: X[{gt_3d[:,0].min():.3f}, {gt_3d[:,0].max():.3f}] "
              f"Y[{gt_3d[:,1].min():.3f}, {gt_3d[:,1].max():.3f}] "
              f"Z[{gt_3d[:,2].min():.3f}, {gt_3d[:,2].max():.3f}]")
        if mp_3d is not None:
            print(f"MediaPipe range:     X[{mp_3d[:,0].min():.3f}, {mp_3d[:,0].max():.3f}] "
                  f"Y[{mp_3d[:,1].min():.3f}, {mp_3d[:,1].max():.3f}] "
                  f"Z[{mp_3d[:,2].min():.3f}, {mp_3d[:,2].max():.3f}]")
        if mmpose_vp3d_result is not None:
            print(f"VideoPose3D:    X[{mmpose_vp3d_result[:,0].min():.3f}, {mmpose_vp3d_result[:,0].max():.3f}] "
                  f"Y[{mmpose_vp3d_result[:,1].min():.3f}, {mmpose_vp3d_result[:,1].max():.3f}] "
                  f"Z[{mmpose_vp3d_result[:,2].min():.3f}, {mmpose_vp3d_result[:,2].max():.3f}]")
        if mmpose_sb3d_result is not None:
            print(f"SimpleBaseline3D:    X[{mmpose_sb3d_result[:,0].min():.3f}, {mmpose_sb3d_result[:,0].max():.3f}] "
                  f"Y[{mmpose_sb3d_result[:,1].min():.3f}, {mmpose_sb3d_result[:,1].max():.3f}] "
                  f"Z[{mmpose_sb3d_result[:,2].min():.3f}, {mmpose_sb3d_result[:,2].max():.3f}]")
        if mmpose_mb_result is not None:
            print(f"MotionBERT:    X[{mmpose_mb_result[:,0].min():.3f}, {mmpose_mb_result[:,0].max():.3f}] "
                  f"Y[{mmpose_mb_result[:,1].min():.3f}, {mmpose_mb_result[:,1].max():.3f}] "
                  f"Z[{mmpose_mb_result[:,2].min():.3f}, {mmpose_mb_result[:,2].max():.3f}]")
    
    # Cleanup MMPose inferencer (no cleanup needed for MMPoseInferencer)
    print("✅ Test completed successfully")

if __name__ == "__main__":
    main()