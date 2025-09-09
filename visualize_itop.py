import h5py
import numpy as np
import matplotlib.pyplot as plt


class ITOPVisualizer:
    def __init__(self, dataset_path):
        """
        Initialize ITOP dataset visualizer
        
        Args:
            dataset_path: Path to the ITOP dataset directory
        """
        self.dataset_path = dataset_path
        
        # ITOP dataset has 15 joints
        # Joint indices based on ITOP dataset structure:
        # 0: Head, 1: Neck, 2: L-Shoulder, 3: R-Shoulder, 4: L-Elbow, 5: R-Elbow,
        # 6: L-Hand, 7: R-Hand, 8: Torso, 9: L-Hip, 10: R-Hip, 11: L-Knee, 
        # 12: R-Knee, 13: L-Foot, 14: R-Foot
        
        # Define skeleton connections (bones) between joints
        self.skeleton_connections = [
            (0, 1),   # Head - Neck
            (1, 2),   # Neck - L-Shoulder
            (1, 3),   # Neck - R-Shoulder
            (2, 4),   # L-Shoulder - L-Elbow
            (3, 5),   # R-Shoulder - R-Elbow
            (4, 6),   # L-Elbow - L-Hand
            (5, 7),   # R-Elbow - R-Hand
            (1, 8),   # Neck - Torso
            (8, 9),   # Torso - L-Hip
            (8, 10),  # Torso - R-Hip
            (9, 11),  # L-Hip - L-Knee
            (10, 12), # R-Hip - R-Knee
            (11, 13), # L-Knee - L-Foot
            (12, 14)  # R-Knee - R-Foot
        ]
        
        # Joint names for reference
        self.joint_names = [
            'Head', 'Neck', 'L-Shoulder', 'R-Shoulder', 'L-Elbow', 'R-Elbow',
            'L-Hand', 'R-Hand', 'Torso', 'L-Hip', 'R-Hip', 'L-Knee',
            'R-Knee', 'L-Foot', 'R-Foot'
        ]
        
        # Colors for different body parts
        self.joint_colors = [
            'red',      # Head
            'orange',   # Neck
            'blue',     # L-Shoulder
            'green',    # R-Shoulder
            'blue',     # L-Elbow
            'green',    # R-Elbow
            'blue',     # L-Hand
            'green',    # R-Hand
            'orange',   # Torso
            'blue',     # L-Hip
            'green',    # R-Hip
            'blue',     # L-Knee
            'green',    # R-Knee
            'blue',     # L-Foot
            'green'     # R-Foot
        ]
    
    def load_data(self, view='side', split='train', max_samples=None):
        """
        Load ITOP dataset
        
        Args:
            view: 'side' or 'top'
            split: 'train' or 'test'
            max_samples: Maximum number of samples to load (None for all)
        
        Returns:
            Dictionary containing loaded data
        """
        filename = f'ITOP_{view}_{split}_labels.h5'
        filepath = f"{self.dataset_path}/{filename}"
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            print(f"Loading {filename}...")
            print(f"Available keys: {list(f.keys())}")
            
            n_samples = len(f['is_valid'])
            if max_samples is not None:
                n_samples = min(n_samples, max_samples)
            
            data['is_valid'] = f['is_valid'][:n_samples]
            data['image_coordinates'] = f['image_coordinates'][:n_samples]  # 2D joints
            data['real_world_coordinates'] = f['real_world_coordinates'][:n_samples]  # 3D joints
            data['visible_joints'] = f['visible_joints'][:n_samples]
            data['segmentation'] = f['segmentation'][:n_samples]
            
            print(f"Loaded {n_samples} samples")
            print(f"Valid samples: {np.sum(data['is_valid'])}")
        
        return data
    
    def visualize_2d_skeleton(self, joints_2d, visible_joints, title="2D Skeleton", 
                            image_size=(320, 240), save_path=None):
        """
        Visualize 2D skeleton
        
        Args:
            joints_2d: 2D joint coordinates (15, 2)
            visible_joints: Visibility of joints (15,)
            title: Plot title
            image_size: Image dimensions (width, height)
            save_path: Path to save the figure (optional)
        """
        plt.figure(figsize=(10, 8))
        
        # Create black background
        plt.xlim(0, image_size[0])
        plt.ylim(0, image_size[1])
        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.gca().set_facecolor('black')
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            joint1_idx, joint2_idx = connection
            if visible_joints[joint1_idx] and visible_joints[joint2_idx]:
                x1, y1 = joints_2d[joint1_idx]
                x2, y2 = joints_2d[joint2_idx]
                plt.plot([x1, x2], [y1, y2], 'w-', linewidth=2, alpha=0.7)
        
        # Draw joints
        for i, (joint, visible) in enumerate(zip(joints_2d, visible_joints)):
            if visible:
                x, y = joint
                plt.scatter(x, y, c=self.joint_colors[i], s=100, zorder=3)
                plt.annotate(f'{i}', (x, y), xytext=(5, 5), 
                           textcoords='offset points', color='white', fontsize=8)
        
        plt.title(title, color='white')
        plt.xlabel('X (pixels)', color='white')
        plt.ylabel('Y (pixels)', color='white')
        plt.grid(True, alpha=0.3)
        
        # Ensure aspect ratio is equal for 2D visualization
        plt.gca().set_aspect('equal', adjustable='box')
        
        if save_path:
            plt.savefig(save_path, facecolor='black', bbox_inches='tight')
        
        plt.show()
    
    def visualize_3d_skeleton(self, joints_3d, visible_joints, title="3D Skeleton", save_path=None):
        """
        Visualize 3D skeleton
        
        Args:
            joints_3d: 3D joint coordinates (15, 3)
            visible_joints: Visibility of joints (15,)
            title: Plot title
            save_path: Path to save the figure (optional)
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            joint1_idx, joint2_idx = connection
            if visible_joints[joint1_idx] and visible_joints[joint2_idx]:
                x1, y1, z1 = joints_3d[joint1_idx]
                x2, y2, z2 = joints_3d[joint2_idx]
                ax.plot([x1, x2], [y1, y2], [z1, z2], 'w-', linewidth=2, alpha=0.7)
        
        # Draw joints
        for i, (joint, visible) in enumerate(zip(joints_3d, visible_joints)):
            if visible:
                x, y, z = joint
                ax.scatter(x, y, z, c=self.joint_colors[i], s=100, zorder=3)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        
        # Set equal aspect ratio with same scale for all axes
        # Get the range of coordinates
        x_coords = [joint[0] for joint, visible in zip(joints_3d, visible_joints) if visible]
        y_coords = [joint[1] for joint, visible in zip(joints_3d, visible_joints) if visible]
        z_coords = [joint[2] for joint, visible in zip(joints_3d, visible_joints) if visible]
        
        if x_coords and y_coords and z_coords:
            # Calculate ranges
            x_range = [min(x_coords), max(x_coords)]
            y_range = [min(y_coords), max(y_coords)]
            z_range = [min(z_coords), max(z_coords)]
            
            # Find the maximum range to use for all axes
            max_range = max(
                x_range[1] - x_range[0],
                y_range[1] - y_range[0], 
                z_range[1] - z_range[0]
            )
            
            # Calculate centers
            x_center = (x_range[0] + x_range[1]) / 2
            y_center = (y_range[0] + y_range[1]) / 2
            z_center = (z_range[0] + z_range[1]) / 2
            
            # Set same range for all axes
            half_range = max_range / 2
            ax.set_xlim(x_center - half_range, x_center + half_range)
            ax.set_ylim(y_center - half_range, y_center + half_range)
            ax.set_zlim(z_center - half_range, z_center + half_range)
        
        # Set equal aspect ratio
        ax.set_box_aspect([1,1,1])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    def visualize_with_depth_image(self, data, sample_idx, view='side', split='train'):
        """
        Visualize skeleton overlaid on depth image
        
        Args:
            data: Loaded dataset
            sample_idx: Sample index to visualize
            view: 'side' or 'top'
            split: 'train' or 'test'
        """
        # Load depth image
        depth_filename = f'ITOP_{view}_{split}_depth_map.h5'
        depth_filepath = f"{self.dataset_path}/{depth_filename}"
        
        try:
            with h5py.File(depth_filepath, 'r') as f:
                depth_image = f['data'][sample_idx]
        except Exception as e:
            print(f"Could not load depth image from {depth_filepath}: {e}")
            return
        
        joints_2d = data['image_coordinates'][sample_idx]
        visible_joints = data['visible_joints'][sample_idx]
        
        plt.figure(figsize=(12, 8))
        
        # Display depth image
        plt.imshow(depth_image, cmap='viridis')
        
        # Draw skeleton connections
        for connection in self.skeleton_connections:
            joint1_idx, joint2_idx = connection
            if visible_joints[joint1_idx] and visible_joints[joint2_idx]:
                x1, y1 = joints_2d[joint1_idx]
                x2, y2 = joints_2d[joint2_idx]
                plt.plot([x1, x2], [y1, y2], 'r-', linewidth=3, alpha=0.8)
        
        # Draw joints
        for i, (joint, visible) in enumerate(zip(joints_2d, visible_joints)):
            if visible:
                x, y = joint
                plt.scatter(x, y, c='red', s=100, zorder=3, edgecolors='white', linewidth=1)
                plt.annotate(self.joint_names[i], (x, y), xytext=(5, 5), 
                           textcoords='offset points', color='white', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        
        plt.title(f'ITOP {view.capitalize()} View - Sample {sample_idx}')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def print_joint_info(self):
        """Print information about joint indices and connections"""
        print("ITOP Dataset Joint Information:")
        print("=" * 40)
        for i, name in enumerate(self.joint_names):
            print(f"Joint {i:2d}: {name}")
        
        print("\nSkeleton Connections:")
        print("=" * 40)
        for i, (j1, j2) in enumerate(self.skeleton_connections):
            print(f"Connection {i:2d}: {self.joint_names[j1]} - {self.joint_names[j2]}")
    
    def visualize_samples(self, view='side', split='train', num_samples=5, start_idx=0):
        """
        Visualize multiple samples from the dataset
        
        Args:
            view: 'side' or 'top'
            split: 'train' or 'test'
            num_samples: Number of samples to visualize
            start_idx: Starting sample index
        """
        print(f"Visualizing {num_samples} samples from ITOP {view} {split} dataset...")
        
        # Load data
        data = self.load_data(view=view, split=split, max_samples=start_idx + num_samples + 50)
        
        # Find valid samples
        valid_indices = np.where(data['is_valid'])[0]
        if len(valid_indices) < start_idx + num_samples:
            print(f"Not enough valid samples. Available: {len(valid_indices)}")
            return
        
        # Visualize samples
        for i in range(num_samples):
            sample_idx = valid_indices[start_idx + i]
            
            joints_2d = data['image_coordinates'][sample_idx]
            joints_3d = data['real_world_coordinates'][sample_idx]
            visible_joints = data['visible_joints'][sample_idx].astype(bool)
            
            print(f"\nVisualizing sample {sample_idx} ({i+1}/{num_samples})...")
            print(f"Visible joints: {np.sum(visible_joints)}/15")
            
            # 2D visualization
            self.visualize_2d_skeleton(
                joints_2d, visible_joints, 
                title=f"2D Skeleton - Sample {sample_idx}"
            )
            
            # 3D visualization
            self.visualize_3d_skeleton(
                joints_3d, visible_joints,
                title=f"3D Skeleton - Sample {sample_idx}"
            )
            
            # With depth image if available
            try:
                self.visualize_with_depth_image(data, sample_idx, view, split)
            except Exception as e:
                print(f"Could not visualize with depth image: {e}")


def main():
    """Main function to demonstrate ITOP visualization"""
    dataset_path = "/home/oliver/Documents/data/ITOP"
    
    # Initialize visualizer
    visualizer = ITOPVisualizer(dataset_path)
    
    # Print joint information
    visualizer.print_joint_info()
    
    # Visualize some samples
    print("\n" + "="*50)
    print("Visualizing ITOP Dataset Samples")
    print("="*50)
    
    # Visualize side view samples
    visualizer.visualize_samples(view='side', split='train', num_samples=3, start_idx=0)
    
    # Uncomment to visualize top view samples
    # visualizer.visualize_samples(view='top', split='train', num_samples=3, start_idx=0)


if __name__ == "__main__":
    main()