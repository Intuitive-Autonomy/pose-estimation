import cv2
import numpy as np
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import queue

# import your converter function
from mediapipe2bvh import mediapipe33_to_bvh31

# MediaPipe imports
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables for thread-safe data sharing
result_queue = queue.Queue(maxsize=1)
latest_result = None
result_lock = threading.Lock()

# MediaPipe pose connections for skeleton drawing
POSE_CONNECTIONS = [
    (0, 2), (2, 7), (0, 5), (5, 8),  # Face
    (9, 10),  # Mouth
    (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),  # Left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),  # Right arm
    (11, 12), (11, 23), (12, 24), (23, 24),  # Torso
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)   # Right leg
]

class RealSensePoseDetector:
    def __init__(self, model_path='pose_landmarker.task'):
        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(self.config)

        # MediaPipe callback
        self.result_callback = self.process_result
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.result_callback
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.timestamp_ms = 0

        # Setup 3D visualization
        self.setup_3d_visualization()

    def setup_3d_visualization(self):
        plt.ion()
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_camera = self.fig.add_subplot(121)
        self.ax_camera.set_title('Camera Feed')
        self.ax_camera.axis('off')
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.ax_3d.set_title('3D Skeletons')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_xlim(-0.5, 0.5)
        self.ax_3d.set_ylim(-0.5, 0.5)
        self.ax_3d.set_zlim(-0.5, 0.5)
        self.ax_3d.view_init(elev=20, azim=45)
        plt.tight_layout()

    def process_result(self, result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        global latest_result
        with result_lock:
            latest_result = result

    def draw_3d_skeleton(self, world_landmarks):
        # preserve view
        curr_elev, curr_azim = self.ax_3d.elev, self.ax_3d.azim
        # clear
        self.ax_3d.cla()
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')
        self.ax_3d.set_xlim(-0.5, 0.5)
        self.ax_3d.set_ylim(-0.5, 0.5)
        self.ax_3d.set_zlim(-0.5, 0.5)
        self.ax_3d.set_title('3D Skeletons')

        if not world_landmarks:
            self.ax_3d.view_init(elev=curr_elev, azim=curr_azim)
            return

        # --- draw raw MediaPipe skeleton ---
        # MediaPipe: x=right, y=down, z=forward -> x=right, y=forward, z=up
        mp_pts = np.array([[lm.x, -lm.z, -lm.y] for lm in world_landmarks[0]])
        
        # Correct conversion: (x, y, z) -> (x, -y, z)
        mp_pts[:, 1] = -mp_pts[:, 1]  # Flip Y only
        
        xs, ys, zs = mp_pts.T
        self.ax_3d.scatter(xs, ys, zs, s=40, alpha=0.6)  # media pipe points
        for a, b in POSE_CONNECTIONS:
            if a < len(xs) and b < len(xs):
                self.ax_3d.plot(
                    [xs[a], xs[b]],
                    [ys[a], ys[b]],
                    [zs[a], zs[b]],
                    linewidth=2, alpha=0.5
                )

        # --- convert & draw BVH skeleton ---
        bvh_pts = mediapipe33_to_bvh31(mp_pts)  # returns (31,3)
        bx, by, bz = bvh_pts.T
        # scatter BVH joints with a different marker
        self.ax_3d.scatter(bx, by, bz, marker='^', s=30, alpha=0.8, color='red')

        # restore view
        self.ax_3d.view_init(elev=curr_elev, azim=curr_azim)

    def draw_2d_landmarks(self, image, landmarks):
        if not landmarks:
            return image
        h, w = image.shape[:2]
        for a, b in POSE_CONNECTIONS:
            if a < len(landmarks[0]) and b < len(landmarks[0]):
                p1, p2 = landmarks[0][a], landmarks[0][b]
                if p1.visibility > 0.5 and p2.visibility > 0.5:
                    pt1 = (int(p1.x * w), int(p1.y * h))
                    pt2 = (int(p2.x * w), int(p2.y * h))
                    cv2.line(image, pt1, pt2, (0,255,0), 2)
        for lm in landmarks[0]:
            if lm.visibility > 0.5:
                cv2.circle(image,
                           (int(lm.x*w), int(lm.y*h)),
                           5, (0,0,255), -1)
        return image

    def run(self):
        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # async detect
                self.timestamp_ms += 33
                self.landmarker.detect_async(mp_image, self.timestamp_ms)

                with result_lock:
                    res = latest_result

                disp = color_image.copy()
                if res and res.pose_landmarks:
                    disp = self.draw_2d_landmarks(disp, res.pose_landmarks)
                if res and res.pose_world_landmarks:
                    self.draw_3d_skeleton(res.pose_world_landmarks)

                # show camera + 2D overlay
                self.ax_camera.clear()
                self.ax_camera.imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
                self.ax_camera.set_title('Camera Feed with 2D Pose')
                self.ax_camera.axis('off')

                plt.pause(0.001)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()
        plt.close('all')
        if hasattr(self, 'landmarker'):
            self.landmarker.close()

def main():
    print("Starting RealSense + MediaPipe + BVH converter...")
    print("Press 'q' to quit")
    detector = RealSensePoseDetector(model_path='pose_landmarker.task')
    detector.run()

if __name__ == "__main__":
    main()
