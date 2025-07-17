# 3D Human Pose Estimation with Monocular RGB Camera
For comparison, MediaPipe output(33 joints) has already been converted to Human3.6M format(17 joints) according to the conversion rule in **mediapipe_h36m.txt**
## Modify or add inference model from MMPose
* 3 models available at **configs/body_3d_keypoint**
* To create an instance of a certain model, initiate the instance by 
```
mmpose_model = MMPoseInferencer(
    pose3d=model_config_path, 
    pose3d_weights=model_link,
    device=device
)
```
* Refer to ```test_h36m_inference.py``` for detailed usage
## Run human pose inference on Human 3.6M dataset
```
python test_h36m_inference.py
```
## Run human pose inference on Realsense camera input
```
python test_realsense_inference.py
```