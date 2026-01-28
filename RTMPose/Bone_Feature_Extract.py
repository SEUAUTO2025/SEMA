import sys, os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)
        
import cv2
# from rtmlib.rtmlib import PoseTracker, Wholebody3d, draw_skeleton
from RTMPose.rtmlib import PoseTracker, Wholebody3d, draw_skeleton
import torch
import numpy as np
from Tools.Gen_dataset.model_config import model_configs

def Keypoint_Extract(path_to_video, image_width=1920, image_height=1080):
    """
    Extract keypoint information from all frames in video, normalize and return tensor
    
    Args:
        path_to_video: Path to video file
        image_width: Video width for normalization
        image_height: Video height for normalization
    
    Returns:
        normalized_data: (frames, 133, 3) normalized coordinates
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backend = 'onnxruntime'
    cap = cv2.VideoCapture(path_to_video)

    wholebody3d = PoseTracker(
        Wholebody3d,
        det_frequency=7,
        tracking=False,
        backend=backend,
        device='cuda')

    frame_idx = -1
    whole_skeleton_data = []
    while cap.isOpened():
        success, frame = cap.read()
        frame_idx += 1
        if frame_idx >= 230:
            break
        if not success:
            break
        keypoints, scores, keypoints_simcc, keypoints_2d = wholebody3d(frame)
        keypoints = keypoints[0]
        whole_skeleton_data.append(keypoints)
        
    whole_skeleton_data = np.stack(whole_skeleton_data, axis=0)
    normalized_data = torch.from_numpy(whole_skeleton_data).float()
    
    normalized_data[:, :, 0] /= image_width
    normalized_data[:, :, 1] /= image_height
    
    return normalized_data

def generate_coordinate_feature(joint_data, bone_connections):
    """
    Generate coordinate features from joint data and bone connections
    
    Args:
        joint_data: Joint position data
        bone_connections: List of bone connection pairs
    
    Returns:
        coordinate_data: Combined joint and bone features
    """
    N, C, T, V, M = joint_data.shape
    print(f'    Joint data shape: {joint_data.shape}')
    
    bone_data = torch.zeros_like(joint_data)
    for v1, v2 in bone_connections:
        if v1 < V and v2 < V:
            bone_data[:, :, :, v1, :] = joint_data[:, :, :, v1, :] - joint_data[:, :, :, v2, :]

    coordinate_data = torch.cat([joint_data, bone_data], dim=1)
    return coordinate_data

def Extract_Bodypart_Data(full_data, image_width=2560, image_height=1440):
    """
    Extract keypoints for specific body parts from full body data

    Args:
        full_data: (frames, 133, 3) full body data
        image_width: Image width for normalization
        image_height: Image height for normalization

    Returns:
        data_dict: Dictionary containing body part data
        math_feature: Biomechanical features
    """
    
    data_dict = {}
    full_data[:, :, 0] /= image_width
    full_data[:, :, 1] /= image_height
    math_feature = cal_math_features(full_data)
    
    for model_name, config in model_configs.items():
        transformed = full_data[:, model_configs[model_name]['indices'], :].permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)
        data = generate_coordinate_feature(transformed, config['bone_connections'])
        data_dict[f'{model_name.replace("_model", "")}'] = data
        
    return data_dict, math_feature

def cal_math_features(keypoints_data):
    """
    Calculate biomechanical features

    Args:
        keypoints_data: (frames, 133, 3) full body keypoint data, 3 represents (x, y, z) coordinates
                       Supports torch.Tensor or numpy.ndarray

    Returns:
        dict: Dictionary containing various biomechanical features
            - 'shoulder_wrist_angle': (frames,) angle between left wrist-left shoulder-right wrist (degrees)
            - 'left_hand_to_chin_dist': (frames,) distance from left hand to chin
            - 'shoulder_foot_x_diff': (frames,) x-coordinate difference between shoulder midpoint and foot midpoint
    """
    if isinstance(keypoints_data, np.ndarray):
        keypoints_data = torch.from_numpy(keypoints_data)

    LEFT_WRIST_IDX = 7
    LEFT_SHOULDER_IDX = 5
    RIGHT_WRIST_IDX = 4
    RIGHT_SHOULDER_IDX = 2
    LOWER_JAW_IDX = 27
    LEFT_ANKLE_IDX = 13
    RIGHT_ANKLE_IDX = 10
    
    left_wrist = keypoints_data[:, LEFT_WRIST_IDX, :]
    left_shoulder = keypoints_data[:, LEFT_SHOULDER_IDX, :]
    right_wrist = keypoints_data[:, RIGHT_WRIST_IDX, :]
    right_shoulder = keypoints_data[:, RIGHT_SHOULDER_IDX, :]

    vector_to_left_wrist = left_wrist - left_shoulder
    vector_to_right_wrist = right_wrist - left_shoulder

    norm_left = torch.norm(vector_to_left_wrist, dim=1, keepdim=True)
    norm_right = torch.norm(vector_to_right_wrist, dim=1, keepdim=True)

    norm_left = torch.where(norm_left == 0, torch.tensor(1e-8), norm_left)
    norm_right = torch.where(norm_right == 0, torch.tensor(1e-8), norm_right)

    vector_to_left_wrist_norm = vector_to_left_wrist / norm_left
    vector_to_right_wrist_norm = vector_to_right_wrist / norm_right

    dot_product = torch.sum(vector_to_left_wrist_norm * vector_to_right_wrist_norm, dim=1)

    dot_product = torch.clamp(dot_product, -1.0, 1.0)

    angle_rad = torch.acos(dot_product)

    angle_deg = angle_rad * 180.0 / torch.pi

    chin = keypoints_data[:, LOWER_JAW_IDX, :]
    vector_left_hand_to_chin = chin - left_wrist
    dist_left_hand_to_chin = torch.norm(vector_left_hand_to_chin, dim=1)

    left_ankle = keypoints_data[:, LEFT_ANKLE_IDX, :]
    right_ankle = keypoints_data[:, RIGHT_ANKLE_IDX, :]
    
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2.0
    print(shoulder_midpoint)
    
    foot_midpoint = (left_ankle + right_ankle) / 2.0
    
    shoulder_foot_x_diff = shoulder_midpoint[:, 0] - foot_midpoint[:, 0]

    features = {
        'shoulder_wrist_angle': angle_deg,
        'left_hand_to_chin_dist': dist_left_hand_to_chin,
        'shoulder_foot_x_diff': shoulder_foot_x_diff
    }

    return features

def extract_action_features(features):
    """
    Extract specific statistical features from action tensors
    
    Args:
        features: Dictionary containing torch.tensors (shape: [frames])
    
    Returns:
        Dictionary with processed scalar values
    """
    results = {}
    window_size = 10

    angle_tensor = features['shoulder_wrist_angle']
    if len(angle_tensor) >= window_size:
        windows = angle_tensor.unfold(0, window_size, 1)
        window_means = windows.mean(dim=1)
        results['max_angle_avg'] = torch.max(window_means).item()
    else:
        results['max_angle_avg'] = angle_tensor.mean().item()

    dist_tensor = features['left_hand_to_chin_dist']
    if len(dist_tensor) >= window_size:
        windows = dist_tensor.unfold(0, window_size, 1)
        window_means = windows.mean(dim=1)
        results['min_dist_avg'] = torch.min(window_means).item()
    else:
        results['min_dist_avg'] = dist_tensor.mean().item()

    x_diff_tensor = features['shoulder_foot_x_diff']
    if len(x_diff_tensor) >= window_size:
        sorted_values, _ = torch.sort(torch.abs(x_diff_tensor))
        top_10_min = sorted_values[:window_size]
        results['min_x_diff_avg'] = top_10_min.mean().item()
    else:
        results['min_x_diff_avg'] = torch.abs(x_diff_tensor).mean().item()

    return results