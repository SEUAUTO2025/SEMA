import sys, os
from DIR import project_root

if project_root not in sys.path:
    sys.path.insert(0, project_root)
        
import cv2
# from rtmlib.rtmlib import PoseTracker, Wholebody3d, draw_skeleton

import torch
import numpy as np
import ruptures as rpt

from Tools.Exe_dataset.model_config import model_configs

def Keypoint_Extract(path_to_video, image_width=1920, image_height=1080):
    from RTMPose.rtmlib import PoseTracker, Wholebody3d, draw_skeleton
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


def extract_keyframes_with_ruptures(
    math_features,
    k=8,
    model="rbf",
    min_size=5,
    jump=1,
    candidate_multiplier=3,
    score_window=5
):
    """
    Extract keyframes from three biomechanical features using ruptures.

    Args:
        math_features: dict containing keys
            - 'shoulder_wrist_angle'
            - 'left_hand_to_chin_dist'
            - 'shoulder_foot_x_diff'
            Values can be torch.Tensor or numpy.ndarray with shape (frames,)
        k: Number of keyframes to return
        model: ruptures cost model, default 'rbf'
        min_size: Minimum segment size for change point detection
        jump: Search step size for change point detection
        candidate_multiplier: Candidate breakpoints multiplier for Top-k selection
        score_window: Window size around breakpoint for intensity scoring

    Returns:
        list[dict]: Sorted keyframes by descending score
            [{"frame_idx": int, "score": float}, ...]
    """
    required_keys = [
        "shoulder_wrist_angle",
        "left_hand_to_chin_dist",
        "shoulder_foot_x_diff",
    ]

    if not isinstance(math_features, dict):
        raise TypeError("math_features must be a dict")
    if rpt is None:
        raise ImportError(
            "ruptures is required for extract_keyframes_with_ruptures. "
            "Please install it (e.g., pip install ruptures)."
        )

    missing_keys = [key for key in required_keys if key not in math_features]
    if missing_keys:
        raise KeyError(f"math_features missing required keys: {missing_keys}")

    if k <= 0:
        return []

    min_size = max(1, int(min_size))
    jump = max(1, int(jump))
    score_window = max(1, int(score_window))
    candidate_multiplier = max(1, int(candidate_multiplier))
    k = int(k)

    def _to_1d_float64(value):
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)
        return arr

    def _fill_invalid_1d(arr):
        arr = np.asarray(arr, dtype=np.float64).copy()
        valid = np.isfinite(arr)
        if valid.sum() == 0:
            return np.zeros_like(arr, dtype=np.float64)
        if valid.sum() == len(arr):
            return arr
        valid_idx = np.where(valid)[0]
        invalid_idx = np.where(~valid)[0]
        arr[invalid_idx] = np.interp(invalid_idx, valid_idx, arr[valid_idx])
        return arr

    def _zscore_columns(x):
        x = np.asarray(x, dtype=np.float64)
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        std_safe = np.where(std < 1e-8, 1.0, std)
        x_std = (x - mean) / std_safe
        x_std[:, (std.reshape(-1) < 1e-8)] = 0.0
        return x_std

    def _fallback_from_diffs(x_std, top_k):
        t = x_std.shape[0]
        if t <= 1:
            return []
        diffs = np.mean(np.abs(x_std[1:] - x_std[:-1]), axis=1)
        if diffs.size == 0:
            return []
        candidate_frames = np.arange(1, t)
        order = sorted(
            range(len(candidate_frames)),
            key=lambda i: (-float(diffs[i]), int(candidate_frames[i])),
        )
        return [
            {"frame_idx": int(candidate_frames[i]), "score": float(diffs[i])}
            for i in order[:top_k]
        ]

    feature_arrays = [_to_1d_float64(math_features[key]) for key in required_keys]
    lengths = [len(arr) for arr in feature_arrays]
    if min(lengths) == 0:
        return []

    if len(set(lengths)) != 1:
        min_len = min(lengths)
        feature_arrays = [arr[:min_len] for arr in feature_arrays]

    filled_arrays = [_fill_invalid_1d(arr) for arr in feature_arrays]
    x = np.column_stack(filled_arrays)
    t = x.shape[0]
    x_std = _zscore_columns(x)

    if t < 4:
        return _fallback_from_diffs(x_std, min(k, max(0, t - 1)))

    n_bkps = min(t - 1, max(k, k * candidate_multiplier))

    try:
        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(x_std)
        bkps = algo.predict(n_bkps=n_bkps)
    except Exception:
        return _fallback_from_diffs(x_std, min(k, max(0, t - 1)))

    candidates = sorted({int(b) for b in bkps if 0 < int(b) < t})
    scored = []
    for b in candidates:
        left_start = max(0, b - score_window)
        right_end = min(t, b + score_window)
        left = x_std[left_start:b]
        right = x_std[b:right_end]
        if left.shape[0] == 0 or right.shape[0] == 0:
            continue
        mu_l = left.mean(axis=0)
        mu_r = right.mean(axis=0)
        score = float(np.mean(np.abs(mu_r - mu_l)))
        scored.append({"frame_idx": int(b), "score": score})

    if not scored:
        return _fallback_from_diffs(x_std, min(k, max(0, t - 1)))

    scored.sort(key=lambda item: (-item["score"], item["frame_idx"]))
    return scored[:k]

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
