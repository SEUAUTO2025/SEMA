import numpy as np
import pickle
import os
import json
import csv
from pathlib import Path

# ==================== Wholebody3D 配置 ====================

# Wholebody3D 88个关节点定义 (索引 0-87)
# 身体 (0-17) + 脚部 (18-23) + 下嘴唇和下巴 (24-45) + 左手 (46-66) + 右手 (67-87)
WHOLEBODY3D_KEYPOINT_NAMES = {
    # 身体关键点 (0-17)
    0: 'nose', 1: 'neck',
    2: 'right_shoulder', 3: 'right_elbow', 4: 'right_wrist',
    5: 'left_shoulder', 6: 'left_elbow', 7: 'left_wrist',
    8: 'right_hip', 9: 'right_knee', 10: 'right_ankle',
    11: 'left_hip', 12: 'left_knee', 13: 'left_ankle',
    14: 'right_eye', 15: 'left_eye',
    16: 'right_ear', 17: 'left_ear',

    # 脚部 (18-23)
    18: 'left_big_toe', 19: 'left_small_toe', 20: 'left_heel',
    21: 'right_big_toe', 22: 'right_small_toe', 23: 'right_heel',

    # 下嘴唇和下巴 (24-45): 只保留面部的下嘴唇和下巴部分，共22个点
    # 下巴下半部分 (24-30): face 5-11
    24: 'lower_jaw_5', 25: 'lower_jaw_6', 26: 'lower_jaw_7', 27: 'lower_jaw_8',
    28: 'lower_jaw_9', 29: 'lower_jaw_10', 30: 'lower_jaw_11',
    # 外嘴唇下半部分 (31-37): face 48, 54-59
    31: 'outer_lip_48', 32: 'outer_lip_54', 33: 'outer_lip_55', 34: 'outer_lip_56',
    35: 'outer_lip_57', 36: 'outer_lip_58', 37: 'outer_lip_59',
    # 内嘴唇下半部分 (38-45): face 60-67
    38: 'inner_lip_60', 39: 'inner_lip_61', 40: 'inner_lip_62', 41: 'inner_lip_63',
    42: 'inner_lip_64', 43: 'inner_lip_65', 44: 'inner_lip_66', 45: 'inner_lip_67',

    # 左手 (46-66)
    46: 'left_hand_root',
    47: 'left_thumb1', 48: 'left_thumb2', 49: 'left_thumb3', 50: 'left_thumb4',
    51: 'left_forefinger1', 52: 'left_forefinger2', 53: 'left_forefinger3', 54: 'left_forefinger4',
    55: 'left_middle_finger1', 56: 'left_middle_finger2', 57: 'left_middle_finger3', 58: 'left_middle_finger4',
    59: 'left_ring_finger1', 60: 'left_ring_finger2', 61: 'left_ring_finger3', 62: 'left_ring_finger4',
    63: 'left_pinky_finger1', 64: 'left_pinky_finger2', 65: 'left_pinky_finger3', 66: 'left_pinky_finger4',

    # 右手 (67-87)
    67: 'right_hand_root',
    68: 'right_thumb1', 69: 'right_thumb2', 70: 'right_thumb3', 71: 'right_thumb4',
    72: 'right_forefinger1', 73: 'right_forefinger2', 74: 'right_forefinger3', 75: 'right_forefinger4',
    76: 'right_middle_finger1', 77: 'right_middle_finger2', 78: 'right_middle_finger3', 79: 'right_middle_finger4',
    80: 'right_ring_finger1', 81: 'right_ring_finger2', 82: 'right_ring_finger3', 83: 'right_ring_finger4',
    84: 'right_pinky_finger1', 85: 'right_pinky_finger2', 86: 'right_pinky_finger3', 87: 'right_pinky_finger4',
}

# ==================== 分部位关键点字典（用于训练5个独立模型） ====================

# 1. 头部模型：下嘴唇和下巴区域 (24-45)
HEAD_KEYPOINT_NAMES = {
    # 下巴下半部分 (0-6): 从24重新映射到0
    0: 'lower_jaw_5', 1: 'lower_jaw_6', 2: 'lower_jaw_7', 3: 'lower_jaw_8',
    4: 'lower_jaw_9', 5: 'lower_jaw_10', 6: 'lower_jaw_11',
    # 外嘴唇下半部分 (7-13): 从31重新映射到7
    7: 'outer_lip_48', 8: 'outer_lip_54', 9: 'outer_lip_55', 10: 'outer_lip_56',
    11: 'outer_lip_57', 12: 'outer_lip_58', 13: 'outer_lip_59',
    # 内嘴唇下半部分 (14-21): 从38重新映射到14
    14: 'inner_lip_60', 15: 'inner_lip_61', 16: 'inner_lip_62', 17: 'inner_lip_63',
    18: 'inner_lip_64', 19: 'inner_lip_65', 20: 'inner_lip_66', 21: 'inner_lip_67',
}

HEAD_BONE_CONNECTIONS = [
    # 下巴轮廓 (连续的点)
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    # 外嘴唇下半部分 (形成闭合轮廓)
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 7),
    # 内嘴唇下半部分 (形成闭合轮廓)
    (14, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 14),
    # 连接外嘴唇和内嘴唇
    (7, 14), (10, 17), (13, 21),
]

# 原始索引映射（用于从133点提取对应子集）
HEAD_ORIGINAL_INDICES = list(range(24, 46))  # [24, 25, ..., 45]


# 2. 手部模型：左手 + 右手 (46-87)
HAND_KEYPOINT_NAMES = {
    # 左手 (0-20): 从46重新映射到0
    0: 'left_hand_root',
    1: 'left_thumb1', 2: 'left_thumb2', 3: 'left_thumb3', 4: 'left_thumb4',
    5: 'left_forefinger1', 6: 'left_forefinger2', 7: 'left_forefinger3', 8: 'left_forefinger4',
    9: 'left_middle_finger1', 10: 'left_middle_finger2', 11: 'left_middle_finger3', 12: 'left_middle_finger4',
    13: 'left_ring_finger1', 14: 'left_ring_finger2', 15: 'left_ring_finger3', 16: 'left_ring_finger4',
    17: 'left_pinky_finger1', 18: 'left_pinky_finger2', 19: 'left_pinky_finger3', 20: 'left_pinky_finger4',

    # 右手 (21-41): 从67重新映射到21
    21: 'right_hand_root',
    22: 'right_thumb1', 23: 'right_thumb2', 24: 'right_thumb3', 25: 'right_thumb4',
    26: 'right_forefinger1', 27: 'right_forefinger2', 28: 'right_forefinger3', 29: 'right_forefinger4',
    30: 'right_middle_finger1', 31: 'right_middle_finger2', 32: 'right_middle_finger3', 33: 'right_middle_finger4',
    34: 'right_ring_finger1', 35: 'right_ring_finger2', 36: 'right_ring_finger3', 37: 'right_ring_finger4',
    38: 'right_pinky_finger1', 39: 'right_pinky_finger2', 40: 'right_pinky_finger3', 41: 'right_pinky_finger4',
}

HAND_BONE_CONNECTIONS = [
    # 左手骨架
    (0, 1), (1, 2), (2, 3), (3, 4),      # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # forefinger
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),# ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),# pinky finger

    # 右手骨架
    (21, 22), (22, 23), (23, 24), (24, 25),    # thumb
    (21, 26), (26, 27), (27, 28), (28, 29),    # forefinger
    (21, 30), (30, 31), (31, 32), (32, 33),    # middle finger
    (21, 34), (34, 35), (35, 36), (36, 37),    # ring finger
    (21, 38), (38, 39), (39, 40), (40, 41),    # pinky finger
]

# 原始索引映射
HAND_ORIGINAL_INDICES = list(range(46, 88))  # [46, 47, ..., 87]


# 3. 脚部模型 (18-23)
FOOT_KEYPOINT_NAMES = {
    # 左脚 (0-2): 从18重新映射到0
    0: 'left_big_toe', 1: 'left_small_toe', 2: 'left_heel',
    # 右脚 (3-5): 从21重新映射到3
    3: 'right_big_toe', 4: 'right_small_toe', 5: 'right_heel',
}

FOOT_BONE_CONNECTIONS = [
    # 左脚连接 (形成三角形)
    (0, 1), (1, 2), (2, 0),
    # 右脚连接 (形成三角形)
    (3, 4), (4, 5), (5, 3),
]

# 原始索引映射
FOOT_ORIGINAL_INDICES = list(range(18, 24))  # [18, 19, 20, 21, 22, 23]


# 4. 躯干模型：颈部、肩膀、髋部 (1, 2, 5, 8, 11)
TORSO_KEYPOINT_NAMES = {
    0: 'neck',           # 从1映射
    1: 'right_shoulder', # 从2映射
    2: 'left_shoulder',  # 从5映射
    3: 'right_hip',      # 从8映射
    4: 'left_hip',       # 从11映射
}

TORSO_BONE_CONNECTIONS = [
    # 躯干骨架（形成一个五边形结构）
    (0, 1),  # neck -> right_shoulder
    (0, 2),  # neck -> left_shoulder
    (1, 3),  # right_shoulder -> right_hip
    (2, 4),  # left_shoulder -> left_hip
    (3, 4),  # right_hip -> left_hip (连接两个髋部)
]

# 原始索引映射
TORSO_ORIGINAL_INDICES = [1, 2, 5, 8, 11]


# 5. 手臂模型：双臂 (2-7，包括肩膀、肘部、手腕)
ARM_KEYPOINT_NAMES = {
    # 右臂 (0-2): 从2重新映射到0
    0: 'right_shoulder', 1: 'right_elbow', 2: 'right_wrist',
    # 左臂 (3-5): 从5重新映射到3
    3: 'left_shoulder', 4: 'left_elbow', 5: 'left_wrist',
}

ARM_BONE_CONNECTIONS = [
    # 右臂
    (0, 1), (1, 2),  # right_shoulder -> right_elbow -> right_wrist
    # 左臂
    (3, 4), (4, 5),  # left_shoulder -> left_elbow -> left_wrist
]

# 原始索引映射
ARM_ORIGINAL_INDICES = [2, 3, 4, 5, 6, 7]


# ==================== 全身模型（原始） ====================

# Wholebody3D 骨骼连接定义（用于构建图结构）
WHOLEBODY3D_BONE_CONNECTIONS = [
    # 身体骨架 - 躯干
    (1, 2), (1, 5),  # neck -> shoulders
    (2, 3), (3, 4),  # right arm: shoulder -> elbow -> wrist
    (5, 6), (6, 7),  # left arm: shoulder -> elbow -> wrist
    (1, 8), (8, 9), (9, 10),  # right leg: hip -> knee -> ankle
    (1, 11), (11, 12), (12, 13),  # left leg: hip -> knee -> ankle

    # 脚部连接
    (10, 21), (10, 22), (10, 23),  # right_ankle -> right foot (big_toe, small_toe, heel)
    (13, 18), (13, 19), (13, 20),  # left_ankle -> left foot (big_toe, small_toe, heel)

    # 下嘴唇和下巴连接
    # 下巴轮廓 (连续的点)
    (24, 25), (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
    # 外嘴唇下半部分 (形成闭合轮廓)
    (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37), (37, 31),
    # 内嘴唇下半部分 (形成闭合轮廓)
    (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44), (44, 45), (45, 38),
    # 连接外嘴唇和内嘴唇 (可选，增强结构连接)
    (31, 38), (34, 41), (37, 45),

    # 左手骨架
    (46, 47), (47, 48), (48, 49), (49, 50),  # thumb
    (46, 51), (51, 52), (52, 53), (53, 54),  # forefinger
    (46, 55), (55, 56), (56, 57), (57, 58),  # middle finger
    (46, 59), (59, 60), (60, 61), (61, 62),  # ring finger
    (46, 63), (63, 64), (64, 65), (65, 66),  # pinky finger

    # 右手骨架
    (67, 68), (68, 69), (69, 70), (70, 71),  # thumb
    (67, 72), (72, 73), (73, 74), (74, 75),  # forefinger
    (67, 76), (76, 77), (77, 78), (78, 79),  # middle finger
    (67, 80), (80, 81), (81, 82), (82, 83),  # ring finger
    (67, 84), (84, 85), (85, 86), (86, 87),  # pinky finger

    # 连接手部到手腕
    (7, 46),   # left_wrist -> left_hand_root
    (4, 67),   # right_wrist -> right_hand_root
]

# 配置
NUM_JOINTS = 133        # Wholebody3D 关节数 (只保留下嘴唇和下巴)
NUM_PERSONS = 1        # 单人姿态评估
MAX_FRAMES = 124       # 最大帧数
NUM_COORDS = 3         # x, y, z 坐标

def load_single_csv_with_multipart_labels(csv_path, max_frames=MAX_FRAMES):
    """
    从单个CSV文件加载数据，并根据CSV中的多个标签列分别提取

    CSV格式（新格式，包含5个标签列）：
    frame,x0,y0,z0,x1,y1,z1,...,x132,y132,z132,label_hand,label_head,label_foot,label_arm,label_torso,label_total

    Args:
        csv_path: CSV文件路径
        max_frames: 最大帧数

    Returns:
        data: (3, T, 133, M) 全身数据
        labels: dict，包含5个标签 {'hand': float, 'head': float, 'feet': float, 'arm': float, 'body': float,'total':float}
    """
    num_keypoints = 133
    data = np.zeros((NUM_COORDS, max_frames, num_keypoints, NUM_PERSONS), dtype=np.float32)
    labels = {'hand': 0.0, 'head': 0.0, 'feet': 0.0, 'arm': 0.0, 'body': 0.0}
    labels_read = False

    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 解析表头，查找标签列索引
    header = lines[0].strip().split(',')
    label_indices = {}
    label_total = 0.0
    for label_name in ['label_hand', 'label_head', 'label_feet', 'label_arm', 'label_body','label_total']: #标注数据放在表头下面的第一行就行了
        if label_name in header:
            label_indices[label_name] = header.index(label_name)

    for line in lines[1:]:  # 跳过表头
        line = line.strip()
        if not line:
            continue

        values = line.split(',')
        frame_id = int(values[0])

        if frame_id >= max_frames:
            continue

        # 提取关键点坐标
        for i in range(num_keypoints):
            x = float(values[1 + i * 3])
            y = float(values[1 + i * 3 + 1])
            z = float(values[1 + i * 3 + 2])

            data[0, frame_id, i, 0] = x
            data[1, frame_id, i, 0] = y
            data[2, frame_id, i, 0] = z

        # 提取标签（只读一次）
        if not labels_read and label_indices:
            if 'label_hand' in label_indices and label_indices['label_hand'] < len(values):
                labels['hand'] = float(values[label_indices['label_hand']])
            if 'label_head' in label_indices and label_indices['label_head'] < len(values):
                labels['head'] = float(values[label_indices['label_head']])
            if 'label_feet' in label_indices and label_indices['label_feet'] < len(values):
                labels['feet'] = float(values[label_indices['label_feet']])
            if 'label_arm' in label_indices and label_indices['label_arm'] < len(values):
                labels['arm'] = float(values[label_indices['label_arm']])
            if 'label_body' in label_indices and label_indices['label_body'] < len(values):
                labels['body'] = float(values[label_indices['label_body']])
            if 'label_total' in label_indices and label_indices['label_total'] < len(values):
                label_total = float(values[label_indices['label_total']])
            labels_read = True
    return data, labels, label_total