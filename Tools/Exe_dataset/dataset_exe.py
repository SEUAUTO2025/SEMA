import numpy as np
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