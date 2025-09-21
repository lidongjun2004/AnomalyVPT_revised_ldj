import cv2
import os
import numpy as np
from torchvision.transforms import transforms


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


'''
paths: List[str] # Image Path List
anomaly_map: Tensor # shape=[batch_size, img_size, img_size]
img_size: int
save_path: str
cls_name: List[str]
'''


def visualizer(pathes, anomaly_map, img_size, save_path, cls_name, norm_type='img'):
    max_value = anomaly_map.max()
    min_value = anomaly_map.min()

    for idx, path in enumerate(pathes):
        cls = path.split('/')[-2]
        filename = path.split('/')[-1]
        vis = cv2.cvtColor(cv2.resize(cv2.imread(
            path), (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
        if norm_type == 'cls':
            mask = normalize(anomaly_map[idx], max_value, min_value)
        else:
            mask = normalize(anomaly_map[idx])
        vis = apply_ad_scoremap(vis, mask)
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
        save_vis = str(os.path.join(save_path, 'imgs', cls_name[idx], cls))
        if not os.path.exists(save_vis):
            os.makedirs(save_vis)
        cv2.imwrite(str(os.path.join(save_vis, filename)), vis)


'''
image: np.ndarray shape=[img_size, img_size, 3]
scoremap: Tensors shape=[img_size, img_size]
'''


# def apply_ad_scoremap(image, scoremap, alpha=0.5):
#     np_image = np.asarray(image, dtype=float)
#     scoremap = np.asarray(scoremap, dtype=float)
#     scoremap = (scoremap * 255).astype(np.uint8)
#     scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
#     scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
#     return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def apply_ad_scoremap(image, scoremap, alpha=0.5, red_threshold=50, blue_threshold=150):
    np_image = np.asarray(image, dtype=float)
    scoremap = np.asarray(scoremap, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGBA)

    # 获取红色和蓝色通道
    red_channel = scoremap[:, :, 0]
    blue_channel = scoremap[:, :, 2]

    # 根据红色和蓝色通道的值设置 Alpha 通道
    alpha_channel = np.where((red_channel < red_threshold) & (blue_channel > blue_threshold), 0, 255).astype(np.uint8)
    scoremap[:, :, 3] = alpha_channel

    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_RGBA2RGB)
    blended = alpha * np_image + (1 - alpha) * scoremap
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return blended


def visualize_return_img(image, anomaly_map, img_size, norm_type='img'):
    max_value = anomaly_map.max()
    min_value = anomaly_map.min()

    vis = cv2.cvtColor(cv2.resize(image, (img_size, img_size)), cv2.COLOR_BGR2RGB)  # RGB
    if norm_type == 'cls':
        mask = normalize(anomaly_map, max_value, min_value)
    else:
        mask = normalize(anomaly_map)
    vis = apply_ad_scoremap(vis, mask)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR

    return vis