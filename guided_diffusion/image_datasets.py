import random
import blobfile as bf
import math
import numpy as np
import cv2
import re
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist


def load_data(
    *,
    data_dir_sar,
    data_dir_opt,
    batch_size,
    image_size,
    class_cond=False,
    deterministic=False,
    random_crop=False,
    random_flip=False,
):
    """
    为数据集创建 (images, kwargs) 对的生成器。

    每个 images 是一个 NCHW 浮点张量，kwargs 字典包含零个或多个键，每个键映射到一个批处理的张量。
    kwargs 字典可用于类标签，键为 "y"，值为类标签的整数张量。

    :param data_dir_sar: SAR 数据集目录。
    :param data_dir_opt: OPT 数据集目录。
    :param batch_size: 每个返回对的批大小。
    :param image_size: 图像调整的目标大小。
    :param class_cond: 如果为 True，返回字典中包含 "y" 键作为类标签。
    :param deterministic: 如果为 True，按确定性顺序返回结果。
    :param random_crop: 如果为 True，随机裁剪图像以进行数据增强。
    :param random_flip: 如果为 True，随机翻转图像以进行数据增强。
    """
    print("data_dir_sar: ", data_dir_sar)
    print("data_dir_opt: ", data_dir_opt)
    if not data_dir_sar or not data_dir_opt:
        raise ValueError("未指定数据目录")

    all_files_sar = _list_image_files_recursively(data_dir_sar)
    all_files_opt = _list_image_files_recursively(data_dir_opt)
    classes = None

    # 按文件名中的数字部分排序（如 001_abudhabi_r00_c00.tif 中的 001）
    def extract_number(filename):
        # 提取文件名（去掉路径和扩展名）中的数字前缀
        basename = bf.basename(filename).split('.')[0]
        match = re.match(r'^(\d+)', basename)
        return int(match.group(1)) if match else 0  # 默认返回 0 如果没有数字

    all_files_sar.sort(key=extract_number)
    all_files_opt.sort(key=extract_number)

    if class_cond:
        # 假设类标签是文件名中下划线前的部分
        class_names = [bf.basename(path).split("_")[0] for path in data_dir_opt]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]

    # 根据是否初始化分布式环境设置 shard 和 num_shards
    shard = dist.get_rank() if dist.is_initialized() else 0
    num_shards = dist.get_world_size() if dist.is_initialized() else 1

    dataset = ImageDataset(
        image_size,
        all_files_sar,
        all_files_opt,
        classes=classes,
        shard=shard,
        num_shards=num_shards,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    递归列出数据目录中的所有图像文件（支持 .tif 格式）。
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1].lower()
        if "." in entry and ext in ["tif", "tiff", "jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths_sar,
        image_paths_opt,
        classes=None,
        shard=0,
        num_shards=1,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images_sar = image_paths_sar[shard:][::num_shards]
        self.local_images_opt = image_paths_opt[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images_sar)

    def __getitem__(self, idx):
        path_sar = self.local_images_sar[idx]
        path_opt = self.local_images_opt[idx]

        # 使用 OpenCV 读取 .tif 文件
        img_sar = cv2.imread(path_sar, cv2.IMREAD_COLOR)
        img_opt = cv2.imread(path_opt, cv2.IMREAD_COLOR)

        if img_sar is None or img_opt is None:
            raise ValueError(f"无法读取图像: {path_sar} 或 {path_opt}")

        # OpenCV 读取的图像是 BGR 格式，转换为 RGB
        img_sar = cv2.cvtColor(img_sar, cv2.COLOR_BGR2RGB)
        img_opt = cv2.cvtColor(img_opt, cv2.COLOR_BGR2RGB)

        # 裁剪图像
        if self.random_crop:
            arr_sar = random_crop_arr(img_sar, self.resolution)
            arr_opt = random_crop_arr(img_opt, self.resolution)
        else:
            arr_sar = center_crop_arr(img_sar, self.resolution)
            arr_opt = center_crop_arr(img_opt, self.resolution)

        # 随机翻转
        if self.random_flip and random.random() < 0.5:
            arr_sar = arr_sar[:, ::-1]
            arr_opt = arr_opt[:, ::-1]

        # 归一化到 [-1, 1]
        arr_sar = arr_sar.astype(np.float32) / 127.5 - 1
        arr_opt = arr_opt.astype(np.float32) / 127.5 - 1

        # 转换为 NCHW 格式
        arr_sar = np.transpose(arr_sar, [2, 0, 1])
        arr_opt = np.transpose(arr_opt, [2, 0, 1])
        arr = np.concatenate((arr_sar, arr_opt), axis=0)

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return arr, out_dict


def center_crop_arr(img, image_size):
    """
    中心裁剪图像到指定大小。
    """
    h, w = img.shape[:2]
    if h < image_size or w < image_size:
        # 如果图像太小，调整大小
        scale = max(image_size / h, image_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        # 直接裁剪
        crop_y = (h - image_size) // 2
        crop_x = (w - image_size) // 2
        img = img[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    return img


def random_crop_arr(img, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    """
    随机裁剪图像到指定大小。
    """
    h, w = img.shape[:2]
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    if min(h, w) < smaller_dim_size:
        # 如果图像太小，调整大小
        scale = smaller_dim_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        # 直接裁剪
        scale = smaller_dim_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    crop_y = random.randrange(img.shape[0] - image_size + 1)
    crop_x = random.randrange(img.shape[1] - image_size + 1)
    return img[crop_y: crop_y + image_size, crop_x: crop_x + image_size]