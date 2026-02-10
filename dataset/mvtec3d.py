"""
  @Author: 王权
  @FileName: mvtec3d.py
  @DateTime: 2026/1/28 18:26
  @SoftWare: PyCharm
"""
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MVTec3DDataset(Dataset):
    """
    为 AF-CLIP 设计的 MVTec 3D-AD 数据集加载类，
    接口仿照 2D 的 MVTecDataset：
        - 有 preprocess()
        - 有 update(category)
        - __getitem__ 返回: img, label, gt, category, img_path
    """

    def __init__(self, root, train=True, category=None, fewshot=0,
                 transform=None, gt_target_transform=None, use_validation=True):
        super(MVTec3DDataset, self).__init__()

        # 10 个 3D 类别
        self.categories = [
            "bagel",
            "cable_gland",
            "carrot",
            "cookie",
            "dowel",
            "foam",
            "peach",
            "potato",
            "rope",
            "tire",
        ]

        self.train = train
        self.category = category
        self.fewshot = fewshot
        # 注意这里用的是 Mvtec3D 这个目录名，和你本地保持一致
        self.root = os.path.join(root, "mvtec_3d_anomaly_detection")
        self.transform = transform
        self.gt_target_transform = gt_target_transform
        self.use_validation = use_validation

        self.preprocess()
        self.update(self.category)

        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)

        self.dataset_name = "mvtec_3d"

    # -------------------------------------------------------------
    # 预处理：收集所有类别 / train/test 所有图像路径和标签
    # -------------------------------------------------------------
    def preprocess(self):
        self.img_paths = {
            "train": {c: [] for c in self.categories},
            "test": {c: [] for c in self.categories},
        }
        self.gt_paths = {
            "train": {c: [] for c in self.categories},
            "test": {c: [] for c in self.categories},
        }
        self.labels = {
            "train": {c: [] for c in self.categories},
            "test": {c: [] for c in self.categories},
        }

        # ------------------ 训练部分：train (+ 可选 validation) ------------------
        for category in self.categories:
            obj_dir = os.path.join(self.root, category)

            train_splits = ["train"]
            if self.use_validation:
                train_splits.append("validation")

            for split in train_splits:
                split_dir = os.path.join(obj_dir, split)
                if not os.path.isdir(split_dir):
                    continue

                defect_types = os.listdir(split_dir)
                for defect_type in defect_types:
                    defect_dir = os.path.join(split_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    # 3D 数据集里，RGB 一般在 rgb 子目录
                    rgb_dir = os.path.join(defect_dir, "rgb")
                    if not os.path.isdir(rgb_dir):
                        continue

                    img_paths = glob.glob(os.path.join(rgb_dir, "*.png"))
                    img_paths.sort()

                    # 训练阶段只关心 good 样本，一般也只有 good
                    # 都视为 label=0，无 gt
                    self.img_paths["train"][category].extend(img_paths)
                    self.gt_paths["train"][category].extend([None] * len(img_paths))
                    self.labels["train"][category].extend([0] * len(img_paths))

        # ------------------ 测试部分：test ------------------
        for category in self.categories:
            obj_dir = os.path.join(self.root, category)
            test_dir = os.path.join(obj_dir, "test")
            if not os.path.isdir(test_dir):
                continue

            defect_types = os.listdir(test_dir)
            for defect_type in defect_types:
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue

                rgb_dir = os.path.join(defect_dir, "rgb")
                if not os.path.isdir(rgb_dir):
                    continue

                img_paths = glob.glob(os.path.join(rgb_dir, "*.png"))
                img_paths.sort()

                if defect_type == "good":
                    # 正常样本，无 gt
                    self.img_paths["test"][category].extend(img_paths)
                    self.gt_paths["test"][category].extend([None] * len(img_paths))
                    self.labels["test"][category].extend([0] * len(img_paths))
                else:
                    # 异常样本，gt 在每个 defect 子目录下的 gt/*.png
                    gt_dir = os.path.join(defect_dir, "gt")
                    gt_paths = glob.glob(os.path.join(gt_dir, "*.png"))
                    gt_paths.sort()

                    assert len(img_paths) == len(
                        gt_paths
                    ), f"[Test] RGB/GT 数量不一致: {category}, defect={defect_type}"

                    self.img_paths["test"][category].extend(img_paths)
                    self.gt_paths["test"][category].extend(gt_paths)
                    self.labels["test"][category].extend([1] * len(img_paths))

    # -------------------------------------------------------------
    # 跟 2D MVTecDataset 一样的 update(category) 接口
    # -------------------------------------------------------------
    def update(self, category=None):
        self.category = category
        self.cur_img_paths, self.cur_gt_paths = [], []
        self.cur_img_labels, self.cur_img_categories = [], []

        phase = "train" if self.train else "test"

        if self.category is not None:
            # 单一类别
            self.cur_img_paths = self.img_paths[phase][self.category]
            self.cur_gt_paths = self.gt_paths[phase][self.category]
            self.cur_img_labels = self.labels[phase][self.category]
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            # 全部类别拼在一起
            for c in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][c])
                self.cur_gt_paths.extend(self.gt_paths[phase][c])
                self.cur_img_labels.extend(self.labels[phase][c])
                self.cur_img_categories.extend(
                    [c] * len(self.img_paths[phase][c])
                )

        # few-shot 只在 train 有意义
        if self.train and self.fewshot != 0:
            randidx = np.random.choice(
                len(self.cur_img_paths), size=self.fewshot, replace=False
            )
            self.cur_img_paths = [self.cur_img_paths[idx] for idx in randidx]
            self.cur_gt_paths = [self.cur_gt_paths[idx] for idx in randidx]
            self.cur_img_labels = [self.cur_img_labels[idx] for idx in randidx]
            self.cur_img_categories = [self.cur_img_categories[idx] for idx in randidx]

    # -------------------------------------------------------------
    def __len__(self):
        return len(self.cur_img_paths)

    # -------------------------------------------------------------
    def __getitem__(self, idx):
        category = self.cur_img_categories[idx]
        img_path = self.cur_img_paths[idx]
        label = self.cur_img_labels[idx]

        img = Image.open(img_path).convert("RGB")

        # 构造 gt：如果没有路径，就用全 0；否则读 png
        if self.cur_gt_paths[idx] is not None:
            gt = np.array(Image.open(self.cur_gt_paths[idx]))
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)
        gt = Image.fromarray(gt)

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        # 注意：这里返回形式要和 MVTecDataset 保持一致
        return img, label, gt, category, img_path
