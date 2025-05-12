import os
import json
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, label_mapping_file='label_mapping_trash.json'):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.label_mapping_file = label_mapping_file

        # 设置图片路径
        self.img_dir = os.path.join(self.root_dir, split)

        # 加载类别文件夹并排序
        self.classes = sorted(os.listdir(self.img_dir))

        self.image_paths = []
        self.ids = []  # 用于储存类别文件夹名

        # 如果 label_mapping.json 文件存在，直接加载；否则生成并保存
        if os.path.exists(self.label_mapping_file):
            with open(self.label_mapping_file, 'r') as f:
                self.label_mapping = json.load(f)
            print(f"Loaded label mapping from {self.label_mapping_file}")
        else:
            # 映射类别文件夹到数字标签
            self.label_mapping = {class_name: idx for idx, class_name in enumerate(self.classes)}
            # 保存 label_mapping 为 JSON 文件
            with open(self.label_mapping_file, 'w') as f:
                json.dump(self.label_mapping, f, indent=4)
            print(f"Generated and saved label mapping to {self.label_mapping_file}")

        # 打印 label_mapping
        print("Label Mapping: ", self.label_mapping)

        # 遍历所有类别文件夹，加载图像路径
        for class_name in self.classes:
            class_folder = os.path.join(self.img_dir, class_name)
            if os.path.isdir(class_folder):  # 确保是文件夹而不是其他类型
                for img_name in os.listdir(class_folder):
                    if img_name.endswith(".jpg"):
                        img_path = os.path.join(class_folder, img_name)
                        self.image_paths.append(img_path)
                        self.ids.append(class_name)  # 存储类别文件夹名

    def __len__(self):
        return len(self.image_paths)  # 返回图像的总数

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        class_name = self.ids[idx]

        # 获取标签
        label = self.label_mapping[class_name]  # 显式映射类别到标签

        # 加载图像
        img = Image.open(img_path).convert('RGB')

        # 对图像进行预处理（包括裁剪、归一化等）
        if self.transform:
            img = self.transform(img)

        return img, label
    
    def get_labels(self, indices):
        labels = []
        for idx in indices:
            class_name = self.ids[idx]
            label = self.label_mapping[class_name]
            labels.append(label)
        labels = np.stack(labels, axis=0)
        return labels

# 自己实现的数据加载器
class  MyDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indexes = list(range(len(dataset)))
        if shuffle:
            random.shuffle(self.indexes)

    def __iter__(self):
        # 如果需要丢弃最后的批次，并且剩余的样本数小于batch_size，跳过
        if self.drop_last:
            num_batches = len(self.dataset) // self.batch_size
        else:
            num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size  # 向上取整

        for i in range(num_batches):
            batch_indexes = self.indexes[i*self.batch_size:(i+1)*self.batch_size]
            batch_imgs = []
            batch_labels = []
            for idx in batch_indexes:
                img, label = self.dataset[idx]
                batch_imgs.append(img)
                batch_labels.append(label)

            # 将图片堆叠成一个batch
            batch_imgs = torch.stack(batch_imgs)
            batch_labels = torch.tensor(batch_labels)
            yield batch_imgs, batch_labels

    def __len__(self):
        # 如果 drop_last 为 True，则丢弃最后一个小于 batch_size 的 batch
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return len(self.dataset) // self.batch_size + (1 if len(self.dataset) % self.batch_size != 0 else 0)

# 数据预处理操作
transform = transforms.Compose([
    transforms.Resize(256),           # 调整图像大小
    transforms.CenterCrop(224),       # 中心裁剪到224x224
    transforms.ToTensor(),            # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# # 创建训练集和验证集的数据加载器
# train_loader = MyDataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
# val_loader = MyDataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False)

# # 迭代训练集
# for imgs, labels in train_loader:
#     print(imgs.shape)  # 打印图像数据的形状
#     print(labels)  # 打印标签

# # 迭代验证集
# for imgs, labels in val_loader:
#     print(imgs.shape)  # 打印图像数据的形状
#     print(labels)  # 打印标签
