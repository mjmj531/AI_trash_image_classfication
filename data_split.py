import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=-1):
    random.seed(seed)
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratio sum must be 1.0"

    for split in ['train', 'val', 'test']:
        (dest_dir / split).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in source_dir.iterdir() if d.is_dir()]

    # 添加 tqdm 外层进度条显示每个类
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        total = len(images)
        n_train = int(total * train_ratio)
        n_val = int(total * val_ratio)

        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:]
        }

        for split, split_images in splits.items():
            class_dest = dest_dir / split / class_dir.name
            class_dest.mkdir(parents=True, exist_ok=True)

            # tqdm 内层复制图片进度条
            for img_path in tqdm(split_images, desc=f"{class_dir.name} -> {split}", leave=False):
                shutil.copy(img_path, class_dest / img_path.name)

    print(f"\n Dataset split completed and saved to {dest_dir}")

source_dir = "/home/stu2/image_classification/Garbage-Classification/garbage-dataset"  
dest_dir = "dataset"             
split_dataset(source_dir, dest_dir)
