import os
from PIL import Image
import numpy as np
import random
from torchvision import transforms

def augment_image(img, augment_idx):

    if augment_idx == 0:  # 第一次增强：水平翻转
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif augment_idx == 1:  # 第二次增强：随机旋转
        img = img.rotate(random.uniform(-30, 30))  # 旋转角度在-30到30度之间
    elif augment_idx == 2:  # 第三次增强：添加噪声
        img = add_noise(img)
    elif augment_idx == 3:  # 第四次增强：随机裁剪
        left = random.randint(0, img.width // 4)
        top = random.randint(0, img.height // 4)
        right = random.randint(img.width // 2, img.width)
        bottom = random.randint(img.height // 2, img.height)
        img = img.crop((left, top, right, bottom))
    elif augment_idx == 4:  # 第五次增强：随机色调变化
        img = random_color_jitter(img)
    
    return img

def add_noise(img):
    img = np.array(img) / 255.0  # 将图像转换为0-1的浮点数
    noise = np.random.normal(0, 0.1, img.shape)  # 高斯噪声，均值为0，标准差为0.1
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0.0, 1.0)  # 保证像素值在0到1之间
    return Image.fromarray((noisy_img * 255).astype(np.uint8))  # 转回PIL格式

def random_color_jitter(img):

    color_jitter = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
    return color_jitter(img)

def augment_and_save_images(root_dir, save_dir, split='train', augment_times=5):
    img_dir = os.path.join(root_dir, split)
    classes = os.listdir(img_dir)

    # 创建保存增强图像的目录
    os.makedirs(save_dir, exist_ok=True)

    # 遍历每个类别文件夹
    for class_name in classes:
        class_folder = os.path.join(img_dir, class_name)
        print("class folder:", class_folder)

        # 忽略非文件夹（如.DS_Store等文件）
        if not os.path.isdir(class_folder):
            continue
        
        img_names = [f for f in os.listdir(class_folder) if f.endswith('.jpg')]

        # 为每个类别创建一个新文件夹
        new_class_folder = os.path.join(save_dir, class_name)
        os.makedirs(new_class_folder, exist_ok=True)

        # 遍历该类别下的每张图像
        for img_name in img_names:
            img_path = os.path.join(class_folder, img_name)

            img = Image.open(img_path).convert('RGB')
            original_img_name = img_name.replace('.jpg', '_0.jpg')
            original_img_path = os.path.join(new_class_folder, original_img_name)
            print("original image path:", original_img_path)
            img.save(original_img_path)  # 保存原始图像

            # 执行多次增强并保存
            for augment_idx in range(augment_times):
                augmented_img = augment_image(img, augment_idx)

                # 创建新的图像文件名：添加_1, _2, _3等后缀
                new_img_name = img_name.replace('.jpg', f'_{augment_idx+1}.jpg')
                new_img_path = os.path.join(new_class_folder, new_img_name)

                # 保存增广后的图像
                augmented_img.save(new_img_path)

# 调用此函数对所有训练集图像进行增广并保存到新文件夹中
root_dir='/home/stu2/image_classification/dataset'
save_dir='/home/stu2/image_classification/dataset/train_augmented'
os.makedirs(save_dir, exist_ok=True)
augment_and_save_images(root_dir, save_dir, split='train', augment_times=5)
