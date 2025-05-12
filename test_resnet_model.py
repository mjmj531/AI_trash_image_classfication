# 用于测试模型性能，加载checkpointes以在测试集上测试模型

import torch
import torch.nn as nn
import os
import json
from torchvision import transforms
from PIL import Image
import numpy as np
from resnet_mine import *
from torchvision import models # 导入预训练模型,用于比较性能差异
from dataloader_for_resnet import CustomDataset, MyDataLoader

# 图像预处理for test model(不用再进行随机翻转等操作)
def get_transform(model_choose='mine'):
    if model_choose == 'mine':
        transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform
    else:
        weights = models.ResNet18_Weights.DEFAULT
        preprocess = weights.transforms()
        print("using torchvision preprocess", preprocess)
        return preprocess

# 读取所有测试集图片
def load_validation_imgs_labels(folder, transform, label_mapping_file):
    
    # 读取标签映射文件
    if os.path.exists(label_mapping_file):
        with open(label_mapping_file, 'r') as f:
            label_map = json.load(f)  # 加载JSON文件为字典
    else:
        raise FileNotFoundError(f"{label_mapping_file} not found.")
    
    images = []
    labels = []
    
    # 遍历类别文件夹，加载图片
    for subfolder in os.listdir(folder):
        subfolder_path = os.path.join(folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            # 获取该类别的标签，如果没有该类别的映射则跳过
            if subfolder in label_map:
                label = label_map[subfolder]
            else:
                print(f"Warning: '{subfolder}' not found in label mapping. Skipping.")
                continue
            
            # 加载该类下的所有图像
            for img_name in os.listdir(subfolder_path):
                if img_name.endswith(".jpg"):
                    img_path = os.path.join(subfolder_path, img_name)
                    try:
                        img = Image.open(img_path).convert('RGB')
                        img = transform(img)  # 进行预处理
                        images.append(img)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
    
    return torch.stack(images), torch.tensor(labels)

def evaluate(model, images, labels, criterion): 
    model.eval()  # 设为评估模式
    correct = 0
    total = 0
    running_loss = 0.0
    
    with torch.no_grad():  # 在验证过程中不计算梯度
        outputs = model(images)  # 对整个验证集进行预测
        loss = criterion(outputs, labels)
        print("outputs:", outputs)
        print("labels:", labels)
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)  # 获取每个样本的预测类别
        print("predicted:", predicted)  # 输出预测结果，检查其与真实标签的匹配情况
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 累加预测正确的样本数
    
    avg_loss = running_loss / len(labels)  # 平均损失
    accuracy = 100 * correct / total  # 准确率
    print(f"Total samples: {total}, Correct predictions: {correct}, Accuracy: {accuracy}%")
    
    return avg_loss, accuracy

if __name__ == "__main__":
    # 设置模型选择方式
    model_choice = input("请输入模型选择(1 或 2 或 3):\n1. 自定义模型(mine)\n2. torchvision中的预训练模型权重\n3. torchvision中的预训练模型的fine-tune\n请输入1或2或3: ")

    if model_choice == '1':
        model_choose = 'mine'  # 选择自定义模型
    elif model_choice == '2':
        model_choose = 'torchvision_model'  # 选择torchvision中的预训练模型(未经过fine-tune)
    elif model_choice == '3':
        model_choose = 'fine_tune'  # 选择fine-tune
    else:
        print("无效输入,请输入1或2或3")
        exit(1)  # 输入无效时退出程序
    print(f"选择{model_choose}模型")

    # 设置测试集路径
    tesr_dir = 'dataset/test'

    # 获取图像预处理
    transform = get_transform(model_choose)

    test_dataset = CustomDataset(
        root_dir='dataset', 
        split='test', 
        transform=transform,
        label_mapping_file='label_mapping_trash.json'  # 替换为实际的标签映射文件路径
    )

    test_loader = MyDataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

    criterion = nn.CrossEntropyLoss()  # 损失函数

    # 创建模型
    if model_choose == 'mine':
        model = resnet18_handwriting(num_classes=10)  # 使用自定义模型

        checkpoint_path = '/home/stu2/image_classification/checkpoints/resnet18_bs64_lr01_dropout_0_SGD_ReduceLROnPlateau_earlystopping/model_resnet18_bs64_epoch24_lr0.01000_acc0.89_best.pth'
        
        model.load_state_dict(torch.load(checkpoint_path))
        print("using mine training model: RESNET18 with my best checkpoint")
        
        # using mine training model: RESNET18 with my best checkpoint
        # Total samples: 2971, Correct predictions: 2633
        # Test Loss: 0.5227
        # Test Accuracy: 88.62%

    elif model_choose == 'torchvision_model':
        model = models.resnet18(weights='IMAGENET1K_V1')  # 使用torchvision预训练模型
        model.fc = nn.Linear(model.fc.in_features, 10)  # 修改全连接层以适应类别数
        
        print("using torchvision model: RESNET18 with IMAGENET1K_V1 weights")

        # using torchvision model: RESNET18 with IMAGENET1K_V1 weights
        # Total samples: 2971, Correct predictions: 250
        # Test Loss: 2.5084
        # Test Accuracy: 8.41%

    elif model_choose == 'fine_tune':
        # 加载 torchvision ResNet50 模型架构（预训练）,注意根据chekpoints对应修改模型层数
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)  
        # 修改全连接层以适应数据集类别数
        num_classes = 10 
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        checkpoint_path = 'checkpoints/fine-tune-model/resnet50_pretrained_1_bs256_lr01_dropout_0_SGD_ReduceLROnPlateau_earlystopping/model_resnet50_bs256_epoch11_lr0.00100_acc0.97_best.pth'

        model.load_state_dict(torch.load(checkpoint_path))
        print("using RESNET18 with imagenet-mini fine-tune model")
        # Total samples: 2971, Correct predictions: 2859
        # Test Loss: 0.1809
        # Test Accuracy: 96.23%
     
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # TODO:
    # 执行评估
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            print(f"Batch {batch_idx + 1}: Loss = {loss.item():.4f}, Accuracy = {(predicted == targets).sum().item() / targets.size(0) * 100:.2f}%")

    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    print(f"\n[Final Test Result]")
    print(f"Total samples: {total}, Correct predictions: {correct}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

