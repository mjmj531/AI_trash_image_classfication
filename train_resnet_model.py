# 本文件实现了基于ResNet模型的图像分类训练流程
# 包括数据预处理、模型加载、数据集加载、训练、验证、保存模型等功能
# 也包括从0开始训练模型和fine-tune预训练模型的功能

import torch 
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm  # 用于显示进度条
import wandb
import matplotlib.pyplot as plt

from dataloader_for_resnet import CustomDataset, MyDataLoader

from model_mine_resnet import *
from torchvision import models # 用于加载预训练模型，进行fine-tune

import argparse
import os

# 定义早停机制：在验证集上准确率连续patience个epoch没有提升时，停止训练。
class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        :param patience: 若干个epoch内验证准确率没有提高时停止训练。
        :param verbose: 是否打印日志。
        :param delta: 用来定义准确率变化的阈值。若变化小于delta，认为没有提高。
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_accuracy = None
        self.early_stop = False

    def __call__(self, val_accuracy, model):
        if self.best_accuracy is None:
            self.best_accuracy = val_accuracy
        elif val_accuracy > self.best_accuracy + self.delta:
            self.best_accuracy = val_accuracy
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"Early stopping triggered. No improvement in accuracy for {self.patience} epochs.")

## -- 数据预处理操作 -- ##
def get_transforms_mine():
    # minecode
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),  # 指定使用双线性插值          # 调整图像大小
        transforms.CenterCrop(224),       # 中心裁剪到224x224
        transforms.ToTensor(),            # 转换为Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
        # transforms.RandomHorizontalFlip(), #
        # transforms.RandomRotation(10) #
    ])
    
    val_transforms = transform

    train_transforms = transforms.Compose([
        transform,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10)
    ])
    return train_transforms, val_transforms

def get_transforms_torchvision(weights):
    # torchvision models 加载 ResNet18 预训练权重以及预处理操作
    # weights = models.ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()

    train_transforms = transforms.Compose([
        preprocess,
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    val_transforms = preprocess

    return train_transforms, val_transforms


## -- 加载model -- ##
def get_mine_resnet_model(resnet_type, num_classes=10, use_dropout=0):
    # minecode
    if resnet_type == "resnet18":
        model = resnet18_handwriting(num_classes=num_classes, use_dropout=bool(use_dropout))
    elif resnet_type == "resnet34":
        model = resnet34_handwriting(num_classes=num_classes, use_dropout=bool(use_dropout))
    elif resnet_type == "resnet50":
        model = resnet50_handwriting(num_classes=num_classes, use_dropout=bool(use_dropout))
    else:
        raise ValueError("Invalid ResNet type.")

    return model

def get_model_torchvision_resnet(weights, resnet_model="resnet18", num_classes=10):
    if resnet_model == "resnet18":
        # 加载 ResNet18 预训练的模型，用于fine-tuning
        ### 这里也改了
        model = models.resnet18(weights=weights)  
    elif resnet_model == "resnet34":
        model = models.resnet34(weights=weights)
    elif resnet_model == "resnet50":
        model = models.resnet50(weights=weights)
    else:
        raise ValueError("Invalid ResNet type.")

    # 修改全连接层以适应数据集类别数
    # num_classes = 10  
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

## -- 创建数据集 -- ##
def get_data_loader(train_transforms, val_transforms, batch_size, dataset = "augmented"):

    # 我自定义的model加载数据集和torchvision models加载数据集逻辑相同，不做区分
    # 创建训练集和验证集的数据集实例，并指定 label_mapping_file 来存储/加载标签映射
    if dataset == "augmented":
        train_dataset = CustomDataset(
            root_dir='dataset', 
            split='train_augmented', 
            transform=train_transforms,
            label_mapping_file='label_mapping_trash.json' 
        )
    elif dataset == "origin":
        train_dataset = CustomDataset(
            root_dir='dataset', 
            split='train', 
            transform=train_transforms,
            label_mapping_file='label_mapping_trash.json' 
        )
    else:
        raise ValueError("Invalid dataset type.")
    
    val_dataset = CustomDataset(
        root_dir='dataset', 
        split='val', 
        transform=val_transforms,
        label_mapping_file='label_mapping_trash.json'  # 使用相同的文件来保证一致性
    )

    # 创建数据加载器
    train_loader = MyDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("len_train_loader: ",len(train_loader))
    val_loader = MyDataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print("len_val_loader: ",len(val_loader))

    return train_loader, val_loader

# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()  # 将模型设为训练模式
    running_loss = 0.0
    correct = 0
    total = 0
    

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        
        # 前向传播
        outputs = model(imgs)
        # 计算损失
        loss = criterion(outputs, labels)
        
        # 后向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失和准确率
        running_loss += loss.item() 
        # 计算当前batch的准确率
        _, predicted = torch.max(outputs, 1) # predicted.shape = (batch_size,)
        # print("predicted: ", predicted)
        # print("labels: ", labels)
        correct_batch = (predicted == labels).sum().item()
        total_batch = labels.size(0)
        accuracy_batch = correct_batch / total_batch
        print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy_batch:.2f}%")
        
        # 每个batch记录一次
        wandb.log({
            "batch_train_loss": loss.item(),
            "batch_train_accuracy": accuracy_batch,  # 当前batch的准确率
            "batch_idx": batch_idx + 1
        })
        
        # 累积整个训练集的准确率
        correct += correct_batch
        total += total_batch
    
    # 计算整个epoch的平均损失和准确率
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total  # 使用累积的正确预测数计算准确率

    return avg_loss, accuracy

# 修改验证函数，将predicted和labels的散点图绘制并保存到文件
def validate(model, val_loader, criterion, device):
    model.eval()  # 将模型设为评估模式
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []

    # # 确保保存目录存在
    # if save_dir is not None:
    #     os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():  # 在验证过程中不计算梯度
        for batch_idx, (imgs, labels) in enumerate(val_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(imgs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 统计损失和准确率
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 记录predicted和labels
            all_predicted.extend(predicted.cpu().numpy())  # 将预测结果转换为numpy并保存
            all_labels.extend(labels.cpu().numpy())  # 将真实标签转换为numpy并保存

            # 每个batch记录一次
            wandb.log({
                "batch_val_loss": loss.item(),
                "batch_val_accuracy": (predicted == labels).sum().item() / labels.size(0),
                "batch_val_idx": batch_idx + 1
            })

    # 计算整个验证集的损失和准确率
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total

    # # 创建文件名
    # plot_filename = os.path.join(save_dir, f"epoch{epoch}_predicted_vs_labels.png")

    # # 绘制 predicted vs labels 的散点图
    # plt.figure(figsize=(6, 6))
    # plt.scatter(all_labels, all_predicted, alpha=0.5, c='blue', label="Predicted vs Labels")
    # plt.plot([0, 10], [0, 10], 'r--', label="Ideal Prediction")  # 理想的对角线
    # plt.xlabel('Labels')
    # plt.ylabel('Predictions')
    # plt.title('Predicted vs Labels')
    # plt.legend()

    # # 将图像保存为文件
    # plt.savefig(plot_filename)
    # plt.close()  # 关闭图像，防止内存泄漏

    return avg_loss, accuracy

if __name__ == '__main__':
    # 改：project name config pathforpth

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train a ResNet model on ImageNet mini dataset.")
    parser.add_argument("--dataset", type=str, default="augmented", choices=["origin", "augmented"], help="Whether to use original ImageNet-mini dataset or dataset after augmentation")
    parser.add_argument("--epoch", type=int, default=50, help="The Maximum number of epochs for training and validation")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--resnet_model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"], help="ResNet model type")
    parser.add_argument("--dropout", type=int, choices=[0, 1], default=0, help="Whether to use dropout in the last fully connected layer (0: False, 1: True)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"], help="Optimizer type")
    parser.add_argument("--scheduler", type=str, default="ReduceLROnPlateau", choices=["ReduceLROnPlateau", "CosineAnnealingLR"], help="Learning rate scheduler if using SGD optimizer")
    parser.add_argument("--early_stopping", action="store_true", help="Use early stopping")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save model checkpoints")
    # parser.add_argument("--save_pic_dir", type=str, default="None", help="Directory to save validation picture for labels and predictions")
    # parser.add_argument("--save_pic_dir", type=str, nargs='?', default=None, help="Directory to save validation picture for labels and predictions. If not provided, pictures will not be saved.")
    parser.add_argument("--pretrained", type=int, choices=[0, 1], default=0, help="Load pretrained weights for fine-tuning (0: False, 1: True) for ResNet18 as an example)") #此处默认加载预训练的ResNet18模型，若要加载其他模型修改相应代码即可
    args = parser.parse_args()

    # 初始化 wandb
    wandb.init(
        project="img_classification_trash",
        # name=f"{args.resnet_model}_bs{args.batch_size}_lr{args.learning_rate}_wd1e-4_{args.optimizer}_small_dataset_no_early_stop_1221",
        name=f"{args.resnet_model}_bs{args.batch_size}_lr{args.learning_rate}_wd1e-4_{args.optimizer}_scheduler_{args.scheduler}_early_stop_{args.early_stopping}_dropout_{args.dropout}_pretrained_{args.pretrained}",
        entity="ma-j22-thu",
        config={
            "dataset": args.dataset,
            "pretrained": args.pretrained,
            "epochs": args.epoch,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": 1e-4,
            "resnet_model": args.resnet_model,
            "dropout": args.dropout,
            "optimizer": args.optimizer,
            "scheduler": args.scheduler,
            "early_stopping": args.early_stopping,
            "save_dir": args.save_dir
            # "save_pic_dir": args.save_pic_dir
        }
    )
    config = wandb.config

    # 确保保存目录存在
    save_dir = os.path.join("checkpoints", config.save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 数据预处理与加载模型
    if config.pretrained:
        #####补充一组数据
        if config.resnet_model == "resnet18":
            weight = models.ResNet18_Weights.IMAGENET1K_V1
        elif config.resnet_model == "resnet34":
            weight = models.ResNet34_Weights.IMAGENET1K_V1
        elif config.resnet_model == "resnet50":
            weight = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            raise ValueError("Invalid ResNet type.")

        train_transforms, val_transforms = get_transforms_torchvision(weight)
        
        model = get_model_torchvision_resnet(weight, config.resnet_model, num_classes=10)
    else:
        train_transforms, val_transforms = get_transforms_mine()
        model = get_mine_resnet_model(config.resnet_model, use_dropout=config.dropout)
    
    # 打印模型结构
    print(model)
    # 将模型移动到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # 加载数据集
    train_loader, val_loader = get_data_loader(train_transforms, val_transforms, batch_size=config.batch_size, dataset=config.dataset)

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 定义优化器
    # 使用 SGD 优化器
    if config.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)

        # 学习率调度器选择
        if config.scheduler == "ReduceLROnPlateau":
            # 平台学习率调度器 当验证损失没有改善时，将学习率减少为原来的 0.1 倍
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)
        elif config.scheduler == "CosineAnnealingLR":
            # Cosine Decay 学习率调度器，设置T_max为总训练轮数(如果用此调度器应当不设置early stopping)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=0, verbose=True)

    # 定义Adam优化器 自适应学习率
    elif config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # 训练与验证
    # 初始化早停机制，设定patience=12（即如果验证acc12个epoch没有改善，则停止训练）
    early_stopping = EarlyStopping(patience=12, verbose=True, delta=0.002) if config.early_stopping else None
    best_val_acc = 0.0

    for epoch in range(config.epochs):     
        print(f"Epoch {epoch+1}/{config.epochs}")
        
        # 训练阶段
        print("Training phase")
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        print(f"epoch {epoch+1}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
        
        # 验证阶段
        print("Validation phase")
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        # 记录学习率
        if config.optimizer == "SGD":
            current_lr = scheduler.get_last_lr()[0]
            if config.scheduler == "ReduceLROnPlateau":
                scheduler.step(val_loss)  # 根据验证集损失调整学习率
            elif config.scheduler == "CosineAnnealingLR":
                scheduler.step()  # 固定学习率，每训练一个epoch调整一次学习率

        elif config.optimizer == "Adam":
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            # Adam调整学习率时，不需要scheduler

        # 记录到 wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_accuracy,
            "val_loss": val_loss,
            "val_acc": val_accuracy,
            "learning_rate": current_lr
        })

        # 保存模型
        # torch.save(model.state_dict(), f"my_resnet_model_nv/my_resnet18_model_lst/resnet18_modelbslrunmatch_bs{config.batch_size}_epoch{epoch+1}_lr{current_lr:.5f}_acc{val_accuracy:.2f}.pth")
        # torch.save(model.state_dict(), f"my_resnet_model_nv/resnet18/resnet18_model_bs{config.batch_size}_epoch{epoch+1}_lr{current_lr:.5f}_acc{val_accuracy:.2f}_with_label_mapping.pth")
        
        # 记录最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            checkpoint_path = os.path.join(save_dir, f"model_{config.resnet_model}_bs{config.batch_size}_epoch{epoch+1}_lr{current_lr:.5f}_acc{val_accuracy:.2f}_best.pth")

            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

        # 检查是否需要进行早停
        if early_stopping:
            early_stopping(val_accuracy, model) 
            if early_stopping.early_stop:
                print("Early stopping triggered. Training has been stopped.")
                break

    wandb.finish()
