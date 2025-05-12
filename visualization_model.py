from model_mine_resnet import resnet18_handwriting
from torchviz import make_dot
import torch

# 定义 ResNet18 模型
model = resnet18_handwriting(num_classes=100)

# 创建随机输入张量
input_tensor = torch.randn(1, 3, 224, 224)  # batch_size=1, 通道数=3, 图像大小=224x224

# 前向传播得到输出
output = model(input_tensor)

# 使用 torchviz 绘制计算图
dot = make_dot(output, params=dict(model.named_parameters()))

# 保存为 PDF 文件
dot.format = 'pdf'
dot.render("resnet18_graph")