## 简要的文件说明

### files
- `data_split.py`：将原先的数据集随机划分为训练集，验证集和测试集。
- `label_mapping_trash.json`：存放类别到id的映射关系，直接加载避免重复计算开销。

- `data_augment.py`: 将原先数据集进行增广以输入ResNet模型。
- `dataloader_for_resnet.py`: 使用ResNet模型加载数据的数据集类，支持批量加载。

- `model_mine_resnet.py`: 自定义ResNet模型。
- `train_resnet_model.py`: 使用ResNet模型进行训练和验证的代码。
- `test_resnet_model.py`: 加载`checkpoints`文件夹和`fine-tune-model`文件夹中的最好结果的模型在测试集上进行测试。

- `visualization_model.py`: 使用torchviz库可视化ResNet模型的结构。不影响训练和测试过程。

- `run_resnet.sh`：运行`train_resnet_model.py`以及记录结果的脚本。

### folders
- `data`文件夹：存放训练集、验证集。其中，`data/imagenet_mini/train`是我增广后的训练集，`data/imagenet_mini/train_origin`是原先的训练集，`data/imagenet_mini/val`是原先的验证集。
- `checkpoints`文件夹：存放从零开始训练的一些典型参数得到的每次验证集的最好结果的模型weights and biases，用于在`model_resnet_test.py`中加载进行测试。
- `checkpoints/fine-tune-model`文件夹: 存放从torchvision加载的预训练resnet18和resnet50并进行微调后的模型的weights and biases，用于在`model_resnet_test.py`中加载进行测试，与从零开始训练的模型进行效果对比。


