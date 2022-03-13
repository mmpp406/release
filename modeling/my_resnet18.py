import torchvision.models as models
import torch
from torch import nn

def resnet18(is_pretrained, num_classes):
    """
    使用pytorch内置的网络结构构造模型
    :param num_classes: int型，分类的数量
    :param is_pretrained: bool类型，True表示使用预训练模型
    :return:返回resnet18
    """
    my_model = models.resnet18(pretrained=is_pretrained,progress=True) # 由于预训练权重模型最后一个全连接层的尺寸为1000，所以不能直接输入num_classes
    my_model.fc = nn.Linear(in_features=my_model.fc.in_features,out_features=num_classes) # 修改模型最后一个全连接层的尺寸
    nn.init.xavier_uniform_(my_model.fc.weight) # 初始化最后一个全连接层的权重

    return my_model

if __name__ == "__main__":
    model = resnet18(is_pretrained=True,num_classes=12)
    x = torch.rand(size=(1,3,224,224)) # 输入必须是3通道
    y = model(x)
    print(y.shape)
    print(y)