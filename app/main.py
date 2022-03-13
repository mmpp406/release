import base64
from flask import request
from flask import Flask
import os
import torch
from torchvision import transforms
import numpy as np
import cv2 as cv
from modeling.my_resnet18 import resnet18

app = Flask(__name__)


# 定义路由
@app.route("/photo", methods=['POST'])
def get_frame():
    # 接收图片
    upload_file = request.files['file']
    print(upload_file)
    # 获取图片名
    # file_name = upload_file.filename
    file_name = "tmp.png"
    # 文件保存目录（桌面）
    file_path = r'./'
    if upload_file:
        # 地址拼接
        file_paths = os.path.join(file_path, file_name)
        # 保存接收的图片到桌面
        upload_file.save(file_paths)
        # 对读入的图片进行预测
        image = cv.imread(file_paths)
        p_result = predict(image)

        response = str(p_result)
        return response

# 对输入的图片进行预测，输出其类别
def predict(image):
    weigths_path = r'./resnet18-100-regular.pth' # 训练好的权重保存路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 选择设备

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        ]
    )
    image = transform(image).view(1,3,224,224)
    image = image.to(device)

    # 加载网络
    net = resnet18(is_pretrained=False,num_classes=12)
    # 将训练好的权重加载到模型中
    net.load_state_dict(torch.load(weigths_path))
    net = net.to(device)

    # 预测结果
    _,p_resuls = net(image).max(1)
    return p_resuls.to('cpu')[0].numpy()


if __name__ == "__main__":
    
    app.run(debug=True,host='0.0.0.0',port=int(os.environ.get('PORT', 80)))
