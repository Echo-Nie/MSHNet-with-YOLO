from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

if __name__ == '__main__':
    # 官方权重下载：https://docs.ultralytics.com/zh/models/yolov9/#performance-on-ms-coco-dataset
    # 可以在这个网站里面看他们的名字，然后你想要哪个模型就在download。py里面改一下名字就行
    model = YOLO("yolo11n.pt")
    model.to(device)

    # 训练+验证
    model.train(data="data.yaml", epochs=40, imgsz=640, batch=64,
                project='runs/uavDetect',
                name='yolo11n_uav')  # train the model
    metrics = model.val()  # evaluate model performance on the validation set

    # 如果没有test数据集的话，直接把下面的代码注释掉就行
    test_results = model.val(data="data.yaml", split='test')
