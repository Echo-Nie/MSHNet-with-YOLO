# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first
import argparse


class DetectionTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionTrainer

        args = dict(model='yolov8n.pt', data='coco8.yaml', epochs=3)
        trainer = DetectionTrainer(overrides=args)
        trainer.train()
        ```
    """

    def build_dataset(self, img_path, mode='train', batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == 'val', stride=gs)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        """Construct and return dataloader."""
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)  # return dataloader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images by scaling and converting to float."""
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        """Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)."""
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data['nc']  # attach number of classes to model
        self.model.names = self.data['names']  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        # TODO: self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a YOLO detection model."""
        model = DetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Returns a DetectionValidator for YOLO model validation."""
        self.loss_names = 'box_loss', 'cls_loss', 'dfl_loss'
        return yolo.detect.DetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def label_loss_items(self, loss_items=None, prefix='train'):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Returns a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train and optimize YOLO model given training data and device."""
    model = cfg.model
    data = cfg.data  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ''

    # 从 cfg 对象中提取其他参数，如果存在的话
    # 例如: epochs, batch, imgsz 等
    # 这里假设 DetectionTrainer 的 overrides 可以接受一个字典
    overrides = {
        'model': model,
        'data': data,
        'device': device,
        # 将 cfg 中的其他相关参数添加到 overrides 字典中
        # 例如: 'epochs': cfg.epochs, 'batch': cfg.batch, etc.
    }
    # 过滤掉值为 None 的参数，避免覆盖默认值
    overrides = {k: v for k, v in cfg.__dict__.items() if v is not None}

    if use_python:
        from ultralytics import YOLO
        # 直接传递 **overrides 会导致参数名冲突，需要调整
        # YOLO().train() 期望的是关键字参数，而不是一个字典
        # 需要根据 YOLO().train() 的签名来传递参数
        # 示例：YOLO(model).train(data=data, device=device, epochs=cfg.epochs, ...)
        # 这里暂时保持原样，但需要注意 use_python=True 的情况可能需要调整
        YOLO(model).train(**overrides)
    else:
        trainer = DetectionTrainer(overrides=overrides)
        trainer.train()


if __name__ == '__main__':
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=DEFAULT_CFG.model, help='model path')
    parser.add_argument('--data', type=str, required=True, help='dataset yaml path') # data 参数通常是必须的
    parser.add_argument('--epochs', type=int, default=DEFAULT_CFG.epochs, help='number of epochs')
    parser.add_argument('--batch', type=int, default=DEFAULT_CFG.batch, help='batch size')
    parser.add_argument('--imgsz', type=int, default=DEFAULT_CFG.imgsz, help='image size')
    parser.add_argument('--device', type=str, default=DEFAULT_CFG.device, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 添加其他你需要的参数...
    # parser.add_argument('--weights', type=str, default=DEFAULT_CFG.weights, help='initial weights path')
    # parser.add_argument('--cfg', type=str, default=DEFAULT_CFG.cfg, help='model.yaml path')
    # parser.add_argument('--hyp', type=str, default=DEFAULT_CFG.hyp, help='hyperparameters path')
    # ... 更多参数

    # 解析已知参数，忽略未知参数，这样可以兼容 ultralytics 内部使用的其他参数
    opt, _ = parser.parse_known_args()

    # 使用解析的参数更新 cfg 对象
    # 注意：这里直接修改 DEFAULT_CFG 可能不是最佳实践，
    # 更好的方式是创建一个新的配置对象或字典
    cfg_override = {
        'model': opt.model,
        'data': opt.data,
        'epochs': opt.epochs,
        'batch': opt.batch,
        'imgsz': opt.imgsz,
        'device': opt.device,
        # 添加其他解析的参数
    }

    # 更新 DEFAULT_CFG 或创建一个新的配置对象传递给 train 函数
    # 这里我们创建一个新的字典传递给 DetectionTrainer
    # 注意：train 函数本身也接受 cfg 参数，但其内部逻辑是基于 overrides 字典
    # 为了保持一致性，我们直接构造 overrides 字典
    trainer = DetectionTrainer(overrides=cfg_override)
    trainer.train()
