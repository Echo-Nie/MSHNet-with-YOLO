# Project Structure
The structure of the project is organized as follows:
```bash
YOLOTest_8_15/
├─ train.py  # Core training script
├─ data.yaml # Dataset configuration file
├─ download.py # Script for downloading models, making setup more convenient
└─ dataset/ # Directory for storing datasets
   ├─ train/
   │  ├─ image/   # Training images
   │  └─ labels/  # YOLO format labels: <class x y w h>, values are normalized coordinates
   ├─ val/
   │  ├─ image/
   │  └─ labels/
   └─ test/ 
      ├─ image/
      └─ labels/  # Can be left empty if no test labels are available

└─ runs      # Automatically generated 
```

# Dataset Naming Convention
It is **critical** that the directories are named exactly `images` and `labels`. YOLO relies on these specific names to generate the `label.cache` file. Incorrect naming will prevent cache generation and cause runtime errors.
```
└─ dataset/
   ├─ train/
   │  ├─ image/   
   │  └─ labels/  
   ├─ val/
   │  ├─ image/
   │  └─ labels/
   └─ test/ 
      ├─ image/
      └─ labels/ 
```

# Running the Model

## Direct Execution
To run the training script directly, use:
```bash
python train.py
```

## cmd

**Train:** To train the model via command line, use:

```bash
yolo train model=yolo11n.pt data=data.yaml epochs=100 imgsz=640 batch=64 project=runs/uavDetect name=yolo11n_uav4
```

**Val:** To evaluate the model on the validation set:

```bash
yolo val model=runs/uavDetect/yolo11n_uav4/weights/best.pt data=data.yaml imgsz=640
```

**Test:** To evaluate the model on the test set:

```bash
yolo val model=runs/uavDetect/yolo11n_uav4/weights/best.pt data=data.yaml split=test imgsz=640
```
