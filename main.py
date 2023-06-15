# pip uninstall pip setuptools
# pip3 install --upgrade pip
# pip3 install -upgrade setuptools
# pip3 install torch torchvision torchaudio--index-url https://download. pytorch.org/whl/cu118
# git clone https://github.com/ultralytics/ultralytics/ultralytics
# pip install ultralytics

import ultralytics
from ultralytics import YOLO
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    model = YOLO(' models/yolov8s.pt')
    model.to('cuda')

    results = model.train(
        data="20230613_coffee_pot_148.yaml",     # 指定訓練任務檔
        workers=1,              # 為解決GPU抓不到 (失敗)
        device=0,               # 為解決GPU抓不到 (失敗)
        imgsz=256,              # 影像大小
        epochs=300,             # 訓練世代數
        patience=50,            # 等待世代數
        batch=-1,               # 為解決GPU抓不到 (成功)
        project='YOLOv8',
        name='cao_coffee;')
