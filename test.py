import ultralytics
ultralytics.checks()

from ultralytics import YOLO
model = YOLO("models/best.pt")
model.predict(
    source="images/test/images",
    conf=0.85,              # 信心水準門限值
    save_txt=True,          # 存偵測結果(YOLO格式)
    save_conf=False,        # 存偵測結果(信心水準)
    save_crop=True,         # 存擷取物件影像
    visualize=False,        # 偵測過程特徵視覺化
    save=True,              # 存偵測結果影像
    device=0                # GPU?
)