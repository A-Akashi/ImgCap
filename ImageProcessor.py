import cv2
import numpy as np
from ultralytics import YOLO

class ImageProcessor:
    def __init__(self, GUIManager):
        self.GUIManager = GUIManager
        # モデルの読み込み
        self.model = YOLO("./model/last.pt")
        self.model.conf = 0.4
        
    def detect_object(self, frame):
        
        # 読み込んだモデルによる物体検知。
        result = self.model.predict(frame)
        if result:
            frame = result[0].plot()
                      
            # 検出された物体のバウンディングボックスの座標を取得
            for box in result[0].boxes:

                x_min, y_min, x_max, y_max = box.xyxy[0]
                
                # 中心座標を計算
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                print(f"center_x : {center_x} ")
                print(f"center_y : {center_y} ")
                self.GUIManager.center_x = center_x
                self.GUIManager.center_y = center_y
        else:
            self.GUIManager.center_x = None
            self.GUIManager.center_y = None
                      
        return frame
    