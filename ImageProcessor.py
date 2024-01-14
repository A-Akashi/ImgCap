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
            #frame = result[0].plot()
                      
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
                               
                
                # 角度の算出
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # バウンディングボックス内の画像を切り出し
                cropped_frame = frame[y1-10:y2+10, x1-10:x2+10]
                copy_frame = cropped_frame.copy()

                # バウンディングボックス内の画像で輪郭検出
                # 緑成分だけを抽出
                #green_channel = cropped_frame[:, :, 1]
                # 2値化
                #_, img_trans = cv2.threshold(green_channel, self.GUIManager.threshold_scale.get(), 255, cv2.THRESH_BINARY_INV)
                
                gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                _,img_trans = cv2.threshold(gray, self.GUIManager.threshold_scale.get(), 255, cv2.THRESH_BINARY_INV)
                cv2.imshow('before',img_trans)
                
                
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                img_trans = cv2.morphologyEx(img_trans, cv2.MORPH_CLOSE, kernel, iterations=3)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                # 2値画像を収縮する。
                #img_trans = cv2.morphologyEx(img_trans, cv2.MORPH_OPEN, kernel, iterations=4)
                cv2.imshow('After',img_trans)


                contours, _ = cv2.findContours(img_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours :
                    # 最大面積の矩形を取得し、角度を計算
                    max_area = 0
                    max_contour = None
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > max_area:
                            max_area = area
                            max_contour = contour
                    
                    rect = cv2.minAreaRect(max_contour)
                    angle = rect[2]
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    
                    
                    cv2.drawContours(copy_frame, [box], 0, (0, 255, 0), 2)
                    cv2.imshow('drawContours',copy_frame)
                    print(f"angle : {angle} ")
                    self.GUIManager.angle = angle
        else:
            self.GUIManager.center_x = None
            self.GUIManager.center_y = None
            self.GUIManager.angle = None
        
        return result[0].plot()
    