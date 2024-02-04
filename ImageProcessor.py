import cv2
import numpy as np
from ultralytics import YOLO

class ImageProcessor:
       
    def __init__(self, GUIManager, DobotManager):
        self.GUIManager = GUIManager
        self.DobotManager = DobotManager
        # モデルの読み込み
        self.model = YOLO("./model/last.pt")
        self.model.conf = 0.4
        
        
    
    def detect_object(self, frame):
        
        # 読み込んだモデルによる物体検知。
        result = self.model.predict(frame)
        
        # AutoによるDobotアーム処理中は処理しない。
        if self.DobotManager.duringAutoSeq == False :  
            if result:
                # 検出された物体のバウンディングボックスの座標を取得
                for box in result[0].boxes:

                    # バウンディングボックスのクラスIDを取得
                    class_id = box.cls
                    # ラベルのIDを保存
                    self.DobotManager.bboxlabel = int(class_id.item())
                    
                    # バウンディングボックスの幅、高さ取得
                    x, y, BBox_width, BBox_height = box.xywh[0]
                    print(f"BBox width : {BBox_width} ")
                    print(f"BBox height : {BBox_height} ")
                    
                    x_min, y_min, x_max, y_max = box.xyxy[0]
                    
                    # 中心座標を計算
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    
                    print(f"center_x : {center_x} ")
                    print(f"center_y : {center_y} ")
                    self.GUIManager.center_x = center_x
                    self.GUIManager.center_y = center_y
                          
                    # バウンディングボックスの幅、高さから物体のアームの回転量を決定する
                    arm_rotation = self.get_rotation(box, frame, BBox_width, BBox_height)
                    print(f"arm_rotation : {arm_rotation} ")
                    self.DobotManager.rotation = arm_rotation
                    
                    
                          
                    """            
                    #print(f"frame.shape : {frame.shape} ")
                    
                    # 角度の算出
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # バウンディングボックス内の画像を切り出し
                    cropped_frame = frame[max(y1-10, 0):min(y2+10, frame.shape[0]), max(x1-10, 0):min(x2+10, frame.shape[1])]
                    copy_frame = cropped_frame.copy()
                    #print(f"x1 : {x1} ")
                    #print(f"y1 : {y1} ")
                    #print(f"x2 : {x2} ")
                    #print(f"y2 : {y2} ")
                    #print(cropped_frame)
                    
                    gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                    _,img_trans = cv2.threshold(gray, self.GUIManager.threshold_scale.get(), 255, self.convert_THRESH_BINARY())
                    #cv2.imshow('before',img_trans)
                    
                    
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                    img_trans = cv2.morphologyEx(img_trans, cv2.MORPH_CLOSE, kernel, iterations=3)

                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

                    # 2値画像を収縮する。
                    #img_trans = cv2.morphologyEx(img_trans, cv2.MORPH_OPEN, kernel, iterations=4)
                    #cv2.imshow('After',img_trans)


                    contours, _ = cv2.findContours(img_trans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours :
                        # 最大面積の矩形を取得し、角度を計算
                        max_area = -1
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
                        
                        
                        #cv2.drawContours(copy_frame, [box], 0, (0, 255, 0), 2)
                        #cv2.imshow('drawContours',copy_frame)
                        #print(f"angle : {angle} ")
                        self.GUIManager.angle = angle
                    """
            else:
                self.GUIManager.center_x = None
                self.GUIManager.center_y = None
                self.GUIManager.angle = None
                self.DobotManager.bboxlabel = None 
        
        return result[0].plot()
    
    def convert_THRESH_BINARY (self) :
        
        if (self.GUIManager.bin_opt_combobox.get() == "THRESH_BINARY") :
            return cv2.THRESH_BINARY
        elif (self.GUIManager.bin_opt_combobox.get() == "THRESH_BINARY_INV") :
            return cv2.THRESH_BINARY_INV
        elif (self.GUIManager.bin_opt_combobox.get() == "THRESH_TRUNC") :
            return cv2.THRESH_TRUNC
        elif (self.GUIManager.bin_opt_combobox.get() == "THRESH_TOZERO") :
            return cv2.THRESH_TOZERO
        elif (self.GUIManager.bin_opt_combobox.get() == "THRESH_TOZERO_INV") :
            return cv2.THRESH_TOZERO_INV
        return cv2.THRESH_BINARY_INV
    
    def get_rotation(self, box, frame, BBox_width, BBox_height) :
        
        isSquare = self.is_almost_square(BBox_width, BBox_height)
        arm_rotation = -35
        
        if isSquare :
            # 斜め方向の向きが不明のため、45度ずらして再度判定する。
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            #cv2.imshow('before',frame)
            
            # 画像の中心座標を取得
            (h, w) = frame.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            
            # 45度回転するためのアフィン変換行列を取得
            M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
            
            # アフィン変換を適用
            rotated = cv2.warpAffine(frame, M, (w, h))
            
            #cv2.imshow('Rotated Image', rotated)
            
            result = self.model.predict(rotated)
            #cv2.imshow('after',result[0].plot())
            
            for box in result[0].boxes:
                # バウンディングボックスの幅、高さ取得
                x, y, BBox_width, BBox_height = box.xywh[0]

                # 縦、横を判定してアームの回転量を設定。
                if BBox_width < BBox_height :
                    # 縦方向の場合、右斜め方向にアームの角度を合わせる
                    arm_rotation = -80 
                else :
                    # 横方向の場合、左斜め方向にアームの角度を合わせる
                    arm_rotation = -10
                
            
        else :
            # 縦、横を判定してアームの回転量を設定。
            if BBox_width < BBox_height :
                # 縦方向
                arm_rotation = -35 
            else :
                # 横方向
                arm_rotation = -125
            
        return arm_rotation
    
    
    def is_almost_square(self, width, height, threshold=0.2):
        """
        縦と横の長さがほぼ等しいかどうかを判定する。

        :param width: 図形の横の長さ
        :param height: 図形の縦の長さ
        :param threshold: 許容する比率の差の最大値（デフォルトは20%）
        :return: ほぼ正方形ならTrue、そうでなければFalse
        """
        # 縦と横の長さの大きい方と小さい方を取得
        longer, shorter = max(width, height), min(width, height)
        
        # 長辺に対する短辺の比率を計算
        ratio = shorter / longer
        
        # 比率が1からの差が閾値以下であればほぼ正方形と判定
        return (1 - ratio) <= threshold