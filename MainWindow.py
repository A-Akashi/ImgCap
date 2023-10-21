import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk
from Controller import CameraController
from ImageProcessor import ImageProcessor
from GUIManager import GUIManager   

class MainWindow:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.setUp()
        self.update()
        self.window.mainloop()

    def setUp(self):
        # カメラ制御クラス
        self.camera_controller = CameraController()
        
        # 画像処理クラス
        self.ImgProc = ImageProcessor()
        
        # TKinterなどの部品を制御するクラス
        self.GUIManager = GUIManager(self.window, self.camera_controller)
       

    def update(self):
        ret, frame = self.camera_controller.get_Frame()
        if self.camera_controller.isCameraOpen() and ret:
            # 画像処理実行。
            frame = self.exec_Img_Proc(frame)
            
            self.photo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.image = Image.fromarray(self.photo)
            self.imageTk = ImageTk.PhotoImage(image=self.image)
            self.GUIManager.canvas.delete(self.GUIManager.current_image)
            self.GUIManager.current_image = self.GUIManager.canvas.create_image(0, 0, image=self.imageTk, anchor=tk.NW)
        self.window.after(10, self.update)

    def exec_Img_Proc(self, frame):
        # 特徴点表示設定時。
        if self.GUIManager.apply_DispFeature:
            frame = self.ImgProc.display_feature_points(frame)
            
        # 物体検知設定時。
        if self.GUIManager.apply_detect and self.GUIManager.reference_image_path:
            frame = self.ImgProc.detect_object(frame, self.GUIManager.reference_image_path)

        return frame

    def __del__(self):
        self.camera_controller.release()