import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

class GUIManager:
    def __init__(self, window, controller, dobotManager):
        self.window = window
        self.camera_controller = controller
        self.dobotManager = dobotManager
        self.auto_exposure_var = tk.IntVar(value=1)
        self.auto_wh_var = tk.IntVar(value=1) 
        self.auto_focus_var = tk.IntVar(value=1)
        self.apply_detect = False
        self.current_image = None
        self.center_x = None
        self.center_y = None
        self.createCanvas()
        self.createButtons()
        
    
    def createCanvas(self):
        self.canvas = tk.Canvas(self.window, 
                                width=self.camera_controller.vid.get(cv2.CAP_PROP_FRAME_WIDTH), 
                                height=self.camera_controller.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()
    
    def createButtons(self):
        self.btn_frame = tk.Frame(self.window)
        self.btn_frame.pack(side=tk.LEFT, pady=15)

        self.btn_open = tk.Button(self.btn_frame, text="Connect", width=10, command=self.open_camera)
        self.btn_open.pack(side=tk.LEFT, padx=5)

        self.btn_close = tk.Button(self.btn_frame, text="Disconnect", width=10, command=self.close_camera)
        self.btn_close.pack(side=tk.LEFT, padx=5)

        self.btn_settings = tk.Button(self.btn_frame, text="Settings", width=10, command=self.open_settings)
        self.btn_settings.pack(side=tk.LEFT, padx=5)
     
        self.btn_sift = tk.Button(self.window, text="Object Detection", width=15, command=self.toggle_detect)
        self.btn_sift.pack(pady=20, side=tk.RIGHT, padx=5)
     
        self.btn_auto = tk.Button(self.window, text="Auto", width=10, command=self.auto_detect)
        self.btn_auto.pack(pady=20, side=tk.RIGHT, padx=5)   
        
        self.btn_auto = tk.Button(self.window, text="Pop balloons", width=10, command=self.pop_balloons)
        self.btn_auto.pack(pady=20, side=tk.RIGHT, padx=5)   
       
       
        
    
    # def updateCanvas(self):
    
    def open_camera(self):
        self.camera_controller.connect()

    def close_camera(self):
        self.camera_controller.disconnect()
        self.canvas.delete(self.current_image)

    def open_settings(self):
        settings_dialog = tk.Toplevel(self.window)
        settings_dialog.title("Settings")
        window_width = self.window.winfo_width()
        window_height = self.window.winfo_height()
        settings_dialog.geometry(f"{window_width}x{window_height}")
        
        
        # 自動露光設定。
        self.auto_exp_frame = tk.Frame(settings_dialog)
        self.auto_exp_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_auto_exposure = tk.Label(self.auto_exp_frame, text="Auto Exposure : ", width=20 , anchor="e" ) 
        self.lbl_auto_exposure.pack(side=tk.LEFT)

        self.btn_auto_exposure_on = tk.Button(self.auto_exp_frame, text="On", width=8, relief=tk.SUNKEN, command=self.set_auto_exposure_on)
        self.btn_auto_exposure_on.pack(side=tk.LEFT, padx=5)

        self.btn_auto_exposure_off = tk.Button(self.auto_exp_frame, text="Off", width=8, relief=tk.RAISED, command=self.set_auto_exposure_off)
        self.btn_auto_exposure_off.pack(side=tk.LEFT, padx=5)
        
        # 自動ホワイトバランス設定。
        self.auto_wh_frame = tk.Frame(settings_dialog)
        self.auto_wh_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_auto_wh = tk.Label(self.auto_wh_frame, text="Auto White Balance : "  ,  width=20 , anchor="e")
        self.lbl_auto_wh.pack(side=tk.LEFT)

        self.btn_auto_wh_on = tk.Button(self.auto_wh_frame, text="On", width=8, relief=tk.SUNKEN, command=self.set_auto_wh_on)
        self.btn_auto_wh_on.pack(side=tk.LEFT, padx=5)

        self.btn_auto_wh_off = tk.Button(self.auto_wh_frame, text="Off", width=8, relief=tk.RAISED, command=self.set_auto_wh_off)
        self.btn_auto_wh_off.pack(side=tk.LEFT, padx=5)
        
        # 自動フォーカス設定。
        self.auto_focus_frame = tk.Frame(settings_dialog)
        self.auto_focus_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_auto_focus = tk.Label(self.auto_focus_frame, text="Auto Focus : " , width=20 ,anchor="e")
        self.lbl_auto_focus.pack(side=tk.LEFT)

        self.btn_auto_focus_on = tk.Button(self.auto_focus_frame, text="On", width=8, relief=tk.SUNKEN, command=self.set_auto_focus_on)
        self.btn_auto_focus_on.pack(side=tk.LEFT, padx=5)

        self.btn_auto_focus_off = tk.Button(self.auto_focus_frame, text="Off", width=8, relief=tk.RAISED, command=self.set_auto_focus_off)
        self.btn_auto_focus_off.pack(side=tk.LEFT, padx=5)
        
        # 二値化オプション設定(THRESH設定)。
        self.bin_opt_frame = tk.Frame(settings_dialog)
        self.bin_opt_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_bin_opt = tk.Label(self.bin_opt_frame, text="Binarization Option : ", width=20 ,anchor="e")
        self.lbl_bin_opt.pack(side=tk.LEFT)
        self.bin_opt_combobox = ttk.Combobox(self.bin_opt_frame, values=["THRESH_BINARY", "THRESH_BINARY_INV", "THRESH_TRUNC", "THRESH_TOZERO", "THRESH_TOZERO_INV"])
        self.bin_opt_combobox.set("THRESH_BINARY_INV")  # 初期値
        self.bin_opt_combobox.pack(side=tk.LEFT, padx=5)
        
        
        # 二値化閾値を調整するためのスライダーを作成
        self.scale_frame = tk.Frame(settings_dialog)
        self.scale_frame.pack(pady=10, anchor=tk.W)        
        self.lbl_Threshold = tk.Label(self.scale_frame, text="Binarization Threshold : ", width=20 ,anchor="e")
        self.lbl_Threshold.pack(side=tk.LEFT)
        self.threshold_scale = tk.Scale(self.scale_frame, from_=0, to=255, orient="horizontal")
        self.threshold_scale.set(126)  # デフォルトの閾値を設定
        self.threshold_scale.pack(side=tk.LEFT, padx=5)
        
        
        # Manual操作の設定項目
        self.manualSetting_frame = tk.Frame(settings_dialog)
        self.manualSetting_frame.pack(pady=10, anchor=tk.W)  
        self.lbl_ManualSetting= tk.Label(self.manualSetting_frame, text="Manual Detect Settings ==================", width=42 ,anchor="e")
        self.lbl_ManualSetting.pack(side=tk.LEFT)
 
        # 移動先座標X設定
        self.DestX_frame = tk.Frame(settings_dialog)
        self.DestX_frame.pack(pady=10, anchor=tk.W)        
        self.lbl_DestX = tk.Label(self.DestX_frame, text="Dest. X Coordinate : ", width=20 ,anchor="e")
        self.lbl_DestX.pack(side=tk.LEFT)
        self.destXEntry = tk.Entry(self.DestX_frame, width=12)
        self.destXEntry.pack(side=tk.LEFT, padx=5)  
        self.destXEntry.insert(0, "170") #初期値。

        # 移動先座標Y設定
        self.DestY_frame = tk.Frame(settings_dialog)
        self.DestY_frame.pack(pady=10, anchor=tk.W)        
        self.lbl_DestY = tk.Label(self.DestY_frame, text="Dest. Y Coordinate : ", width=20 ,anchor="e")
        self.lbl_DestY.pack(side=tk.LEFT)
        self.destYEntry = tk.Entry(self.DestY_frame, width=12)
        self.destYEntry.pack(side=tk.LEFT, padx=5)  
        self.destYEntry.insert(0, "-120") #初期値。
        
        # 移動先座標Z設定
        self.DestZ_frame = tk.Frame(settings_dialog)
        self.DestZ_frame.pack(pady=10, anchor=tk.W)        
        self.lbl_DestZ = tk.Label(self.DestZ_frame, text="Dest. Z Coordinate : ", width=20 ,anchor="e")
        self.lbl_DestZ.pack(side=tk.LEFT)
        self.destZEntry = tk.Entry(self.DestZ_frame, width=12)
        self.destZEntry.pack(side=tk.LEFT, padx=5)  
        self.destZEntry.insert(0, "20") #初期値。
        
        # グリッパー回転量設定
        self.Rot_frame = tk.Frame(settings_dialog)
        self.Rot_frame.pack(pady=10, anchor=tk.W)        
        self.lbl_Rot = tk.Label(self.Rot_frame, text="Gripper Rotation : ", width=20 ,anchor="e")
        self.lbl_Rot.pack(side=tk.LEFT)
        self.rotEntry = tk.Entry(self.Rot_frame, width=12)
        self.rotEntry.pack(side=tk.LEFT, padx=5)  
        self.rotEntry.insert(0, "-35") #初期値。
        
        # 移動ボタン。
        self.MoveImg_frame = tk.Frame(settings_dialog)
        self.MoveImg_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_Move_Image = tk.Label(self.MoveImg_frame, text="" , width=20 ,anchor="e")
        self.lbl_Move_Image.pack(side=tk.LEFT)
        
        self.btn_Move_Image = tk.Button(self.MoveImg_frame, text="Move", width=10, command=self.move)
        self.btn_Move_Image.pack(side=tk.LEFT, padx=5)
        
        # アラームリセットボタン。
        self.btn_Move_Image = tk.Button(self.MoveImg_frame, text="Reset Alarm", width=10, command=self.reset_alarm)
        self.btn_Move_Image.pack(side=tk.LEFT, padx=5)
        
        # 掴むボタン。
        self.GripImg_frame = tk.Frame(settings_dialog)
        self.GripImg_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_Grip_Image = tk.Label(self.GripImg_frame, text="Gripper Action : " , width=20 ,anchor="e")
        self.lbl_Grip_Image.pack(side=tk.LEFT)
        
        self.btn_Grip_Image = tk.Button(self.GripImg_frame, text="Grip", width=10, command=self.grip)
        self.btn_Grip_Image.pack(side=tk.LEFT, padx=5)
        
        # 離すボタン。             
        self.btn_Release_Image = tk.Button(self.GripImg_frame, text="Release", width=10, command=self.release)
        self.btn_Release_Image.pack(side=tk.LEFT, padx=5)
        
    def toggle_detect(self):
        self.apply_detect = not self.apply_detect
        
    def auto_detect(self):
        self.dobotManager.auto_detect()
        return
    
    def pop_balloons(self):
        self.dobotManager.pop_balloons()
        return
        
    def move(self):
        self.dobotManager.move()
        return 
    
    def reset_alarm(self):
        self.dobotManager.reset_alarm()
        return 
    
    def grip(self):
        self.dobotManager.grip()
        return 
        
    def release(self):
        self.dobotManager.release()
        return 
    
    # 自動露光設定。
    def set_auto_exposure_on(self):
        if self.auto_exposure_var.get() == 0:
            self.auto_exposure_var.set(1)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self.btn_auto_exposure_on.config(relief=tk.SUNKEN)
            self.btn_auto_exposure_off.config(relief=tk.RAISED)

    def set_auto_exposure_off(self):
        if self.auto_exposure_var.get() == 1:
            self.auto_exposure_var.set(0)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.btn_auto_exposure_on.config(relief=tk.RAISED)
            self.btn_auto_exposure_off.config(relief=tk.SUNKEN)

    # 自動ホワイトバランス設定。
    def set_auto_wh_on(self):
        if self.auto_wh_var.get() == 0:
            self.auto_wh_var.set(1)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTO_WB, 1)
            self.btn_auto_wh_on.config(relief=tk.SUNKEN)
            self.btn_auto_wh_off.config(relief=tk.RAISED)
            
    def set_auto_wh_off(self):
        if self.auto_wh_var.get() == 1: 
            self.auto_wh_var.set(0)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTO_WB, 0)
            self.btn_auto_wh_on.config(relief=tk.RAISED)
            self.btn_auto_wh_off.config(relief=tk.SUNKEN)
            
    # 自動フォーカス設定。
    def set_auto_focus_on(self):
        if self.auto_focus_var.get() == 0:
            self.auto_focus_var.set(1)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTOFOCUS, 1) 
            self.btn_auto_focus_on.config(relief=tk.SUNKEN)
            self.btn_auto_focus_off.config(relief=tk.RAISED)
            
    def set_auto_focus_off(self):
        if self.auto_focus_var.get() == 1:
            self.auto_focus_var.set(0)
            self.camera_controller.vid.set(cv2.CAP_PROP_AUTOFOCUS, 0) 
            self.btn_auto_focus_on.config(relief=tk.RAISED)
            self.btn_auto_focus_off.config(relief=tk.SUNKEN)
            

