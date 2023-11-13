import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

class GUIManager:
    def __init__(self, window, controller):
        self.window = window
        self.camera_controller = controller
        self.apply_detect = False
        self.apply_DispFeature = False
        self.reference_image_path = None
        self.current_image = None
        self.auto_exposure_var = tk.IntVar(value=1)
        self.auto_wh_var = tk.IntVar(value=1) 
        self.auto_focus_var = tk.IntVar(value=1)
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
        
        self.btn_snapshot = tk.Button(self.window, text="Disp. Feature Point", width=15, command=self.toggle_DispFeature)
        self.btn_snapshot.pack(pady=20, side=tk.RIGHT, padx=5)
        
        self.btn_snapshot = tk.Button(self.window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_snapshot.pack(pady=20, side=tk.RIGHT, padx=5)
    
    # def updateCanvas(self):
    
    def open_camera(self):
        self.camera_controller.connect()

    def close_camera(self):
        self.camera_controller.disconnect()
        self.canvas.delete(self.current_image)

    def toggle_detect(self):
        self.apply_detect = not self.apply_detect
        
    def toggle_DispFeature(self):
        self.apply_DispFeature = not self.apply_DispFeature

    def snapshot(self):
        self.camera_controller.snapshot()


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
        
        # 検出アルゴリズム。
        self.algo_frame = tk.Frame(settings_dialog)
        self.algo_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_algorithm = tk.Label(self.algo_frame, text="Detection Algorithm : ", width=20 ,anchor="e")
        self.lbl_algorithm.pack(side=tk.LEFT)
        self.algorithm_combobox = ttk.Combobox(self.algo_frame, values=["SIFT", "Canny", "Cascade(Face)", "Cascade(PushPin)"])
        self.algorithm_combobox.set("SIFT")  # 初期値
        self.algorithm_combobox.pack(side=tk.LEFT, padx=5)
        
        # 検出対象画像入力。
        self.SelectImg_frame = tk.Frame(settings_dialog)
        self.SelectImg_frame.pack(pady=10, anchor=tk.W)
        
        self.lbl_select_Image = tk.Label(self.SelectImg_frame, text="" , width=20 ,anchor="e")
        self.lbl_select_Image.pack(side=tk.LEFT)
        
        self.btn_select_Image = tk.Button(self.SelectImg_frame, text="Select Image", width=10, command=self.select_image)
        self.btn_select_Image.pack(side=tk.LEFT, padx=5)
        

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
            
    def select_image(self):
        self.reference_image_path = filedialog.askopenfilename()
