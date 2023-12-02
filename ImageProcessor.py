import cv2
import numpy as np

class ImageProcessor:
    def __init__(self, GUIManager):
        self.GUIManager = GUIManager
        
    def display_feature_points(self, frame):
        
        if self.GUIManager.algorithm_combobox.get() == "SIFT":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sift = cv2.SIFT_create()
            kp = sift.detect(gray, None)
            frame = cv2.drawKeypoints(frame, kp, frame)  
        elif self.GUIManager.algorithm_combobox.get() == "Canny":
            # Canny
            frame = cv2.Canny(frame, 50, 150)       
        return frame
    
    def detect_object(self, frame):
        
        if self.GUIManager.algorithm_combobox.get() == "SIFT"  and self.GUIManager.reference_image_path:
            frame = self.detectBySIFT(frame, self.GUIManager.reference_image_path)
        elif self.GUIManager.algorithm_combobox.get() == "Canny"  and self.GUIManager.reference_image_path:
            frame = self.detectByCanny(frame, self.GUIManager.reference_image_path)
        elif self.GUIManager.algorithm_combobox.get() == "Cascade(Face)":
            frame = self.detectByCascadeFace(frame)
        elif self.GUIManager.algorithm_combobox.get() == "Cascade(PushPin)":
            #frame = self.detectByCascadePushPin(frame)
            frame = self.detectByCascadePushPinContours(frame)     
        elif self.GUIManager.algorithm_combobox.get() == "findContours":
            frame = self.detectByfindContours(frame)
        return frame
    
    # SIFTアルゴリズムによる物体検知。（TODO　Algo毎にクラス化）
    def detectBySIFT(self, frame, reference_image_path):
        
        # 比較対象画像読み込み(グレースケール)
        reference_image = cv2.imread(reference_image_path, 0)
        # SIFT検出器初期化
        sift = cv2.SIFT_create()
        # 比較対象画像のキーポイントを抽出。
        kp1, des1 = sift.detectAndCompute(reference_image, None)
        
        # カメラ映像フレームのグレースケール化。
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # カメラ映像フレームのキーポイントを抽出。
        kp2, des2 = sift.detectAndCompute(gray, None)
        
        # FLANN パラメータ設定。(近似最近傍点を検索するためのライブラリ)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        # 近似最近傍マッチング。
        matches = flann.knnMatch(des1,des2,k=2)
        
        # Loweの比率テストに基づいた特徴点マッチング。
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
        
        # 十分な特徴点が得られた場合、カメラ映像の対象物に矩形を描画。
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            
            h, w = reference_image.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            # 矩形描画。
            frame = cv2.polylines(frame, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
        return frame
    

    """
    比較画像との単純なテンプレートマッチング
    """
    def detectByCanny(self, frame, reference_image_path):
        # 比較対象画像読み込み(グレースケール)、Cannyによるエッジ検出。
        reference_image = cv2.imread(reference_image_path, cv2.IMREAD_GRAYSCALE)
        reference_edges = cv2.Canny(reference_image, 50, 150)

        # カメラフレーム画像読み込み(グレースケール)、Cannyによるエッジ検出。
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_edges = cv2.Canny(gray_frame, 50, 150)

        # 比較対象画像の読み込み失敗時は何もしない。
        if reference_edges.shape[0] >= frame_edges.shape[0] or reference_edges.shape[1] >= frame_edges.shape[1]:
            return frame

        # エッジ検出結果でテンプレートマッチング実行。
        result = cv2.matchTemplate(frame_edges, reference_edges, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # テンプレートマッチング結果で一致点が閾値以上あればカメラ映像の対象物に矩形を描画。
        if max_val > 0.4:
            top_left = max_loc
            bottom_right = (top_left[0] + reference_image.shape[1], top_left[1] + reference_image.shape[0])
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

        return frame
    
    """
    Cascade分類器(顔認識)を使用した検出
    """
    def detectByCascadeFace(self, frame):
        # カメラフレーム画像読み込み(グレースケール)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       
        # 学習済み顔認識モデルの読み込み
        cascade = cv2.CascadeClassifier("./data/cascade/haarcascade_frontalface_default.xml")
        # 顔を検出する
        lists = cascade.detectMultiScale(gray_frame, minSize=(100, 100))
        if len(lists):
            # 顔を検出した場合、forですべての顔を赤い長方形で囲む
            for (x,y,w,h) in lists:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
        return frame
    
    """
    Cascade分類器(画鋲)を使用した検出
    """
    def detectByCascadePushPin(self, frame):
        
        # カメラフレーム画像読み込み(グレースケール)
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        # 前処理
        preProcessed_frame = self.preProc(frame)
           
        # 学習済み画鋲認識モデルの読み込み
        cascade = cv2.CascadeClassifier("./data/cascade/cascade_pushpin.xml")
        # 画鋲を検出する
        lists = cascade.detectMultiScale(preProcessed_frame, 
                                         scaleFactor=float(self.GUIManager.sfEntry.get()), 
                                         minNeighbors=int(self.GUIManager.mnEntry.get()),
                                         minSize=(50, 50))
        if len(lists):
            # 画鋲を検出した場合、forですべての画鋲を緑色の長方形で囲む
            for (x,y,w,h) in lists:
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)
        return frame       
    
            

    """
    Cascade分類器(画鋲)を使用した検出
    """
    def detectByCascadePushPinContours(self, frame):
        
        # 輪郭を検出して領域を取得
        regions = self.getContoursArea(frame)
    
        # 学習済み画鋲認識モデルの読み込み
        cascade = cv2.CascadeClassifier("./data/cascade/cascade_pushpin.xml")
               
        # 指定された各領域に対してCascade分類器を適用
        for (x, y, w, h) in regions:
            # 指定された領域のみを対象に検出を実行
            region_gray = frame[y:y+h, x:x+w]
            lists = cascade.detectMultiScale(region_gray, 
                                             scaleFactor=float(self.GUIManager.sfEntry.get()), 
                                             minNeighbors=int(self.GUIManager.mnEntry.get()))

            # 最大の矩形を見つける
            max_area = 0
            max_rect = None
            for (rx, ry, rw, rh) in lists:
                area = rw * rh
                if area > max_area:
                    max_area = area
                    max_rect = (rx, ry, rw, rh)

            # 検出された物体のうち最大の物体のみを緑色の長方形で囲む
            if max_rect is not None:
                rx, ry, rw, rh = max_rect
                cv2.rectangle(frame, (x + rx, y + ry), (x + rx + rw, y + ry + rh), (0, 255, 0), thickness=2)
            
            # 検出された物体を緑色の長方形で囲む
            #for (rx, ry, rw, rh) in lists:
            #    cv2.rectangle(frame, (x + rx, y + ry), (x + rx + rw, y + ry + rh), (0, 255, 0), thickness=2)

        return frame  
    
    """
    findContoursを使用した輪郭検出
    """
    def detectByfindContours(self, frame):
        # カメラフレーム画像読み込み(グレースケール)
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        # 前処理
        preProcessed_frame = self.preProc(frame)
        
        # 二値化
        _, threshold = cv2.threshold(preProcessed_frame, self.GUIManager.threshold_scale.get(), 255, cv2.THRESH_BINARY)

        # 輪郭の検出
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 輪郭に緑色の枠を描画
        for contour in contours:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
        return frame
    
    
    
    def getContoursArea(self, frame):
        # カメラフレーム画像読み込み(グレースケール)
        #gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 前処理
        preProcessed_frame = self.preProc(frame)

        # 二値化
        _, threshold = cv2.threshold(preProcessed_frame, self.GUIManager.threshold_scale.get(), 255, cv2.THRESH_BINARY)

        # 輪郭の検出
        _, contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 拡張するピクセル数
        expansion_pixels = 100  # 周辺100ピクセルに設定

        # 検出した輪郭の外接矩形を格納するリスト
        bounding_rects = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:
                
                # 外接矩形の取得
                x, y, w, h = cv2.boundingRect(contour)

                # 矩形の座標を調整して拡張
                x_expanded = max(0, x - expansion_pixels)
                y_expanded = max(0, y - expansion_pixels)
                w_expanded = w + 2 * expansion_pixels
                h_expanded = h + 2 * expansion_pixels

                # 拡張した矩形をリストに追加
                bounding_rects.append((x_expanded, y_expanded, w_expanded, h_expanded))
                
        print(len(bounding_rects))
        
        return bounding_rects

    """
    検知前処理
    """
    def preProc(self, frame):
        # カメラフレーム画像読み込み(グレースケール)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
        
        #  ヒストグラム均一化（Histogram Equalization）
        Equalized_frame = cv2.equalizeHist(gray_frame)
        cv2.imshow('Equalized Image', Equalized_frame)

        # 適応的ヒストグラム均一化（Adaptive Histogram Equalization）
        # CLAHEオブジェクトを作成
        clahe = cv2.createCLAHE()

        # CLAHEを適用
        clahe_frame = clahe.apply(gray_frame)
        cv2.imshow('Adaptive Equalized Image', clahe_frame)

        return Equalized_frame
    