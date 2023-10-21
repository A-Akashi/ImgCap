import cv2
import numpy as np

class ImageProcessor:
    # def __init__(self):
        
    def display_feature_points(self, frame):
        
        # TODO 引数のAlgoで検知方式を呼び分ける。
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp = sift.detect(gray, None)
        frame = cv2.drawKeypoints(frame, kp, frame)  
        return frame
    
    def detect_object(self, frame, reference_image_path):
        
        # TODO 引数のAlgoで検知方式を呼び分ける。
        return self.detectBySIFT(frame, reference_image_path)
    
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
            if m.distance < 0.7 * n.distance:
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
        
