#-*- coding:utf-8 -*-
import cv2
import numpy as np


def filter2d(src, kernel):
    # カーネルサイズ
    m, n = kernel.shape

    # 畳み込み演算をしない領域の幅
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]

    # 出力画像用の配列（要素は全て0）
    dst = np.zeros((h, w))

    for y in range(d, h - d):
        for x in range(d, w - d):
            # 畳み込み演算
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)

    return dst


# 入力画像を読み込み
img = cv2.imread("tower.bmp")

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# カーネル（輪郭検出用）
kernel = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 1,  1]])

# 方法1
dst = filter2d(gray, kernel)

# 結果を出力
cv2.imwrite("2.bmp", dst)