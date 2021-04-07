#-*- coding:utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt





print("Insert picture name(.bmp)")
str = input()
# 入力画像を読み込み
img = cv2.imread(str)[:,:,::-1]

# グレースケール変換
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# カーネル（輪郭検出用）
kernel4 = np.array([[0, 1,  0],
                   [1, -4, 1],
                   [0, 1,  0]], np.float32)

kernel8 = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 1,  1]], np.float32)

dst4 = cv2.filter2D(gray, cv2.CV_64F, kernel4)
dst8 = cv2.filter2D(gray, cv2.CV_64F, kernel8)


fig = plt.figure()
axes=[]
rows = 1
cols = 3
# 結果を出力
axes.append( fig.add_subplot(rows, cols, 1) )
subplot_title=("default")
axes[-1].set_title(subplot_title) 
plt.imshow(img)
axes.append( fig.add_subplot(rows, cols, 2) )
subplot_title=("4kinbou")
axes[-1].set_title(subplot_title) 
plt.imshow(dst4, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
axes.append( fig.add_subplot(rows, cols, 3) )
subplot_title=("8kinbou")
axes[-1].set_title(subplot_title) 
plt.imshow(dst8, cmap='gray', vmin = 0, vmax = 255,interpolation='none')
plt.show()
