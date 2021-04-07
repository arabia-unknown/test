#-*- coding:utf-8 -*-
import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import random

def expand(image, ratio):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src * ratio
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (2*w, 2*h), cv2.INTER_LANCZOS4)

def random_shift(image, shifts):
    h, w = image.shape[:2]
    src = np.array([[0.0, 0.0],[0.0, 1.0],[1.0, 0.0]], np.float32)
    dest = src + shifts.reshape(1,-1).astype(np.float32)
    affine = cv2.getAffineTransform(src, dest)
    return cv2.warpAffine(image, affine, (w, h))

def rotate_center(image, angle):
    h, w = image.shape[:2]
    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    return cv2.warpAffine(image, affine, (w, h))

def rotate_fit(image, angle):
    h, w = image.shape[:2]
    # 回転後のサイズ
    radian = np.radians(angle)
    sine = np.abs(np.sin(radian))
    cosine = np.abs(np.cos(radian))
    tri_mat = np.array([[cosine, sine],[sine, cosine]], np.float32)
    old_size = np.array([w,h], np.float32)
    new_size = np.ravel(np.dot(tri_mat, old_size.reshape(-1,1)))
    # 回転アフィン
    affine = cv2.getRotationMatrix2D((w/2.0, h/2.0), angle, 1.0)
    # 平行移動
    affine[:2,2] += (new_size-old_size)/2.0
    # リサイズ
    affine[:2,:] *= (old_size / new_size).reshape(-1,1)
    return cv2.warpAffine(image, affine, (w, h))

if __name__ == "__main__":
    print("Insert picture name(.bmp)")
    str = input()
    fig = plt.figure()

    indx1 = random.uniform(0, 2)
    indx2 = random.uniform(2, 3)
    ind = []
    #rand = (random.randint(-100, 100),random.randint(-100, 100))
    rand = (-50,-50)
    ind.append(rand)
    indx3 = np.array(ind)
    #indx4 = random.randrange(0, 360, 6)
    indx4 = -45


    image = cv2.imread(str)[:,:,::-1]
    converted1 = expand(image,indx1)
    converted2 = expand(image,indx2)
    converted3 = random_shift(image,indx3)
    converted4 = rotate_center(image,indx4)
    converted5 = rotate_fit(image,indx4)
    
    axes=[]
    rows = 2
    cols = 3
    axes.append( fig.add_subplot(rows, cols, 1) )
    subplot_title=("default")
    axes[-1].set_title(subplot_title) 
    plt.imshow(image)
    axes.append( fig.add_subplot(rows, cols, 2) )
    subplot_title=("shrink")
    axes[-1].set_title(subplot_title) 
    plt.imshow(converted1)
    axes.append( fig.add_subplot(rows, cols, 3) )
    subplot_title=("expand")
    axes[-1].set_title(subplot_title) 
    plt.imshow(converted2)
    axes.append( fig.add_subplot(rows, cols, 4) )
    subplot_title=("shift")
    axes[-1].set_title(subplot_title) 
    plt.imshow(converted3)
    axes.append( fig.add_subplot(rows, cols, 5) )
    subplot_title=("rotate1")
    axes[-1].set_title(subplot_title) 
    plt.imshow(converted4)
    axes.append( fig.add_subplot(rows, cols, 6) )
    subplot_title=("rotate2")
    axes[-1].set_title(subplot_title) 
    plt.imshow(converted5)
    plt.show()