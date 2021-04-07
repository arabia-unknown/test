#-*- coding:utf-8 -*-
import cv2
import numpy as np


def filter2d(src, kernel):
    # �J�[�l���T�C�Y
    m, n = kernel.shape

    # ��ݍ��݉��Z�����Ȃ��̈�̕�
    d = int((m-1)/2)
    h, w = src.shape[0], src.shape[1]

    # �o�͉摜�p�̔z��i�v�f�͑S��0�j
    dst = np.zeros((h, w))

    for y in range(d, h - d):
        for x in range(d, w - d):
            # ��ݍ��݉��Z
            dst[y][x] = np.sum(src[y-d:y+d+1, x-d:x+d+1]*kernel)

    return dst


# ���͉摜��ǂݍ���
img = cv2.imread("tower.bmp")

# �O���[�X�P�[���ϊ�
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# �J�[�l���i�֊s���o�p�j
kernel = np.array([[1, 1,  1],
                   [1, -8, 1],
                   [1, 1,  1]])

# ���@1
dst = filter2d(gray, kernel)

# ���ʂ��o��
cv2.imwrite("2.bmp", dst)