#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 21:00
# @Author  : xi_fen
# @Site    : 
# @File    : getH.py
# @Software: PyCharm
import cv2 as cv
import numpy as np
import os
img_srcpath = r'C:\Users\33947\Desktop\S\S005\5\1.jpg'
#img_srcpath = r'E:\AIC23_Track1_MTMC_Tracking_2\data\test\S003\c014\img1\00011001.jpg'
# img_srcpath = 'output.png'
img_destpath =r'C:\Users\33947\Desktop\S\S005\map.jpg'#鸟瞰图


if __name__ == '__main__':

    image_base, _ = os.path.split(img_srcpath)#分割路径和文件名
    image_base, _ = os.path.split(image_base)
    image_base, imgPrefix = os.path.split(image_base)
    _, sPrefix = os.path.split(image_base)

    txtPath = f"{imgPrefix}.txt"
    txtPath = os.path.join(f"./{sPrefix}", txtPath)

    savePath = os.path.join(f"./{sPrefix}", f"{imgPrefix}-H1.txt")
    savePath2 = os.path.join(f"./{sPrefix}", f"{imgPrefix}-H2.txt")

    if not os.path.exists(txtPath):
        print("对应点文件不存在！")


    points = np.loadtxt(txtPath, dtype=np.int32, delimiter=' ')#读取点文件

    pts_src1 = []
    pts_dst1 = []
    for i in points:
        pts_src1.append([i[1], i[2]])
        pts_dst1.append([i[3], i[4]])

    # Calculate Homography
    # 计算Homography矩阵
    img_src_coordinate = np.array(pts_src1)
    replace_coordinate = np.array(pts_dst1)


    #该函数需要输入两个参数：源图像中目标的坐标和目标图像中目标的坐标，然后输出两张图片之间的单应性矩阵和一个掩码（mask）。
    # 其中，单应性矩阵是一个3x3的矩阵，用于将源图像中的目标映射到目标图像中的目标。
    #具体来说，代码中的img_src_coordinate表示源图像中目标的坐标，
    # replace_coordinate表示目标图像中目标的坐标。
    # method=cv.RANSAC表示使用RANSAC算法进行单应性矩阵的计算，该算法可以通过排除噪声和异常值来提高单应性矩阵的精度。
    #代码中先通过findHomography()函数计算了源图像到目标图像的单应性矩阵matrix，
    # 然后通过np.linalg.inv()函数计算了matrix的逆矩阵matrix_inv。
    # 接着，代码又通过findHomography()函数计算了目标图像到源图像的单应性矩阵matrix2。
    # 最后，代码分别输出了matrix、matrix_inv和matrix2的值
    matrix, mask = cv.findHomography(img_src_coordinate, replace_coordinate, method=cv.RANSAC )
    print(f'matrix: {matrix}')

    print(f"matrix_inv:{np.linalg.inv(matrix)}")

    matrix2, mask = cv.findHomography(replace_coordinate,img_src_coordinate, method=cv.RANSAC )
    print(f'matrix2: {matrix2}')

    # np.savetxt('c014.txt', matrix)
    np.savetxt(savePath, matrix, fmt="%.8f", delimiter=" ")
    np.savetxt(savePath2, matrix2, fmt="%.8f", delimiter=" ")

    x1 = np.ones((1, img_src_coordinate.shape[0]))
    x2 = np.vstack((img_src_coordinate.T,x1))
    y = np.dot(matrix, x2)
    y2 = y / y[2]

    print(f'dst: {y2.T}')

    img_src = cv.imread(img_srcpath)
    img_dest = cv.imread(img_destpath)
    perspective_img = cv.warpPerspective(img_src, matrix, (img_dest.shape[1], img_dest.shape[0]))
    height4, width4 = perspective_img.shape[:2]
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.resizeWindow('img', width4, height4)
    cv.imshow('img', perspective_img)  # 显示图像数据


    # 降噪，去掉最大或最小的像素点
    retval, threshold_img = cv.threshold(perspective_img, 0, 255, cv.THRESH_BINARY)
    # # 将降噪后的图像与之前的图像进行拼接
    cv.copyTo(src=threshold_img, mask=np.tile(threshold_img, 1), dst=img_dest)
    cv.copyTo(src=perspective_img, mask=np.tile(perspective_img, 1), dst=img_dest)
    height3, width3 = img_dest.shape[:2]
    cv.namedWindow('result', cv.WINDOW_NORMAL)
    cv.resizeWindow('result', width3, height3)
    cv.imshow('result', img_dest)  # 显示图像数据
    cv.waitKey()
    cv.destroyAllWindows()