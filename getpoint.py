
import cv2
import numpy as np
import os
from tkinter import messagebox
img1_path = r'C:\Users\33947\Desktop\S\S001\1\1.jpg'
#img1_path = r'E:\AIC23_Track1_MTMC_Tracking_2\data\test\S003\c014\img1\00010001.jpg'
img2_path = r'C:\Users\33947\Desktop\S\S001\map.jpg'
# img2_path = r'E:\AIC23_Track1_MTMC_Tracking_2\data\test\S003\00000001.jpg'


def save_point():
    image_base, _ = os.path.split(img1_path)
    image_base, _ = os.path.split(image_base)
    image_base, imgPrefix = os.path.split(image_base)
    _, sPrefix = os.path.split(image_base)

    txtPath = f"temp.txt"

    if not os.path.exists(f"./{sPrefix}"):
        os.makedirs(f"./{sPrefix}")

    txtPath = os.path.join(f"./{sPrefix}", txtPath)
    dout = open(txtPath, 'w')

    if len(img1_pts) > len(img2_pts) and len(img1_pts) > 0:
        for i, pts in enumerate(img2_pts):
            dout.write('{} {} {} {} {}\n'.format(i + 1, img1_pts[i][0], img1_pts[i][1], pts[0], pts[1]))
    else:
        for i, pts in enumerate(img1_pts):
            dout.write('{} {} {} {} {}\n'.format(i + 1, pts[0], pts[1], img2_pts[i][0], img2_pts[i][1]))

    dout.close()


def saveTure():#保存目标在两张图片的ID和坐标信息
    image_base, _ = os.path.split(img1_path)
    image_base, _ = os.path.split(image_base)
    image_base, imgPrefix = os.path.split(image_base)
    _, sPrefix = os.path.split(image_base)

    txtPath = f"{imgPrefix}.txt"

    if not os.path.exists(f"./{sPrefix}"):
        os.makedirs(f"./{sPrefix}")

    txtPath = os.path.join(f"./{sPrefix}", txtPath)
    dout = open(txtPath, 'w')

    if len(img1_pts) > len(img2_pts) and len(img1_pts) > 0:
        for i, pts in enumerate(img2_pts):
            dout.write('{} {} {} {} {}\n'.format(i + 1, img1_pts[i][0], img1_pts[i][1], pts[0], pts[1 ]))
    else:
        for i, pts in enumerate(img1_pts):
            dout.write('{} {} {} {} {}\n'.format(i + 1, pts[0], pts[1], img2_pts[i][0], img2_pts[i][1]))

    dout.close()


def draw_circles(img, pts, param, ):
    id = 0
    for pt in pts:
        id += 1
        (x, y) = pt
        cv2.putText(img, str(id), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 69, 186), 1)  # 绘制ID号
        cv2.circle(img, pt, 3, (0, 0, 255), -1)#绘制圆圈，图片，圆心，半径，颜色，填充

    image_base, _ = os.path.split(img1_path)
    image_base, _ = os.path.split(image_base)
    image_base, imgPrefix = os.path.split(image_base)

    if param == 'img1':
        cv2.imshow(f'{imgPrefix}', img)
    if param == 'img2':
        cv2.imshow('map', img)


def draw_anchor(img, pts, param):
    id = 0
    for pt in pts:
        id += 1
        (x, y) = pt
        cv2.putText(img, str(id), (x - 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (224, 69, 186), 1)  # 绘制ID号
        cv2.circle(img, pt, 3, (255, 0, 0), -1)  # 绘制蓝色圆点

    if param == 'img1':
        cv2.imshow(f'{imgPrefix}', img)
    if param == 'img2':
        cv2.imshow('map', img)


# 定义鼠标回调函数
def draw_circle(event, x, y, flags, param):
    global img1, img2, img1_pts, img2_pts, prev_img1_pts, prev_img2_pts

    if event == cv2.EVENT_LBUTTONDOWN and flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_LBUTTONDOWN:  # ctrl+左键
        print("ctrl+左键")
        if param == 'img1':
            # 在图片1上标注点，并绘制圆圈
            img1_pts.append((x, y))
            imgs = img1.copy()
            draw_circles(imgs, img1_pts, param)
        elif param == 'img2':
            # 在图片2上标注点，并绘制圆圈
            img2_pts.append((x, y))
            imgs = img2.copy()
            draw_circles(imgs, img2_pts, param)

        save_point()


    if event == cv2.EVENT_RBUTTONDOWN and flags == cv2.EVENT_FLAG_CTRLKEY + cv2.EVENT_RBUTTONDOWN:  # ctrl+右键
        print("ctrl+右键")
    if event == cv2.EVENT_MBUTTONUP and flags == cv2.EVENT_FLAG_CTRLKEY:  # ctrl+中键
        print("ctrl+中键")


    if event == cv2.EVENT_LBUTTONUP and flags != cv2.EVENT_FLAG_CTRLKEY :
        if param == 'img1':
            # 在图片1上标注点，并绘制圆圈
            img1_pts.append((x, y))
            imgs = img1.copy()
            draw_circles(imgs, img1_pts, param)


        elif param == 'img2':
            # 在图片2上标注点，并绘制圆圈
            img2_pts.append((x, y))
            imgs = img2.copy()
            draw_circles(imgs, img2_pts, param)

        save_point()

    # 如果按下鼠标右键，则返回上一个状态
    if event == cv2.EVENT_RBUTTONUP and flags != cv2.EVENT_FLAG_CTRLKEY :
        if param == 'img1':
            if len(img1_pts) > 0:
                img1_pts.pop()
                imgs = img1.copy()
                draw_circles(imgs, img1_pts, param)

        elif param == 'img2':
            if len(img2_pts) > 0:
                img2_pts.pop()
                imgs = img2.copy()
                draw_circles(imgs, img2_pts, param)

        save_point()

    if event == cv2.EVENT_MBUTTONUP and flags != cv2.EVENT_FLAG_CTRLKEY:  # 中键按下事件

        if param == 'img1':
            # 在图片1上锚点，并绘制圆圈
            imgs = img1.copy()
            draw_circles(imgs, img1_pts, param)
            cv2.circle(imgs, (x, y), 4, (255, 0, 0), -1)  # 绘制蓝色圆点
            cv2.imshow(f'{imgPrefix}', imgs)
            if len(img1_pts) > 4 and len(img2_pts) > 4:
                _, point = clc_H(img1_pts, img2_pts, [x, y])
                imgs = img2.copy()
                draw_circles(imgs, img2_pts, "img2")
                cv2.circle(imgs, point, 4, (255, 0, 0), -1)  # 绘制蓝色圆点
                cv2.imshow('map', imgs)

        elif param == 'img2':
            # 在图片2上锚点，并绘制圆圈
            imgs = img2.copy()
            draw_circles(imgs, img2_pts, param)
            cv2.circle(imgs, (x, y), 4, (255, 0, 0), -1)  # 绘制蓝色圆点
            cv2.imshow('map', imgs)
            if len(img1_pts) > 4 and len(img2_pts) > 4:
                _, point = clc_H(img2_pts,img1_pts, [x, y])
                imgs = img1.copy()
                draw_circles(imgs, img1_pts, "img1")
                cv2.circle(imgs, point, 4, (255, 0, 0), -1)  # 绘制蓝色圆点
                cv2.imshow(f'{imgPrefix}', imgs)



def clc_H(pts1, pts2, point):
    pts_src = []
    pts_dst = []
    for i in range(min(len(pts1), len(pts2))):
        pts_src.append([pts1[i][0], pts1[i][1]])
        pts_dst.append([pts2[i][0], pts2[i][1]])

    pts_src = np.array(pts_src)
    pts_dst = np.array(pts_dst)

    matrix, _ = cv2.findHomography(pts_src, pts_dst, method=cv2.RANSAC)

    point = np.array(point)

    x2 = np.array([point[0], point[1], 1])
    y = np.dot(matrix, x2)
    y2 = y / y[2]

    print(f'dst: {y2.T}')

    return matrix, (int(y2[0]), int(y2[1]))


def confirm_overwrite():
    user_input = messagebox.askyesno("提示", "是否保存修改?")
    if user_input:
        saveTure()
        print("文件已经保存.")
    else:
        # 如果用户点击“No”，则取消操作
        print("文件放在暂存temp.txt中,还没有保存.")


img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)#读取图像数据

# 显示图片，并等待用户操作

image_base, _ = os.path.split(img1_path)
image_base, _ = os.path.split(image_base)
image_base, imgPrefix = os.path.split(image_base)
_, sPrefix = os.path.split(image_base)


height, width = img1.shape[:2]
cv2.namedWindow(f'{imgPrefix}', cv2.WINDOW_NORMAL)
cv2.resizeWindow(f'{imgPrefix}', width, height)
cv2.imshow(f'{imgPrefix}', img1)

height2, width2 = img2.shape[:2]
cv2.namedWindow('map', cv2.WINDOW_NORMAL)
cv2.resizeWindow('map',width2,height2)
cv2.imshow('map', img2)#显示图像数据
# 初始化标注点列表
img1_pts = []
img2_pts = []
txtPath = f"{imgPrefix}.txt"
txtPath = os.path.join(f"./{sPrefix}", txtPath)#路径合并
if os.path.exists(txtPath) and os.path.getsize(txtPath):
    points = np.loadtxt(txtPath, dtype=np.int32, delimiter=' ')
    for i in points:
        img1_pts.append([i[1], i[2]])
        img2_pts.append([i[3], i[4]])

    imgs = img1.copy()
    draw_circles(imgs, img1_pts, "img1")
    imgs = img2.copy()
    draw_circles(imgs, img2_pts, "img2")

# 保存锚点列表
anchor_img1_pts = []
anchor_img2_pts = []

# 设置鼠标回调函数
cv2.setMouseCallback(f'{imgPrefix}', draw_circle, 'img1')
cv2.setMouseCallback('map', draw_circle, 'img2')


cv2.waitKey(0)
cv2.destroyAllWindows()

confirm_overwrite()

