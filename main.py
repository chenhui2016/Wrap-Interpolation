import matplotlib.pyplot as plt
from dataload import read_img_cam
from utils import grid_abs
import numpy as np
import cv2
from transformtion import transform_mat
from interpolation import bilinear_interpolation

ref_path = '/home/ch/gm_d/datasets/coco_dtu_dtu_tr/dtu_training/dtu_training/mvs_training/dtu/Rectified/scan29_train/rect_031_0_r5000.png'

source_path = '/home/ch/gm_d/datasets/coco_dtu_dtu_tr/dtu_training/dtu_training/mvs_training/dtu/Rectified/scan29_train/rect_033_0_r5000.png'

# [512,640,3], [3,3], [4,4], [128,160]
r_im, r_k, r_e, r_d = read_img_cam(ref_path)
r_d = cv2.resize(r_d, (640, 512))
# [512,640,3], [3,3], [4,4], [128,160]
s_im, s_k, s_e, s_d = read_img_cam(source_path)

# 定义网格
h, w = r_im.shape[:2]
grid = grid_abs(h, w)  # 3维的齐次坐标[3, h*w] z轴全为1, 因此其是像素坐标(还不到图像坐标，因为内参包含cx，cy)

# 计算source像su坐标
source_coords = transform_mat(
    r_im, r_k, r_e, r_d, s_im, s_k, s_e, s_d, grid, h, w)

# 得到source像su坐标，使用插值的方法，将原图图像的像素按照坐标放到新的坐标中
# 使用双线性插值将原图像素放到新的坐标中

wraped_source = bilinear_interpolation(s_im, source_coords).astype(int)

# 显示
plt.figure()
plt.subplot(221)
plt.imshow(r_im)
plt.subplot(222)
plt.imshow(s_im)
plt.subplot(223)
plt.imshow(wraped_source)
plt.show()
