import numpy as np

def transform_mat(r_im, r_k, r_e, r_d, s_im, s_k, s_e, s_d, grid, h, w):
    # 将像素坐标转为相机坐标，进而利用R和t完成点对应关系，P2=R2R1^TP1+(T2-R2R1^TT1) ， R`=R2R1^T, T`=T2-R2R1^TT1
    # 像素坐标转相机坐标 Dpx = KPc，  Pc=K^-1px*1*D
    # 这里只是将source转为ref，因此需要通过ref得到source 图像坐标， 也就是source pixel frame

    ## 以下得到ref像素坐标系到source像素坐标系
    # ref像素坐标系得到ref相机坐标系, Pc=K^-1px*D
    ref_cam_grid = np.linalg.inv(r_k) @ grid * r_d.reshape(1, -1)  # [3, h*w]

    # ref相机坐标系得到source相机坐标系
    # 需要得知两个相机的变换矩阵，R`=R2R1^T, T`=T2-R2R1^TT1  （R为正交矩阵因此转置=逆）

    ref_R = r_e[:3, :3]
    ref_T = r_e[:3, 3]

    source_R = s_e[:3, :3]
    source_T = s_e[:3, 3]

    R = source_R @ ref_R.transpose(1, 0)
    T = source_T - source_R @ ref_R.transpose(1, 0) @ ref_T
    last_colum = np.array([[0, 0, 0, 1]])

    transorm_matrix = np.concatenate([np.concatenate([R, T.reshape(-1, 1)], axis=1), last_colum], axis=0)

    # source相机坐标系到source图像坐标系， px_s=K_sPC_s=K_s@[R|t]@PC_t=K_s@[R|t]@K_t^-1@px_t*D_t
    # 投影矩阵
    hom = np.array([[0,0,0,1]])
    proj = np.concatenate((np.concatenate((s_k, np.zeros((3, 1))), axis=1), hom), axis=0) @ transorm_matrix

    # 将grid带入，得到source pixel grid
    s_grid = proj @ np.concatenate([ref_cam_grid, np.ones((1, h * w))], axis=0)  # [3, h*w]
    x = s_grid[0:1, :]
    y = s_grid[1:2, :]
    z = s_grid[2:3, :]
    x_pixel = x / (z + 1e-10)
    y_pixel = y / (z + 1e-10)
    source_coords = np.concatenate([x_pixel, y_pixel], axis=0)
    source_coords = source_coords.reshape(2, w, h)

    return source_coords
