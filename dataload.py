import cv2
import os
import numpy as np
import re


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_img_cam(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sense_id = int(path.split('/')[-2].split('_')[0][-1:])
    idx = int(path.split('/')[-1].split('.')[0].split('_')[1])
    filename = os.path.join(path.split('mvs_training')[
                            0], 'mvs_training/dtu/Cameras/train/{:0>8}_cam.txt'.format(idx))
    with open(filename, 'r') as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]

    # read intrinsic
    intrinsic = np.fromstring(
        ' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # read extrinsic
    extrinsic = np.fromstring(
        ' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))

    depth_path = os.path.join(path.split('mvs_training')[0],
                              'mvs_training/dtu/Depths/scan{}_train/depth_map_{:0>4}.pfm'.format(sense_id, idx))
    # read depth
    depth = np.array(read_pfm(depth_path)[0], dtype=np.float32)
    # import matplotlib.pyplot as plt
    # plt.imshow(depth)
    # plt.show()
    return img, intrinsic, extrinsic, depth
