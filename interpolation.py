import numpy as np


def bilinear_interpolation(img, source_coords):
    img = img.transpose(2, 0, 1)
    h = img.shape[1]
    w = img.shape[2]

    px = source_coords[:1]
    py = source_coords[1]

    # # 归一化使坐标属于[-1,1}
    # px = px / (w - 1) * 2.0 - 1.0
    # py = py / (h - 1) * 2.0 - 1.0

    x = px.reshape(-1)
    y = py.reshape(-1)

    max_x = int(w - 1)
    max_y = int(h - 1)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, max_x)
    x1 = np.clip(x1, 0, max_x)
    y0 = np.clip(y0, 0, max_y)
    y1 = np.clip(y1, 0, max_y)

    # 插值公式: x 在a,b,c,d之间点，
    # f(x) = (x2-x)(y2-y)f(a) + (x2-x)(y-y1)f(b) + (x-x1)(y2-y)f(c) + (x-x1)(y-y1)f(d)

    # 先得到abcd的坐标 (python3 没有long类型)
    base = w * h
    base = np.zeros(base)
    base = base.astype(int)
    base_y0 = base + y0.astype(int) * w
    base_y1 = base + y1.astype(int) * w
    idx_a = base_y0 + x0.astype(int)
    idx_b = base_y1 + x0.astype(int)
    idx_c = base_y0 + x1.astype(int)
    idx_d = base_y1 + x1.astype(int)

    # import matplotlib.pyplot as plt
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(x0.reshape(h, w)/max(x0))
    # plt.subplot(122)
    # plt.imshow(y0.reshape(h, w)/max(y0))
    # plt.show()

    # import cv2
    # cv2.imshow('a', x0.reshape(h, w)/max(x0))
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    img = img.transpose(1, 2, 0)
    im_flat = img.reshape(-1, img.shape[2]).astype(float)
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = (x1.astype(float) - x) * (y1.astype(float) - y)
    wb = (x1.astype(float) - x) * (1.0 - (y1.astype(float) - y))
    wc = (1.0 - (x1.astype(float) - x)) * (y1.astype(float) - y)
    wd = (1.0 - (x1.astype(float) - x)) * (1.0 - (y1.astype(float) - y))
    wa, wb, wc, wd = wa[:, np.newaxis], wb[:,
                                           np.newaxis], wc[:, np.newaxis], wd[:, np.newaxis]
    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(h, w, img.shape[2])

    # import matplotlib.pyplot as plt
    # plt.imshow(img.transpose(1,2,0))
    # im_flat.reshape(h, w, 3).astype(int) == img.reshape
    # plt.imshow(im_flat.reshape(h,w,3).astype(int))
    # plt.show()

    return output
