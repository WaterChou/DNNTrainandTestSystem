import pickle
import numpy as np
import random
import matplotlib.pyplot as plt
import math
import scipy.misc as scimg_misc
import scipy.ndimage as scimg_nd
import skimage.exposure as skimg_exp
import skimage.transform as skimg_trans
import cv2
import colorsys
import math
from random import shuffle
from PIL import Image, ImageOps, ImageEnhance
# matplotlib.use('Agg')           # noqa: E402



"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. 
There are 50000 training images and 10000 test images.

CIFAR-100 is just like the CIFAR-10, except it has 100 classes containing 600 images each.
There are 500 training images and 100 testing images per class.
The 100 classes in the CIFAR-100 are grouped into 20 superclasses.
Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
"""

cifar10_label_name_path = file = './CIFAR-10/batches.meta'
cifar10_test_data_path = './CIFAR-10/test_batch'

cifar10_train_data_path = ['./CIFAR-10/data_batch_1',
                           './CIFAR-10/data_batch_2',
                           './CIFAR-10/data_batch_3',
                           './CIFAR-10/data_batch_4',
                           './CIFAR-10/data_batch_5']

cifar100_label_name = './CIFAR-100/meta'
cifar100_test_data_path = './CIFAR-100/test'
cifar100_train_data_path = './CIFAR-100/train'


def unpickle(file_path):

    with open(file_path, 'rb') as fo:
        # data_dict = pickle.load(fo, encoding='bytes')  # official
        data_dict = pickle.load(fo, encoding='latin1')

    return data_dict


def read_cifar10_label_name(file_path):

    data_dict = unpickle(file_path)
    # dict_keys(['num_vis', 'label_names', 'num_cases_per_batch'])

    label_name = data_dict['label_names']  # len=10

    return label_name


def read_cifar10_train(train_path):

    for i in range(5):
        data_dict = unpickle(train_path[i])
        # dict_keys(['filenames', 'batch_label', 'data', 'labels'])
        x_data = data_dict['data']   # shape(10000, 3072)
        y_data = np.array(data_dict['labels'])  # shape(10000,)
        x_data = x_data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

        if i == 0:
            x_train = x_data
            y_train = y_data
        else:
            x_train = np.concatenate((x_train, x_data), axis=0)
            y_train = np.concatenate((y_train, y_data), axis=0)

    return x_train, y_train


def read_cifar10_test(test_path):

    data_dict = unpickle(test_path)
    x_test = data_dict['data']  # shape(10000, 3072)
    y_test = np.array(data_dict['labels'])  # shape(10000,)
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)  # shape(10000, 32, 32, 3)

    return x_test, y_test


def read_cifar100_label_name(file_path, flag_fine=1):

    data_dict = unpickle(file_path)
    # dict_keys(['fine_label_names', 'coarse_label_names'])
    if flag_fine:
        label_name = data_dict['fine_label_names']
    else:
        label_name = data_dict['coarse_label_names']

    return label_name


def read_cifar100_train(train_path, flag_fine=1):

    data_dict = unpickle(train_path)
    # dict_keys(['fine_labels', 'batch_label', 'coarse_labels', 'filenames', 'data'])
    x_train = data_dict['data']  # shape(50000, 3072)
    x_train = x_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1)

    if flag_fine:
        y_train = np.array(data_dict['fine_labels'])
    else:
        y_train = np.array(data_dict['coarse_labels'])

    return x_train, y_train


def read_cifar100_test(test_path, flag_fine):

    data_dict = unpickle(test_path)
    x_test = data_dict['data']
    x_test = x_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)

    if flag_fine:
        y_test = np.array(data_dict['fine_labels'])  # shape(10000,)
    else:
        y_test = np.array(data_dict['coarse_labels'])

    return x_test, y_test


x_train, y_train = read_cifar10_train(cifar10_train_data_path)


def img_flip(img, mode):

    new_img = np.zeros_like(img)

    if mode == 'horizon':
        new_img = np.flip(img, axis=1)

    elif mode == 'vertical':
        new_img = np.flip(img, axis=0)

    return new_img


def img_transition(img, par, mode='nearest'):

    img_r = img.shape[0]
    img_c = img.shape[1]
    (shift_r, shift_c) = par

    # shift_r = random.randint(-int(img_r * 0.3), int(img_r * 0.3))
    # shift_c = random.randint(-int(img_c * 0.3), int(img_c * 0.3))

    new_img = scimg_nd.shift(img, (shift_r, shift_c, 0), mode=mode)

    return new_img


def img_scaling(img, scaling_r, scaling_c):
    img_r = img.shape[0]
    img_c = img.shape[1]

    # scaling = random.uniform(0.5, 1.5)

    r = int(img_r * scaling_r)
    c = int(img_c * scaling_c)

    img_tmp = cv2.resize(img, None, fx=scaling_r, fy=scaling_c)

    if r < img_r:
        up = int(abs(r - img_r) / 2)
        start_r = 0
    else:
        start_r = int((r - img_r) / 2)

    if c < img_c:
        left = int(abs(c - img_c) / 2)
        start_c = 0
    else:
        start_c = int((c - img_c) / 2)

    if start_c or start_r:
        new_img = img_tmp[start_r: start_r+img_r, start_c: start_c+img_c]
    else:
        print('copy')
        new_img = cv2.copyMakeBorder(img_tmp, up, up, left, left, borderType=cv2.BORDER_REPLICATE)

    return new_img


def img_rotation(img, ang):

    img_r = img.shape[0]
    img_c = img.shape[1]

    matrix = cv2.getRotationMatrix2D((img_c/2, img_r/2), ang, 1)

    new_img = cv2.warpAffine(img, matrix, (img_r, img_c), borderMode=cv2.BORDER_REPLICATE)

    return new_img


def img_random_noise(img, theta):   # theta<=30

    noise = np.random.randint(0, theta, size=img.shape)

    new_img = np.uint8(img+noise)

    return new_img


def img_random_erasing(img, mode):

    img_r = img.shape[0]
    img_c = img.shape[1]

    mask_r = round(img_r*random.uniform(0.0, 0.5))
    mask_c = round(img_c*random.uniform(0.0, 0.5))

    mask_x = random.randint(0, img_c-mask_c-1)
    mask_y = random.randint(0, img_r-mask_r-1)

    new_img = np.zeros_like(img, dtype=np.uint8)

    if mode == 'noise':
        mask = np.random.randint(0, 30, size=(mask_r, mask_c, img.shape[-1]))
    elif mode == 'constant':
        mask = -1*img[mask_y:mask_y++mask_r, mask_x:mask_x+mask_c, :]
        print(mask)
    elif mode == 'nearest':
        mask_tmp = img[mask_y:mask_y+1, mask_x:mask_x+mask_c, :]
        for i in range(mask_r):
            if i == 0:
                mask = mask_tmp
            else:
                mask = np.concatenate((mask, mask_tmp), axis=0)

    new_img[mask_y:mask_y+mask_r, mask_x:mask_x+mask_c, :] = mask

    new_img = img+new_img

    return new_img


def img_rgb_casting(img, cast_channel):

    img_r = img.shape[0]
    img_c = img.shape[1]

    new_img = np.zeros_like(img, dtype=np.uint8)

    if cast_channel == 'r':
        cast = np.array([0, 1, 1])
    elif cast_channel == 'g':
        cast = np.array([1, 0, 1])
    elif cast_channel == 'b':
        cast = np.array([1, 1, 0])

    for i in range(img_r):
        for j in range(img_c):
            new_img[i][j] = np.multiply(img[i][j], cast)

    return new_img


def img_rgb_adjust(img, adjust_array):  # sum(adjust_array) = 1

    img_r = img.shape[0]
    img_c = img.shape[1]

    new_img = np.zeros_like(img, dtype=np.uint8)

    for i in range(img_r):
        for j in range(img_c):
            new_img[i][j] = np.multiply(img[i][j], adjust_array)

    return new_img


def img_photometric(img):   # 明暗

    bright = random.randint(1, 20)*0.1

    new_img = skimg_exp.adjust_gamma(img, bright)

    return new_img


def img_contrast(img):

    img_norm = cv2.normalize(img, dst=None, alpha=500, beta=0, norm_type=cv2.NORM_MINMAX)

    return img_norm


def img_shear(img, angle_x, angle_y):

    angle_x = math.pi*angle_x / 180.0
    angle_y = math.pi * angle_y / 180.0

    shape = img.shape
    shape_size = shape[:2]

    M_shear = np.array([[1, np.tan(angle_x), 0],
                        [np.tan(angle_y), 1, 0]], dtype=np.float32)
    # print(M_shear.shape)

    return cv2.warpAffine(img, M_shear, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101


def img_hue(img, target_hue):

    # new_img = abs(img-np.ones_like(img)*theta)

    # new_img = abs(theta * img - np.ones_like(img)*alpha)
    # new_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    r, b, g = np.split(img, 3, axis=-1)  # 将 RGB 色值分离
    result_r, result_g, result_b = [], [], []
    # 依次对每个像素点进行处理
    for pixel_r, pixel_g, pixel_b in zip(np.nditer(r), np.nditer(g), np.nditer(b)):
        # 转为 HSV 色值 色域h:[0,1] (*360degree) 饱和度s:[0,1] 亮度v:[0,1]
        # 红色为0°, 绿色为120°,蓝色为240°, 黄色为60°, 青色为180°,品红为300°
        h, s, v = colorsys.rgb_to_hsv(pixel_r / 255., pixel_g / 255., pixel_b / 255.)
        # 转回 RGB 色系
        rgb = colorsys.hsv_to_rgb(target_hue/360., s, v)
        pixel_r, pixel_g, pixel_b = [int(x * 255.) for x in rgb]
        # 每个像素点结果保存
        result_r.append(pixel_r)
        result_g.append(pixel_g)
        result_b.append(pixel_b)

    result_r = np.reshape(np.array(result_r), newshape=r.shape)
    result_g = np.reshape(np.array(result_g), newshape=g.shape)
    result_b = np.reshape(np.array(result_b), newshape=b.shape)
    new_img = np.concatenate((result_r, result_g, result_b), axis=-1)
    return new_img


def img_mismatch(img, num):

    img_s = img.shape[0]
    new_img = Image.new('RGB', (img_s, img_s))
    img = Image.fromarray(img)
    crop_coord = list()  # corp coordinate
    cropped_size = img_s//int(math.sqrt(num))
    # print(cropped_size)

    for i in range(int(math.sqrt(num))):
        for j in range(int(math.sqrt(num))):
            crop_coord.append((i*cropped_size, j*cropped_size, (i+1)*cropped_size, (j+1)*cropped_size))
            # print((i*cropped_size, j*cropped_size, (i+1)*cropped_size, (j+1)*cropped_size))

    misorder_index = list(range(num))
    shuffle(misorder_index)
    for i in range(num):
        new_img.paste(img.crop(crop_coord[i]), crop_coord[misorder_index[i]])

    return np.array(new_img)


def img_invert(img):

    img = Image.fromarray(img)

    new_img = ImageOps.invert(img)

    return np.array(new_img)


def img_contrast(img, par):
    # par=0, grey image
    # par=1, original image

    img = Image.fromarray(img)
    enh_con = ImageEnhance.Contrast(img)
    new_img = enh_con.enhance(par)

    return np.array(new_img)


def img_sharp(img, par):

    img = Image.fromarray(img)
    enh_sha = ImageEnhance.Sharpness(img)
    new_img = enh_sha.enhance(par)

    return np.array(new_img)


def img_bright(img, par):

    img = Image.fromarray(img)
    enh_bri = ImageEnhance.Brightness(img)
    new_img = enh_bri.enhance(par)

    return np.array(new_img)


def img_color(img, par):    #色度

    img = Image.fromarray(img)
    enh_col = ImageEnhance.Color(img)
    new_img = enh_col.enhance(par)

    return np.array(new_img)


def img_hue(img, par):  # par:0,1
    import colorsys

    img = Image.fromarray(img)
    ld = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            r, g, b = ld[x, y]
            h, s, v = colorsys.rgb_to_hsv(r / 255., g / 255., b / 255.)
            h = (h + -90.0 / 360.0) % par
            s = s ** 0.65
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            ld[x, y] = (int(r * 255.9999), int(g * 255.9999), int(b * 255.9999))

    return np.array(img)


def img_histogram_equalization(img):

    img = Image.fromarray(img)
    new_img = ImageOps.equalize(img)

    return np.array(new_img)


def img_posterize(img, par):

    img = Image.fromarray(img)
    new_img = ImageOps.posterize(img, par)

    return np.array(new_img)


def img_solarize(img, par):

    img = Image.fromarray(img)
    new_img = ImageOps.solarize(img, par)

    return np.array(new_img)


def img_autocontrast(img, par=None):
    img = Image.fromarray(img)
    new_img = ImageOps.autocontrast(img)

    return np.array(new_img)


'''
new_img0 = img_flip(x_train[0], 'horizon')
plt.figure(1)
plt.imshow(x_train[0])
plt.show()

plt.figure(1)
plt.imshow(new_img0)
plt.show()
'''
car = x_train[4]
frog = x_train[0]
print(car)
img = img_hue(car, 1)
print(img)
print(img.shape)
plt.figure(1)
plt.imshow(img)
# plt.imsave('C:/Users/49435/Dropbox/硕士毕设/终期/Jinyi Thesis 0302/figure/MRDesign/MRdemo/car_saturability.png', img,  format='png')
plt.show()
# frog = x_train[0]
# # plt.imsave('C:/Postgraduate/T.Y. related/MR_TrainSet/CIFAR10/img/demo/org.png', x_train[4], format='png')
# img = img_mismatch(car, 16)
# print(img.shape)
# # print(img[18][18])
# plt.figure(1)
# plt.imshow(img)
# plt.show()
# plt.imsave('C:/Postgraduate/T.Y. related/MR_TrainSet/CIFAR10/img/demo/rotation.png',img,  format='png')
# img = img_rgb_adjust(x_train[4], [0.3, 0.1, 0.6])
# plt.figure(1)
# plt.imshow(img)
# plt.show()
# plt.imsave('C:/Postgraduate/T.Y. related/MR_TrainSet/CIFAR10/img/demo/color.png', img, format='png')
'''
img = img_flip(x_train[4], 'horizon')
plt.figure(1)
plt.imshow(img)
plt.show()
plt.imsave('C:/Postgraduate/T.Y. related/MR_TrainSet/CIFAR10/img/demo/flip-h.png', img, format='png')
'''




