import numpy as np
import random
import scipy.ndimage as scimg_nd
import skimage.exposure as skimg_exp
import skimage.util as skimg_util
import cv2
import math
from PIL import Image, ImageEnhance, ImageOps


def img_flip(img, mode='horizon'):

    new_img = np.zeros_like(img, dtype=np.float)

    if mode == 'horizon':
        new_img = np.flip(img, axis=1)

    elif mode == 'vertical':
        new_img = np.flip(img, axis=0)

    return new_img


def img_transition(img, par, mode='nearest'):

    # mode = ‘constant’, ‘nearest’
    (shift_r, shift_c) = par
    # print(shift_r)
    new_img = scimg_nd.shift(img, (shift_r, shift_c, 0), mode=mode)

    return new_img


def img_scaling(img, par):

    img_r = img.shape[0]
    img_c = img.shape[1]
    (scaling_r, scaling_c) = par

    img_tmp = cv2.resize(img, None, fx=scaling_r, fy=scaling_c)
    r, c = img_tmp.shape[0], img_tmp.shape[1]
    up = left = 0
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

    if start_c > 0 or start_r > 0:
        new_img = img_tmp[start_r: start_r + img_r, start_c: start_c + img_c]
    else:
        new_img = cv2.copyMakeBorder(img_tmp, up, img_r-img_tmp.shape[0]-up, left, img_c-img_tmp.shape[1]-left,
                                     borderType=cv2.BORDER_REPLICATE)

    return new_img


def img_rotation(img, ang):

    img_r = img.shape[0]
    img_c = img.shape[1]

    matrix = cv2.getRotationMatrix2D((img_c / 2, img_r / 2), ang, 1)

    new_img = cv2.warpAffine(img, matrix, (img_r, img_c), borderMode=cv2.BORDER_REPLICATE)

    return new_img


def img_pepper_noise(img, theta):   # theta <= 10

    img_r = img.shape[0]
    img_c = img.shape[1]

    size = img_r*img_c

    new_img = img.copy()

    n_pepper = int(theta*size)

    for i in range(n_pepper):

        randx = np.random.randint(1, img_r - 1)  # 生成一个 1 至 img_r-1 之间的随机整数
        randy = np.random.randint(1, img_c - 1)  # 生成一个 1 至 img_c-1 之间的随机整数

        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            new_img[randx, randy] = 0
        else:
            new_img[randx, randy] = 1

    return new_img


def img_gaussian_noise(img, theta):

    return skimg_util.random_noise(img, mode='gaussian', var=theta, clip=True)


def img_poisson_noise(img):

    return skimg_util.random_noise(img, mode='poisson', clip=True)


def img_multiplicative_noise(img, theta):

    return skimg_util.random_noise(img, mode='speckle', var=theta, clip=True)


def img_random_erasing(img, mask_r, mask_c, mode='noise'):

    img_r = img.shape[0]
    img_c = img.shape[1]

    # mask_r = round(img_r*random.uniform(0.0, 0.5))
    # mask_c = round(img_c*random.uniform(0.0, 0.5))

    mask_x = random.randint(0, img_c-mask_c-1)
    mask_y = random.randint(0, img_r-mask_r-1)

    new_img = np.zeros_like(img, dtype=np.float)

    if mode == 'noise':
        mask = np.random.randint(0, 30, size=(mask_r, mask_c, img.shape[-1]))
    elif mode == 'constant':
        mask = -1 * img[mask_y:mask_y + +mask_r, mask_x:mask_x + mask_c, :]

    new_img[mask_y:mask_y + mask_r, mask_x:mask_x + mask_c, :] = mask

    new_img = img + new_img

    return new_img


def img_hug(img, target_hue):
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

    result_r = np.reshape(np.array(result_r, dtype=np.uint8), newshape=r.shape)
    result_g = np.reshape(np.array(result_g, dtype=np.uint8), newshape=g.shape)
    result_b = np.reshape(np.array(result_b, dtype=np.uint8), newshape=b.shape)
    new_img = np.concatenate((result_r, result_g, result_b), axis=-1)

    return new_img


def img_random_line(img, theta):

    line = np.ones(shape=(img.shape[-2], img.shape[-1]))*abs(1-img[0][0][0])

    n_line = int(img.shape[0]*theta)
    new_img = img

    for i in range(n_line):
        new_img[random.randint(0, img.shape[0]-1)] = line

    return new_img


def img_shear(img, par):

    (angle_x, angle_y) = par
    angle_x = math.pi*angle_x / 180.0
    angle_y = math.pi * angle_y / 180.0

    shape = img.shape
    shape_size = shape[:2]

    M_shear = np.array([[1, np.tan(angle_x), 0],
                        [np.tan(angle_y), 1, 0]], dtype=np.float32)
    # print(M_shear.shape)

    return cv2.warpAffine(img, M_shear, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)  # cv2.BORDER_REFLECT_101


def img_affine(img, alpha_affine):

    random_state = np.random.RandomState(None)

    shape = img.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0]+square_size, center_square[1]-square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # pts1 – Coordinates of triangle vertices in the source image.
    # pts2 – Coordinates of the corresponding triangle vertices in the destination image.
    affine_matrix = cv2.getAffineTransform(pts1, pts2)  # (2,3)
    new_img = cv2.warpAffine(img, affine_matrix, shape_size[::-1], borderMode=cv2.BORDER_REPLICATE)

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
    random.shuffle(misorder_index)
    for i in range(num):
        new_img.paste(img.crop(crop_coord[i]), crop_coord[misorder_index[i]])

    return np.array(new_img)


def img_invert(img, par=None):

    # img = Image.fromarray(img)
    #
    # new_img = ImageOps.invert(img)
    #
    # return np.array(new_img)

    new_img = abs(np.ones_like(img)-img)
    return new_img


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


def img_color(img, par):

    img = Image.fromarray(img)
    enh_col = ImageEnhance.Color(img)
    new_img = enh_col.enhance(par)

    return np.array(new_img)


def img_histogram_equalize(img, par=None):

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


def img_elastic(image, par, random_state=None):

    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    (alpha, sigma) = par
    shape = image.shape

    dx = scimg_nd.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = scimg_nd.filters.gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    # dx_tmp = (random_state.rand(*shape) * 2 - 1) * alpha
    # dy_tmp = (random_state.rand(*shape) * 2 - 1) * alpha
    # dz_tmp = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    # indices_tmp = np.reshape(y + dy_tmp, (-1, 1)), np.reshape(x + dx_tmp, (-1, 1)), np.reshape(z, (-1, 1))

    new_img = scimg_nd.map_coordinates(np.reshape(image, shape), indices, order=1, mode='reflect').reshape(shape)
    # new_img_tmp = scimg_nd.map_coordinates(np.reshape(image, shape), indices_tmp, order=1, mode='reflect').reshape(shape)

    # return new_img, new_img_tm
    return new_img


def mr_imgs_generator(x_data, y_data, k, func, func_par):
    """
    :param x_data: 源样本
    :param y_data: 样本标签
    :param k: 1个源样本生成 k个随从样本
    :param func: mr function name
    :param func_par: mr function parameter
    :return: x_mr, y_mr 随从样本及其标签
    """
    img_r = x_data.shape[1]
    img_c = x_data.shape[2]
    img_cha = x_data.shape[3]

    x_mr = np.zeros((1, img_r, img_c, img_cha), dtype=np.uint8)
    y_mr = np.zeros((1, y_data.shape[1]), dtype=np.uint8)

    for i in range(k):
        # print('{0}/{1}'.format(i + 1, k))

        for j in range(x_data.shape[0]):
            # print('{0}/{1}'.format(j + 1, x_data.shape[0]), end='\r')
            # par = func_par()
            # print(par)
            img = np.reshape(func(x_data[j], func_par()), (-1, img_r, img_c, img_cha))
            x_mr = np.concatenate((x_mr, img), axis=0)
            y_mr = np.concatenate((y_mr, np.expand_dims(y_data[j], axis=0)), axis=0)

            # plt.imsave('./train_mr_img/{}_label{}.png'.format(j+1, np.argmax(y_data[j], axis=0)), x_data[j], format='png')
            # plt.imsave('./train_mr_img/{}_label{}_mr.png'.format(j+1, np.argmax(y_data[j], axis=0)), np.reshape(img, (32,32,3)), format='png')
    x_mr = np.delete(x_mr, 0, axis=0)
    y_mr = np.delete(y_mr, 0, axis=0)

    # print('x_mr.shape', x_mr.shape)

    return x_mr, y_mr


def mr_output(x_data, y_data, mr_name, k, par=None):
    img_r = x_data.shape[1]
    img_c = x_data.shape[2]

    if mr_name == 'affine':
        mr_func = img_affine
        if par is None:
            def func_par(): return random.randint(8, 12)
        else:
            def func_par(): return par

    elif mr_name == 'autocontrast':
        mr_func = img_autocontrast
        def func_par(): return None

    elif mr_name == 'bright':
        mr_func = img_bright
        if par is None:
            def func_par():
                while True:
                    ran_par = random.uniform(0.1, 1.9)
                    if abs(ran_par-1) > 0.01:
                        return ran_par
        else:
            def func_par(): return par

    elif mr_name == 'color':
        mr_func = img_color
        if par is None:
            def func_par():
                while True:
                    ran_par = random.uniform(0.1, 1.9)
                    if abs(ran_par-1) > 0.3:
                        return ran_par
        else:
            def func_par(): return par

    elif mr_name == 'contrast':
        mr_func = img_contrast
        if par is None:
            def func_par():
                while True:
                    ran_par = random.uniform(0.1, 1.9)
                    if abs(ran_par-1) > 0.5:
                        return ran_par
        else:
            def func_par(): return par

    elif mr_name == 'elastic':
        mr_func = img_elastic
        if par is None:
            def func_par():
                elastic_alpha = random.randint(75, 80)
                elastic_sigma = 80
                ran_par = (elastic_alpha, elastic_sigma)
                return ran_par
        else:
            def func_par():
                return par

    elif mr_name == 'equalize':
        mr_func = img_histogram_equalize
        def func_par(): return None

    elif mr_name == 'flip_h':
        mr_func = img_flip
        def func_par(): return 'horizon'

    elif mr_name == 'hug':
        mr_func = img_hug
        if par is None:
            def func_par():
                tar_color = [0, 60, 120, 180, 240, 300]
                return random.choice(tar_color)
        else:
            def func_par(): return par

    elif mr_name == 'invert':
        mr_func = img_invert
        def func_par(): return None

    elif mr_name == 'mismatch':
        mr_func = img_mismatch
        if par is None:
            def func_par(): return 4
        else:
            def func_par(): return par

    elif mr_name == 'pepper':
        mr_func = img_pepper_noise
        if par is None:
            def func_par(): return random.randint(10, 20)
        else:
            def func_par(): return par

    elif mr_name == 'posterize':
        mr_func = img_posterize
        if par is None:
            def func_par():
                return random.randint(4, 8)
        else:
            def func_par(): return par

    elif mr_name == 'sharp':
        mr_func = img_sharp
        if par is None:
            def func_par():
                while True:
                    ran_par = random.uniform(0.1, 1.9)
                    if abs(ran_par-1) > 0.5:
                        return ran_par
        else:
            def func_par(): return par

    elif mr_name == 'solarize':
        mr_func = img_solarize
        if par is None:
            def func_par():
                return random.randint(0, 200)
        else:
            def func_par(): return par

    elif mr_name == 'rotation':
        mr_func = img_rotation
        if par is None:
            def func_par():
                return random.randint(10, 45)
        else:
            def func_par(): return par

    elif mr_name == 'scalling':
        mr_func = img_scaling
        if par is None:
            def func_par():
                while True:
                    scaling = random.uniform(0.4, 1.6)
                    if scaling > 0.8 and scaling < 1.2:
                        continue
                    else:
                        ran_par = (scaling, scaling)
                        return ran_par

    elif mr_name == 'shear':
        mr_func = img_shear
        if par is None:
            def func_par():
                while True:
                    theta_x = random.randrange(-20, 20)  # [a,b]
                    theta_y = random.randrange(-20, 20)
                    if (abs(theta_x) >= 10) or (abs(theta_y) >= 10):
                        break
                ran_par = (theta_x, theta_y)
                return ran_par
        else:
            def func_par():
                return par

    elif mr_name == 'transition':
        mr_func = img_transition
        if par is None:
            def func_par():
                while True:
                    shift_r = random.randint(-int(img_r * 0.2), int(img_r * 0.2))
                    shift_c = random.randint(-int(img_c * 0.2), int(img_c * 0.2))
                    if shift_r + shift_c > 0:
                        break
                ran_par = (shift_r, shift_c)
                return ran_par
        else:
            def func_par(): return par
    elif mr_name == 'transitionX':
        mr_func = img_transition
        if par is None:
            def func_par():
                while True:
                    shift_r = 0
                    shift_c = random.randint(-int(img_c * 0.2), int(img_c * 0.2))
                    if shift_r + shift_c > 0:
                        break
                ran_par = (shift_r, shift_c)
                return ran_par
        else:
            def func_par(): return (par, 0)
    elif mr_name == 'transitionY':
        mr_func = img_transition
        if par is None:
            def func_par():
                while True:
                    shift_r = random.randint(-int(img_r * 0.2), int(img_r * 0.2))
                    shift_c = 0
                    if shift_r + shift_c > 0:
                        break
                ran_par = (shift_r, shift_c)
                return ran_par
        else:
            def func_par(): return (0, par)

    elif mr_name == 'zoom_in':
        mr_func = img_scaling
        if par is None:
            def func_par():
                scaling = random.uniform(1.2, 1.6)
                ran_par = (scaling, scaling)
                return ran_par
        else:
            def func_par(): return par

    elif mr_name == 'zoom_out':
        mr_func = img_scaling
        if par is None:
            def func_par():
                scaling = random.uniform(0.4, 0.8)
                ran_par = (scaling, scaling)
                return ran_par
        else:
            def func_par():
                return par

    else:
        print('\nMR not found')
        return 0, 0

    return mr_imgs_generator(x_data, y_data, k, mr_func, func_par)


