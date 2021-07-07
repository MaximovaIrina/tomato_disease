import numpy as np
import torch
import cv2
import os

from scipy.interpolate import make_interp_spline
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.features import Features


def show_less_10(img_file):
    img = cv2.imread(img_file)
    img = img[:, :, 2]
    res = np.zeros(img.shape)
    res[(img < 10) & (img > 0)] = 200
    cv2.imwrite('plot\\less_10.png', res)


def features_labels(file, scale=False):
    dataset = torch.load(file)
    features = np.asarray(dataset['features'])
    if scale is True:
        features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-9)
    labels = np.asarray(dataset['labels'])
    return features, labels


def scale(x, mean=None, std=None):
    if mean is None and std is None:
        mean, std = np.mean(x, 0), np.std(x, 0)
        return (x - mean) / std, mean, std
    else:
        return (x - mean) / std


def transform_features(x, length):
    x = x.copy()
    if length == 'long':
        return x
    elif length == 'middle':
        indices = list(range(8))
        for i in range(8, 68, 12):
            indices += list(range(i, i + 4))
        x = x[:, indices]
    elif length == 'short':
        indices = list(range(8))
        for i in range(8, 68, 12):
            ind = list(range(i, i + 4))
            x[:, i] = np.mean(x[:, ind], axis=1)
            indices += [i]
        x = x[:, indices]
    else:
        raise ValueError(f'Arg \'length\'={length} is not valid. Must be [\'long\', \'middle\', \'short\']')
    return x


def common_healthy_stat(root, path):
    file = os.path.join(os.getcwd(), path)
    if os.path.exists(file):
        data = torch.load(file)
    else:
        data_r = []
        data_ndvi = []
        files = os.listdir(os.path.join(os.getcwd(), root))
        for file in files:
            img = cv2.imread(os.path.join(root, file))
            ''' red '''
            red = img[:, :, 2]
            red_nz = red[red > 10]
            data_r += list(red_nz)
            ''' ndvi '''
            green = img[:, :, 1]
            ndvi = (green - red) / (green + red + 1e-9)
            ndvi[(red == 0) & (green == 0)] = 0
            ndvi = ndvi[(ndvi != 0) & (-0.95 < ndvi) & (ndvi < 0.95)]
            data_ndvi += list(ndvi)
        data = {'R_mean': np.mean(data_r), 'R_std': np.std(data_r),
                'ndvi_mean': np.mean(data_ndvi), 'ndvi_std': np.std(data_ndvi)}
    return data


def name_features():
    stat = ['MEAN', 'STD', 'MAX', 'MIN']
    hist = ['BIN_' + str(i) for i in range(4)]
    glcm = []
    for prop in ['CON', 'HOM', 'ENG', 'COR', 'ENT']:
        for d in [1, 4, 8]:
            for th in ['0', 'Pi/4', 'Pi/2', '3Pi/4']:
                glcm += [prop + '_' + str(d) + '_' + th]
    return np.concatenate([stat, hist, glcm])


def cimbine_features(channel='R', dst_root='comb'):
    fileg = os.path.join(os.getcwd(), f'..\\data\\global\\test\\{channel}.pth')
    filel = os.path.join(os.getcwd(), f'..\\data\\local\\test\\{channel}.pth')
    xg, y = features_labels(fileg, scale=False)
    xl, _ = features_labels(filel, scale=False)
    x = np.concatenate([xl[:, :8], xg[:, 8:]], axis=1)
    file = os.path.join(os.getcwd(),  dst_root, 'test', f'{channel}.pth')
    data = {'features': x, 'labels': y}
    torch.save(data, file)

    fileg = os.path.join(os.getcwd(), f'..\\data\\global\\train\\{channel}.pth')
    filel = os.path.join(os.getcwd(), f'..\\data\\local\\train\\{channel}.pth')
    xg, y = features_labels(fileg, scale=False)
    xl, _ = features_labels(filel, scale=False)
    x = np.concatenate([xl[:, :8], xg[:, 8:]], axis=1)
    file = os.path.join(os.getcwd(), dst_root, 'train', f'{channel}.pth')
    data = {'features': x, 'labels': y}
    torch.save(data, file)


def get_loc_feach_as_image(img_pth):
    PI = np.pi
    MODE, CHANNEL = 'local', 'ndvi'
    bins, dist = [-2, -1, 0, 1, 2], [1, 4, 8]
    theta = [0, PI / 4, PI / 2, 3 * PI / 4]
    ds_stat = common_healthy_stat('..\\ds\\Healthy', path=f'..\\data\\stat.pth')
    model = Features(MODE, CHANNEL, ds_stat, bins, dist, theta)

    image = cv2.imread(img_pth)
    image = np.transpose(image, (2, 0, 1))
    image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
    image = np.expand_dims(image, 0)
    image = torch.tensor(image)
    # delete AVP in model
    features = model(image)
    torch.save(features, '..\\data\\Healthy_sample.pth')


def healthy_img_filter(root):
    files = os.listdir(root)
    i = 0
    for file in tqdm(files):
        img = cv2.imread(os.path.join(root, file))
        red = img[:, :, 2]
        bins = list(range(0, 256, 5))
        q_img = np.digitize(red, bins=bins) - 1

        g = greycomatrix(q_img, [1], [0], levels=len(bins) + 1, normed=True, symmetric=True)
        g = g[1:, 1:, :, :]
        hom = greycoprops(g, 'homogeneity')[0][0]
        std = np.std(red)

        if 0.35 < hom < 0.42 and 30 < std < 60:
            save = os.path.join(os.getcwd(), 'sort', file)
            cv2.imwrite(save, img)
            i += 1

        # Balance dataset
        if i > 1000:
            break


def plot_pixel_hists_fe_classes(root, delta=5):
    dirs = os.listdir(root)
    data = {k: group_flatten(os.path.join(root, k), delta) for k in tqdm(dirs)}
    bins = np.arange(0, 1, 0.1)
    hists = {k: np.histogram(data[k], bins)[0] for k in tqdm(dirs)}
    for i, k in enumerate(dirs):
        x = np.linspace(0, 1, len(hists[k]))
        hist = np.asarray(hists[k])
        hist = hist / hist.sum()
        spl = make_interp_spline(x, hist, k=2)
        xnew = np.linspace(0, 1, 100)
        power_smooth = spl(xnew)
        plt.plot(xnew, power_smooth, label=k)

    xticklabels = ['-3σ', '-2σ', '-σ', '0', 'σ', '2σ', '3σ']
    bins = np.arange(-3, 4)
    loc = np.full((len(bins),), 0.5) + np.asarray(bins) * 1./6.
    plt.xticks(loc, xticklabels)

    _, t = plt.ylim()
    plt.ylim(0, t)
    plt.xlabel('Интенсивность')
    plt.ylabel('Вероятность')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plot\\accum_hist.png')


def group_flatten(root, delta, mean=99.267, std=43):
    files = os.listdir(root)
    data = []
    for file in files:
        img = cv2.imread(os.path.join(root, file))
        img = img[:, :, 2]
        img = img[img > delta]

        # ''' ndvi '''
        # red = img[:, :, 2]
        # green = img[:, :, 1]
        # ndvi = (green - red) / (green + red + 1e-9)
        # ndvi[(red == 0) & (green == 0)] = 0
        # ndvi = ndvi[(ndvi != 0) & (-0.95 < ndvi) & (ndvi < 0.95)]
        # img = ndvi

        img = (img - mean) / std
        img = (img + 3) / 6
        data += list(img)
    return data