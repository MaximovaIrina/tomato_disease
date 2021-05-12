from skimage.feature import greycomatrix, greycoprops
from joblib import Parallel, delayed
from torch import nn
import numpy as np


class Features(nn.Module):
    GLCM_PROPERTIES = ('contrast', 'homogeneity', 'energy', 'correlation')

    def __init__(self, mode: str, channel: str, ds_stat: dict, bins, dist, theta, n_jobs=8):
        super().__init__()
        if channel not in ['B', 'G', 'R', 'ndvi']:
            raise ValueError(f'Arg \'channel\'={channel} is not valid')
        if len(dist) is None or len(theta) is None:
            raise ValueError(f'Arg \'dist\' or \'theta\' is empty')
        if ds_stat is None:
            raise ValueError(f'Arg \'stats\' is empty')

        self.stride = [4, 4] if mode == 'local' else [1, 1]
        self.mask_size = [17, 17] if mode == 'local' else [257, 257]
        self.bins = bins
        self.n_bins = len(self.bins) + 1
        self.channel = channel
        self.mean = ds_stat[f'{channel}_mean']
        self.std = ds_stat[f'{channel}_std']
        self.dist = dist
        self.theta = theta
        self.n_jobs = n_jobs

    def __str__(self):
        return f"\nFeatures properties:\n\t" \
            f"mode\t{'local' if self.stride==[4, 4] else 'global'}\n\t" \
            f"channel\t{self.channel}\n\t" \
            f"mean\t{self.mean}\n\t"\
            f"std \t{self.std}\n\t" \
            f"bins\t{self.bins}\n\t" \
            f"dist\t{self.dist}\n\t" \
            f"theta\t{self.theta}"

    def all_sliding_windows(self, imgs):
        """RETURNS ARRAY OF SHAPE (B, C, Nx, Ny, MaskX, MaskY)"""
        shape = imgs.shape[:-2] + ((imgs.shape[-2] - self.mask_size[-2]) // self.stride[-2] + 1,) + \
                ((imgs.shape[-1] - self.mask_size[-1]) // self.stride[-1] + 1,) + tuple(self.mask_size)
        strides = imgs.strides[:-2] + (imgs.strides[-2] * self.stride[-2],) + (imgs.strides[-1] * self.stride[-1],) + \
                  imgs.strides[-2:]
        return np.lib.stride_tricks.as_strided(imgs, shape=shape, strides=strides)

    @staticmethod
    def stat(img):
        img = img[img != 0]
        mean = img.mean()
        std = img.std()
        maximum = (img.max() - mean) / (std + 1e-9)
        minimum = (mean - img.min()) / (std + 1e-9)
        return [mean, std, maximum, minimum]

    def quantize(self, img):
        if self.channel != 'ndvi':
            img = (img - self.mean) / self.std
            img = (img + 3) / 6
        bins_ = np.full((len(self.bins),), 0.5) + np.asarray(self.bins) * 1. / 6.
        q_img = np.digitize(img, bins_)
        return q_img

    def hist(self, q_img):
        hist, _ = np.histogram(q_img.flatten(), bins=np.arange(self.n_bins + 1), density=True)
        return hist[1: -1]

    def glcm(self, q_img):
        g = greycomatrix(q_img, self.dist, self.theta, self.n_bins, normed=True, symmetric=True)
        g = g[1:, 1:, :, :]
        props = np.array([greycoprops(g, p) for p in self.GLCM_PROPERTIES]).reshape(-1)
        entropy = -np.sum(np.multiply(g, np.log2(g + 1e-8)), axis=(0, 1)).reshape(-1)
        props = np.concatenate([props, entropy])
        return props

    def stat_hist_glcm(self, img):
        if np.max(img) == 0:
            return [0] * (4 + len(self.bins) - 1 + 3 * 4 * 5)
        stat = self.stat(img)
        q_img = self.quantize(img)
        hist = self.hist(q_img)
        glcm = self.glcm(q_img)
        return np.concatenate([stat, hist, glcm])

    def parallel_extractor(self, imgs):
        if self.channel != 'ndvi':
            imgs[imgs < 10] = 0

        shp = np.shape(imgs)
        loc_imgs = imgs.reshape((-1,) + shp[-2:])
        gcs = Parallel(n_jobs=self.n_jobs)(delayed(self.stat_hist_glcm)(img) for img in loc_imgs)
        gcs = np.stack(gcs, axis=0)
        gcs = gcs.reshape(shp[:-2] + (-1,))
        gcs = gcs.transpose(0, -1, 2, 3, 1)
        return gcs.squeeze(-1)

    # x: B, C, H, W
    def forward(self, x):
        if self.channel == 'ndvi':
            red = x[:, 2, :, :]
            green = x[:, 1, :, :]
            ndvi = (green - red) / (green + red + 1e-9)
            ndvi = (ndvi - self.mean) / self.std
            ndvi = (ndvi + 3) / 6
            ndvi[(red == 0) & (green == 0)] = 0
            x = ndvi
        else:
            ch = ['B', 'G', 'R'].index(self.channel)
            x = x[:, ch, :, :]

        # x: B, 1, H, W
        x = x.numpy()
        x = np.expand_dims(x, 1)

        # x: B, 1, Nx, Ny, MaskX, MaskY
        x = self.all_sliding_windows(x)

        # features: B, [STAT, HIST, GLCM], Nx, Ny
        features = self.parallel_extractor(x)

        # AVPool features: B, [STAT, HIST, GLCM]
        features = np.sum(features, axis=(-2, -1)) / (np.count_nonzero(features, axis=(-2, -1)) + 1e-9)
        return features
