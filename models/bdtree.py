import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class BDTree:
    def __init__(self, lim_std, h_ind=None):
        self.lim_std = lim_std
        self.h_ind = h_ind
        self.indices = None
        self.hel_l, self.hel_r = None, None
        self.hists = []
        self.bins = []

    def fit(self, x, y):
        if self.h_ind is not None:
            y = [label == self.h_ind for label in y]

        data = [['Здоровые' if label else 'Больные'] + list(f) for label, f in zip(y, x)]
        df = pd.DataFrame(data)
        disease = df.loc[df[0] == 'Больные']
        healthy = df.loc[df[0] == 'Здоровые']

        dis_l, dis_r = self.get_limits(disease)
        self.hel_l, self.hel_r = self.get_limits(healthy)
        delta = abs(dis_l - self.hel_l) + abs(dis_r - self.hel_r)
        self.indices = np.argsort(delta)[::-1]
        self.hists, self.bins = self.plot_hists(healthy)


    def plot_hists(self, df):
        cols = df.columns[1:]
        hists, bins = [], []
        for i in self.indices:
            feach = df[cols[i]].to_numpy()
            h, b = np.histogram(feach, density=True, bins=100)
            hists += [h]
            bins += [b]
        return hists, bins

    def get_limits(self, df):
        mean = df.mean().values
        std = df.std().values
        left = mean - self.lim_std * std
        right = mean + self.lim_std * std
        return left, right

    def predict(self, x):
        prob = []
        for sample in x:
            pr = 0
            for i in self.indices:
                if self.hel_l[i] < sample[i] < self.hel_r[i]:
                    pr = max(pr, self.hists[i][np.argmax(self.bins[i] >= sample[i]) - 1])
                else:
                    pr = 0
                    break
            prob += [pr]
        return prob

    def plot(self, df):
        df.columns = ['', 'MEAN', 'STD', 'MAX', 'MIN']
        fig, axs = plt.subplots(1, len(self.indices))
        std = 3
        loc = list(range(-std, std + 1))
        xticklabels = ['-3σ', '-2σ', '-σ', '0', 'σ', '2σ', '3σ']
        for i, ax in zip(self.indices, axs):
            sns.kdeplot(data=df, x=df.columns[i + 1], hue='', ax=ax)
            ax.set_ylabel('Вероятность')
            ax.set_xlim(-3.5, 3.5)
            ax.set_xticks(loc)
            ax.set_xticklabels(xticklabels)
            ylim = ax.get_ylim()
            ax.vlines(self.hel_l[i], ylim[0], ylim[1], color='k', linestyle='--', alpha=0.3)
            ax.vlines(self.hel_r[i], ylim[0], ylim[1], color='k', linestyle='--', alpha=0.3)
            ax.fill_between(np.arange(self.hel_l[i], self.hel_r[i], 1e-2), ylim[0], ylim[1], facecolor='green',
                            alpha=0.05)
        fig.set_size_inches(16, 4)
        plt.tight_layout()
        plt.show()
