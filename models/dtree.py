import numpy as np

from models.bdtree import BDTree


class DTree:
    def __init__(self, lim_std=2):
        self.forestDT = None
        self.is_bin = False
        self.lim_std = lim_std

    def fit(self, x, y):
        if len(set(y)) == 2:
            self.is_bin = True
            self.forestDT = [BDTree(self.lim_std, None)]
        else:
            self.forestDT = [BDTree(self.lim_std, cls) for cls in set(y)]

        for DT in self.forestDT:
            DT.fit(x, y)

    def predict(self, x):
        prob = []
        for DT in self.forestDT:
            prb = DT.predict(x)
            prob.append(prb)
        prob = np.asarray(prob).transpose()

        predict = []
        for pr in prob:
            if self.is_bin:
                predict += [pr[0] != 0]
            else:
                predict += [np.argmax(pr)]
        return predict
