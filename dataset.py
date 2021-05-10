from torch.utils.data import Dataset
import numpy as np
import json
import cv2
import os


class PlantDiseaseDataset(Dataset):
    def __init__(self, json_file):
        if json_file is None:
            raise ValueError('Arg \'json_file\' is None')
        self.root, self.paths, self.labels = self._load_info(json_file)

    @staticmethod
    def _load_info(json_file):
        with open(json_file) as f:
            data = json.load(f)
        paths = [os.path.join(data['group_names'][grp], img)
                 for img, grp in zip(data['image_names'], data['group_id'])]
        return data['root'], paths, data['group_id']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # H, W, C = [BGR]
        path = os.path.join(self.root, self.paths[item])
        image = cv2.imread(path)
        image = image.astype(np.float32)
        # CONVERT TO PYTORCH CONVENTION: C, H, W
        image = np.transpose(image, (2, 0, 1))
        image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
        return image, self.labels[item]
