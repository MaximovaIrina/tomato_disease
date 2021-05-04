import os
import re
import json
import argparse
import numpy as np


class Serialize:
    __atoi = staticmethod(lambda text: int(text) if text.isdigit() else text)

    def __init__(self, root):
        if root is None:
            raise ValueError('Arg \'root\' is empty')
        self.root = os.path.join(os.getcwd(), root)
        self.data = self._load_data()

    def _load_data(self):
        group_names = os.listdir(self.root)
        group_id = []
        image_names = []
        for i, group_name in enumerate(group_names):
            group_root = os.path.join(self.root, group_name)
            names = sorted(os.listdir(group_root),
                           key=lambda text: [self.__atoi(c) for c in re.split('(\d+)', text)])
            group_id.extend([i] * len(names))
            image_names.extend(names)
        return [self.root, group_names, image_names, group_id]

    @staticmethod
    def _to_dict(data):
        return {'root': data[0],
                'group_names': data[1],
                'image_names': data[2],
                'group_id': data[3]}

    def _split_data(self, split):
        root, group_names, image_names, group_id = self.data
        group_sizes = {k: group_id.count(k) for k in set(group_id)}
        n = int(split * min(group_sizes.values()))

        indices = list(range(len(image_names)))

        left = 0
        train_ind = []
        for gs in group_sizes.values():
            train_ind += indices[left: left + n]
            left += gs
        test_ind = [i for i in indices if i not in train_ind]

        train_image_names = [image_names[i] for i in train_ind]
        train_group_id = [group_id[i] for i in train_ind]

        test_image_names = [image_names[i] for i in test_ind]
        test_group_id = [group_id[i] for i in test_ind]

        train_data = [root, group_names, train_image_names, train_group_id]
        test_data = [root, group_names, test_image_names, test_group_id]
        return train_data, test_data

    def get_serialize_data(self, split=None):
        if split is None:
            return self._to_dict(self.data)
        elif split < 0 or split > 1:
            raise ValueError('Arg \'split\' must be between 0 and 1 or None')
        else:
            train_data, test_data = self._split_data(split)
            return self._to_dict(train_data), self._to_dict(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Serialize module')
    parser.add_argument('-src', action='store', dest='src', help='Dataset root folder')
    parser.add_argument('-dst', action='store', dest='dst', help='Folder for saving JSON')

    args = parser.parse_args()
    SRC_ROOT = args.src
    DST_ROOT = args.dst

    s = Serialize(SRC_ROOT)
    train_data, test_data = s.get_serialize_data(split=0.85)

    with open(f'{DST_ROOT}//train_ds.json', 'w') as f:
        json.dump(train_data, f)
        print(f'Save {f.name}')

    with open(f'{DST_ROOT}//test_ds.json', 'w') as f:
        json.dump(test_data, f)
        print(f'Save {f.name}')
