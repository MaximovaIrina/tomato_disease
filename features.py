from torch.utils.data import SubsetRandomSampler, DataLoader
from tqdm import tqdm
import numpy as np
import random
import torch
import os

from dataset import PlantDiseaseDataset
from models.features import Features
from utils.utils import common_healthy_stat

PI = np.pi
np.random.seed(42)
random.seed(1001)
torch.manual_seed(1002)


def get_and_save_features(loader, model):
    data = {'features': [], 'labels': []}
    for images, labels in tqdm(loader, desc=f'Extract features'):
        features = model(images)
        data['features'] += [features]
        data['labels'] += labels
    data['features'] = np.concatenate(data['features'], axis=0)
    return data


def get_loader(json):
    dataset = PlantDiseaseDataset(json)
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    sampler = SubsetRandomSampler(indices)
    loader = DataLoader(dataset, batch_size=50, sampler=sampler)
    return loader


if __name__ == '__main__':
    MODE = 'local'
    CHANNEL = 'ndvi'

    train_loader = get_loader(f'data\\train_ds.json')
    test_loader = get_loader(f'data\\test_ds.json')

    ds_stat = common_healthy_stat(root='ds\\Healthy', path=f'data\\stat.pth')
    print(f'\nGlobal stat: {ds_stat}')

    bins = [-2, -1, 0, 1, 2]
    dist = [1, 4, 8]
    theta = [0, PI/4, PI/2, 3*PI/4]
    model = Features(MODE, CHANNEL, ds_stat, bins, dist, theta)
    print(model)

    torch.save(model, f'data\\{CHANNEL}_{MODE}_backbone.pth')

    train_data = get_and_save_features(train_loader, model)
    save_path = os.path.join('data', MODE, 'train', CHANNEL + '.pth')
    torch.save(train_data, save_path)
    print(f'Save {save_path}')

    test_data = get_and_save_features(test_loader, model)
    save_path = os.path.join('data', MODE, 'test', CHANNEL + '.pth')
    torch.save(test_data, save_path)
    print(f'Save {save_path}')
