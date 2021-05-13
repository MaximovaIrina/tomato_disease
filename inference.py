import numpy as np
import torch
import time
import cv2

from models.healthyPlant import HealthyPlant


def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    image = image.astype(np.float32)
    image = np.transpose(image, (2, 0, 1))
    image = np.pad(image, [(0, 0), (0, 1), (0, 1)], mode='constant')
    image = np.expand_dims(image, 0)
    return torch.from_numpy(image)


def load_model(backbone_path, classifier_path):
    data = torch.load(classifier_path)
    classifier = data['classifier']
    mean = data['mean']
    std = data['std']
    backbone = torch.load(backbone_path)
    backbone.eval()
    return HealthyPlant(backbone, classifier, mean, std)


if __name__ == '__main__':
    backbone_path = 'data\\R_local_backbone.pth'
    classifier_path = 'data\\state\\SVC_comb_R.pth.pth'
    image_path = 'sample\\Bacterial_spot.jpg'
    model = load_model(backbone_path, classifier_path)
    image = load_and_preprocess(image_path)

    start = time.perf_counter()
    prediction = model(image)
    end = time.perf_counter()
    print(f"Time: ", end - start)
    print(f'Prediction: {prediction}')
