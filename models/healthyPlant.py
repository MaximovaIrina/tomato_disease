import torch


class HealthyPlant(torch.nn.Module):
    def __init__(self, backbone, classifier, mean, std):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.clf_mean = mean
        self.clf_std = std

    def normalize(self, data):
        data = (data - self.clf_mean) / self.clf_std
        return data

    def forward(self, x):
        features = self.backbone(x)
        features = self.normalize(features)
        return self.classifier.predict(features)
