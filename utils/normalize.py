import torch

def mean_std_normalize(img):
    return (img-torch.mean(img)) / torch.std(img)

def min_max_normalize(img):
    return (img-torch.max(img)) / (torch.max(img) - torch.min(img))

    