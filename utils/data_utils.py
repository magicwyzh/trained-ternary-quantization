import numpy as np
from PIL import Image, ImageEnhance
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def get_folders():
    
    data_dir = '/home/ubuntu/data/tiny-imagenet-200/'

    enhancers = {
        0: lambda image, f: ImageEnhance.Color(image).enhance(f),
        1: lambda image, f: ImageEnhance.Contrast(image).enhance(f),
        2: lambda image, f: ImageEnhance.Brightness(image).enhance(f),
        3: lambda image, f: ImageEnhance.Sharpness(image).enhance(f)
    }

    factors = {
        0: lambda: np.random.normal(1.0, 0.3),
        1: lambda: np.random.normal(1.0, 0.1),
        2: lambda: np.random.normal(1.0, 0.1),
        3: lambda: np.random.normal(1.0, 0.3),
    }
    
    # random enhancers in random order
    def enhance(image):
        order = [0, 1, 2, 3]
        np.random.shuffle(order)
        for i in order:
            f = factors[i]()
            image = enhancers[i](image, f)
        return image
    
    # train data augmentation on the fly
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(enhance),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # for validation data
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    train_folder = ImageFolder(data_dir + 'training', train_transform)
    val_folder = ImageFolder(data_dir + 'validation', val_transform)
    return train_folder, val_folder
