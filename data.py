import torch
from torchvision import datasets, transforms


def get_transforms():
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.25,))
    ])
    return transform
    
def get_data(train=False):
    transforms = get_transforms()
    print(transforms)
    if train:
        data = datasets.CIFAR10('./data', train=True, download=True, transform=transforms)
        return data
    else:
        data = datasets.CIFAR10('./data', train=False, download=True, transform=transforms)
        return data

if __name__ == "__main__":
    print(get_data(train=True))