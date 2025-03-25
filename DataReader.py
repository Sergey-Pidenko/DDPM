import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class NormalizeToMinusOneToOne(object):
    def __call__(self, x):
        return x * 2.0 - 1.0

# Функция для преобразования в float32
class ToFloat32(object):
    def __call__(self, x):
        return x.to(torch.float32)

class SuperResolutionDataset(Dataset):
    def __init__(self, dir, size=10000, transform=None):
        self.dir = dir
        self.high_res_files = sorted(os.listdir(dir))
        if size:
            self.high_res_files = self.high_res_files[:size]
        
        # Применяем последовательность преобразований
        self.transform = transform if transform is not None else transforms.Compose([
            transforms.ToTensor(),
            NormalizeToMinusOneToOne(),
            ToFloat32()
        ])

    def __len__(self):
        return len(self.high_res_files)

    def __getitem__(self, idx):
        high_res_path = os.path.join(self.dir, self.high_res_files[idx])

        with Image.open(high_res_path) as img:
            high_res_image = img.convert('RGB')
            low_res_image = high_res_image.resize((128, 128), Image.BICUBIC)

            # Применение преобразований
            high_res_image = self.transform(high_res_image)
            low_res_image = self.transform(low_res_image)

        return low_res_image, high_res_image