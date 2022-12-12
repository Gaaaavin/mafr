import sys
sys.path.append("../")

import os
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


np.random.seed(42)
torch.manual_seed(42)


class TrainDataset(ImageFolder):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)
        self.msk_root = self.root + "_masked"
        if transform is None:
            # Deafault transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop([112, 122]),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index: int):
        raw_path, id_target = self.imgs[index]
        img_path = raw_path.split('/')[-2:]
        msk_path = os.path.join(self.msk_root, *img_path)
        if os.path.exists(msk_path):
            success = True
        else:
            success = False
            msk_path = raw_path
        img_raw = self.transform(self.loader(raw_path))
        img_msk = self.transform(self.loader(msk_path))
        return {"masked image": img_msk, "raw image": img_raw, "identity": id_target, "mask": success}


class EvalDataset(ImageFolder):
    def __init__(self, root: str, prob_msk=1, transform=None):
        super().__init__(root, transform)
        self.msk_root = self.root + "_masked"
        self.label_same = np.random.binomial(1, prob_msk, size=len(self))
        if transform is None:
            # Deafault transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop([112, 122]),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        raw_path, id_target = self.imgs[index]
        if self.label_same[index]:
            img_path = raw_path.split('/')[-2:]
            msk_path = os.path.join(self.msk_root, *img_path)
            if os.path.exists(msk_path):
                img_path = msk_path
                mask = True
            else:
                img_path = raw_path
                mask = False
        else:
            img_path = raw_path
            mask = False
        img = self.transform(self.loader(img_path))
        return {"image": img, "identity": id_target, "mask": mask}


class DataBaseSet(ImageFolder):
    def __init__(self, root: str, transform=None):
        super().__init__(root, transform)
        if transform is None:
            # Deafault transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.CenterCrop([112, 122]),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, index):
        class_path = os.path.join(self.root, self.classes[index])
        img_name = os.listdir(class_path)[0]
        img_path = os.path.join(class_path, img_name)
        return self.transform(self.loader(img_path))

