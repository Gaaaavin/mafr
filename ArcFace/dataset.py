import sys
sys.path.append("../")

import os
from tqdm import tqdm

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from utils import load_n_add_mask


np.random.seed(42)
torch.manual_seed(42)


class TrainDataset(ImageFolder):
    def __init__(self, root: str, msk_root: str, msk_args, transform=None):
        super().__init__(root, transform)
        self.msk_root = msk_root
        self.msk_args = msk_args
        if transform is None:
            # Deafault transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __getitem__(self, index: int):
        raw_path, id_target = self.imgs[index]
        img_path = raw_path.split()[-2:]
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
    def __init__(self, root: str, msk_args, prob_pos=0.5, prob_msk=0.5, transform=None):
        super().__init__(root, transform)
        self.msk_args = msk_args
        self.classes = os.listdir(root)
        self.msk_prob=prob_msk
        if transform is None:
            # Deafault transform
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Generate list
        self.img_path = []
        self.label_same = []
        self.label_mask = []
        for i in range(len(self)):
            anchor_path, anchor_id = self.samples[i]
            j = anchor_id
            if np.random.uniform(0, 1) < prob_pos:
                self.label_same.append(1)
            else:
                self.label_same.append(0)
                while j == anchor_id:
                    j = np.random.choice(len(self.classes))
            choices = os.listdir(os.path.join(root, self.classes[j]))
            other_path = os.path.join(root, self.classes[j], choices[np.random.choice(len(choices))])

            self.label_mask.append(np.random.uniform(0, 1, size=2)) # seperate for anchor and other
           
            self.img_path.append((anchor_path, other_path))


    # def __len__(self):
    #     return len(self.classes)

    def __getitem__(self, index):
        anchor_path, other_path = self.img_path[index]
        if self.label_mask[index][0] < self.msk_prob:
            img_anchor, _, _ = load_n_add_mask(anchor_path, self.msk_args)
        else:
            img_anchor = self.loader(anchor_path)

        if self.label_mask[index][1] < self.msk_prob:
            img_other, _, _ = load_n_add_mask(other_path, self.msk_args)
        else:
            img_other = self.loader(anchor_path)

        img_anchor = self.transform(img_anchor)
        img_other = self.transform(img_other)

        return {"anchor": img_anchor, "other": img_other, "same": self.label_same[index]}
