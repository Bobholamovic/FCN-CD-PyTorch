from os.path import join, expanduser, basename, exists, splitext

import torch
import torch.utils.data as data
import numpy as np

from .common import (default_loader, to_tensor)


class CDDataset(data.Dataset):
    def __init__(
        self, 
        root, phase,
        transforms,
        repeats
    ):
        super().__init__()
        self.root = expanduser(root)
        if not exists(self.root):
            raise FileNotFoundError
        self.phase = phase
        self.transforms = list(transforms)
        self.transforms += [None]*(3-len(self.transforms))
        self.repeats = int(repeats)

        self.t1_list, self.t2_list, self.label_list = self._read_file_paths()
        self.len = len(self.label_list)

    def __len__(self):
        return self.len * self.repeats

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index = index % self.len
        
        t1 = self.fetch_image(self.t1_list[index])
        t2 = self.fetch_image(self.t2_list[index])
        label = self.fetch_label(self.label_list[index])
        t1, t2, label = self.preprocess(t1, t2, label)
        if self.phase == 'train':
            return t1, t2, label
        else:
            return self.get_name(index), t1, t2, label

    def _read_file_paths(self):
        raise NotImplementedError
        
    def fetch_label(self, label_path):
        return default_loader(label_path)

    def fetch_image(self, image_path):
        return default_loader(image_path)

    def get_name(self, index):
        return splitext(basename(self.label_list[index]))[0]+'.bmp'

    def preprocess(self, t1, t2, label):
        if self.transforms[0] is not None:
            # Applied on all
            t1, t2, label = self.transforms[0](t1, t2, label)
        if self.transforms[1] is not None:
            # For images solely
            t1, t2 = self.transforms[1](t1, t2)
        if self.transforms[2] is not None:
            # For labels solely
            label = self.transforms[2](label)
        
        return to_tensor(t1).float(), to_tensor(t2).float(), to_tensor(label).long()