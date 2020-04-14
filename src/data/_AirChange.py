import abc
from os.path import join, basename
from functools import lru_cache

import numpy as np

from . import CDDataset
from .common import default_loader
from .augmentation import Crop


class _AirChangeDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1
    ):
        super().__init__(root, phase, transforms, repeats)
        self.cropper = Crop(bounds=(0, 0, 748, 448))

    @property
    @abc.abstractmethod
    def LOCATION(self):
        return ''

    @property
    @abc.abstractmethod
    def TEST_SAMPLE_IDS(self):
        return ()

    @property
    @abc.abstractmethod
    def N_PAIRS(self):
        return 0

    def _read_file_paths(self):
        if self.phase == 'train':
            sample_ids = [i for i in range(self.N_PAIRS) if i not in self.TEST_SAMPLE_IDS]
            t1_list = [join(self.root, self.LOCATION, str(i+1), 'im1') for i in sample_ids]
            t2_list = [join(self.root, self.LOCATION, str(i+1), 'im2') for i in sample_ids]
            label_list = [join(self.root, self.LOCATION, str(i+1), 'gt') for i in sample_ids]
        else:
            t1_list = [join(self.root, self.LOCATION, str(i+1), 'im1') for i in self.TEST_SAMPLE_IDS]
            t2_list = [join(self.root, self.LOCATION, str(i+1), 'im2') for i in self.TEST_SAMPLE_IDS]
            label_list = [join(self.root, self.LOCATION, str(i+1), 'gt') for i in self.TEST_SAMPLE_IDS]

        return t1_list, t2_list, label_list


    @lru_cache(maxsize=8)
    def fetch_image(self, image_name):
        image = self._bmp_loader(image_name)
        return image if self.phase == 'train' else self.cropper(image)

    @lru_cache(maxsize=8)
    def fetch_label(self, label_name):
        label = self._bmp_loader(label_name)
        label = (label / 255.0).astype(np.uint8)    # To 0,1
        return label if self.phase == 'train' else self.cropper(label)

    def get_name(self, index):
        return '{loc}-{id}-cm.bmp'.format(loc=self.LOCATION, id=index)

    @staticmethod
    def _bmp_loader(bmp_path_wo_ext):
        # Case insensitive .bmp loader
        try:
            return default_loader(bmp_path_wo_ext+'.bmp')
        except FileNotFoundError:
            return default_loader(bmp_path_wo_ext+'.BMP')