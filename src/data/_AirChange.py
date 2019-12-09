import abc
from os.path import join, basename
from multiprocessing import Manager

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

        self._manager = Manager()
        sync_list = self._manager.list
        self.images = sync_list([sync_list([None]*self.N_PAIRS), sync_list([None]*self.N_PAIRS)])
        self.labels = sync_list([None]*self.N_PAIRS)

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
            sample_ids = range(self.N_PAIRS)
            t1_list = ['-'.join([self.LOCATION,str(i),'0.bmp']) for i in sample_ids if i not in self.TEST_SAMPLE_IDS]
            t2_list = ['-'.join([self.LOCATION,str(i),'1.bmp']) for i in sample_ids if i not in self.TEST_SAMPLE_IDS]
            label_list = ['-'.join([self.LOCATION,str(i),'cm.bmp']) for i in sample_ids if i not in self.TEST_SAMPLE_IDS]
        else:
            t1_list = ['-'.join([self.LOCATION,str(i),'0.bmp']) for i in self.TEST_SAMPLE_IDS]
            t2_list = ['-'.join([self.LOCATION,str(i),'1.bmp']) for i in self.TEST_SAMPLE_IDS]
            label_list = ['-'.join([self.LOCATION,str(i),'cm.bmp']) for i in self.TEST_SAMPLE_IDS]

        return t1_list, t2_list, label_list

    def fetch_image(self, image_name):
        _, i, t = image_name.split('-')
        i, t = int(i), int(t[:-4])
        if self.images[t][i] is None:
            image = self._bmp_loader(join(self.root, self.LOCATION, str(i+1), 'im'+str(t+1)))
            self.images[t][i] = image if self.phase == 'train' else self.cropper(image)
        return self.images[t][i]

    def fetch_label(self, label_name):
        index = int(label_name.split('-')[1])
        if self.labels[index] is None:
            label = self._bmp_loader(join(self.root, self.LOCATION, str(index+1), 'gt'))
            label = (label / 255.0).astype(np.uint8)    # To 0,1
            self.labels[index] = label if self.phase == 'train' else self.cropper(label)
        return self.labels[index]

    @staticmethod
    def _bmp_loader(bmp_path_wo_ext):
        # Case insensitive .bmp loader
        try:
            return default_loader(bmp_path_wo_ext+'.bmp')
        except FileNotFoundError:
            return default_loader(bmp_path_wo_ext+'.BMP')

        