import os
from glob import glob
from os.path import join, basename
from multiprocessing import Manager

import numpy as np

from . import CDDataset
from .common import default_loader

class OSCDDataset(CDDataset):
    __BAND_NAMES = (
        'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 
        'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
    )
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        cache_level=1
    ):
        super().__init__(root, phase, transforms, repeats)
        # 0 for no cache, 1 for caching labels only, 2 and higher for caching all
        self.cache_level = int(cache_level)
        if self.cache_level > 0:
            self._manager = Manager()
            self._pool = self._manager.dict()

    def _read_file_paths(self):
        image_dir = join(self.root, 'Onera Satellite Change Detection dataset - Images')
        label_dir = join(self.root, 'Onera Satellite Change Detection dataset - Train Labels')
        txt_file = join(image_dir, 'train.txt')
        # Read cities
        with open(txt_file, 'r') as f:
            cities = [city.strip() for city in f.read().strip().split(',')]
        if self.phase == 'train':
            # For training, use the first 11 pairs
            cities = cities[:-3]
        else:
            # For validation, use the remaining 3 pairs
            cities = cities[-3:]
            
        # Use resampled images
        t1_list = [[join(image_dir, city, 'imgs_1_rect', band+'.tif') for band in self.__BAND_NAMES] for city in cities]
        t2_list = [[join(image_dir, city, 'imgs_2_rect', band+'.tif') for band in self.__BAND_NAMES] for city in cities]
        label_list = [join(label_dir, city, 'cm', city+'-cm.tif') for city in cities]

        return t1_list, t2_list, label_list

    def fetch_image(self, image_paths):
        key = '-'.join(image_paths[0].split(os.sep)[-3:-1])
        if self.cache_level >= 2:
            image = self._pool.get(key, None)
            if image is not None:
                return image
        image = np.stack([default_loader(p) for p in image_paths], axis=-1).astype(np.float32)
        if self.cache_level >= 2:
            self._pool[key] = image
        return image

    def fetch_label(self, label_path):
        key = basename(label_path)
        if self.cache_level >= 1:
            label = self._pool.get(key, None)
            if label is not None:
                return label
        # In the tif labels, 1 for NC and 2 for C
        # Thus a -1 offset is needed
        label = default_loader(label_path) - 1
        if self.cache_level >= 1:
            self._pool[key] = label
        return label
