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
        cache_labels=True
    ):
        super().__init__(root, phase, transforms, repeats)
        self.cache_on = cache_labels
        if self.cache_on:
            self._manager = Manager()
            self.label_pool = self._manager.dict()

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
        # t1_list, t2_list = [], []
        # for city in cities:
        #     t1s = glob(join(image_dir, city, 'imgs_1', '*_B??.tif'))
        #     t1_list.append(t1s) # Populate t1_list
        #     # Recognize t2 from t1
        #     prefix = glob(join(image_dir, city, 'imgs_2/*_B01.tif'))[0][:-5]
        #     t2_list.append([prefix+t1[-5:] for t1 in t1s])
        #
        # Use resampled images
        t1_list = [[join(image_dir, city, 'imgs_1_rect', band+'.tif') for band in self.__BAND_NAMES] for city in cities]
        t2_list = [[join(image_dir, city, 'imgs_2_rect', band+'.tif') for band in self.__BAND_NAMES] for city in cities]
        label_list = [join(label_dir, city, 'cm', city+'-cm.tif') for city in cities]
        
        
        
        #准备数据
        print('preparing %s data ... \n'%self.phase)
        pb = tqdm(list(range(len(t1_list))))
        self.t1_imgs = []
        self.t2_imgs = []
        for i in pb:
            self.t1_imgs.append(self.fetch_image(t1_list[i]))
            self.t2_imgs.append(self.fetch_image(t2_list[i]))


        return t1_list, t2_list, label_list
    
    #重写该方法
    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        index = index % self.len
        
        t1 = self.t1_imgs[index]
        t2 = self.t2_imgs[index]
        label = self.fetch_label(self.label_list[index])
        t1, t2, label = self.preprocess(t1, t2, label)
        if self.phase == 'train':
            return t1, t2, label
        else:
            return self.get_name(index), t1, t2, label
    

    def fetch_image(self, image_paths):
        return np.stack([default_loader(p) for p in image_paths], axis=-1).astype(np.float32)

    def fetch_label(self, label_path):
        if self.cache_on:
            label = self.label_pool.get(label_path, None)
            if label is not None:
                return label
        # In the tif labels, 1 for NC and 2 for C
        # Thus a -1 offset is needed
        label = default_loader(label_path) - 1
        if self.cache_on:
            self.label_pool[label_path] = label
        return label

        
