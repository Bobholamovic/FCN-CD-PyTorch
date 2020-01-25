from glob import glob
from os.path import join, basename

import numpy as np

from . import CDDataset
from .common import default_loader

class LebedevDataset(CDDataset):
    def __init__(
        self, 
        root, phase='train', 
        transforms=(None, None, None), 
        repeats=1,
        subsets=('real', 'with_shift', 'without_shift')
    ):
        self.subsets = subsets
        super().__init__(root, phase, transforms, repeats)

    def _read_file_paths(self):
        t1_list, t2_list, label_list = [], [], []

        for subset in self.subsets:
            # Get subset directory
            if subset == 'real':
                subset_dir = join(self.root, 'Real', 'subset')
            elif subset == 'with_shift':
                subset_dir = join(self.root, 'Model', 'with_shift')
            elif subset == 'without_shift':
                subset_dir = join(self.root, 'Model', 'without_shift')
            else:
                raise RuntimeError('unrecognized key encountered')

            pattern = '*.bmp' if (subset == 'with_shift' and self.phase in ('test', 'val')) else '*.jpg'
            refs = sorted(glob(join(subset_dir, self.phase, 'OUT', pattern)))
            t1s = (join(subset_dir, self.phase, 'A', basename(ref)) for ref in refs)
            t2s = (join(subset_dir, self.phase, 'B', basename(ref)) for ref in refs)

            label_list.extend(refs)
            t1_list.extend(t1s)
            t2_list.extend(t2s)

        return t1_list, t2_list, label_list

    def fetch_label(self, label_path):
        # To {0,1}
        return (super().fetch_label(label_path) / 255.0).astype(np.uint8)  