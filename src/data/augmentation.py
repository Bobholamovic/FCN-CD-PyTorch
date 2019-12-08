import random
from functools import partial, wraps

import numpy as np
import cv2

rand = random.random
randi = random.randint
choice = random.choice
uniform = random.uniform
# gauss = random.gauss
gauss = random.normalvariate    # This one is thread-safe

# The transformations treat numpy ndarrays only

def _istuple(x): return isinstance(x, (tuple, list))


class Transform:
    def __init__(self, random_state=False):
        self.random_state = random_state
    def _transform(self, x):
        raise NotImplementedError
    def __call__(self, *args):
        if self.random_state: self._set_rand_param()
        assert len(args) > 0
        return self._transform(args[0]) if len(args) == 1 else tuple(map(self._transform, args))
    def _set_rand_param(self):
        raise NotImplementedError

class Compose:
    def __init__(self, *tf):
        assert len(tf) > 0
        self.tfs = tf
    def __call__(self, *x):
        if len(x) > 1:
            for tf in self.tfs: x = tf(*x)
        else:
            x = x[0]
            for tf in self.tfs: x = tf(x)
        return x
        
class Scale(Transform):
    def __init__(self, scale=(0.5,1.0)):
        if _istuple(scale):
            assert len(scale) == 2
            self.scale_range = scale #sorted(scale)
            self.scale = scale[0]
            super(Scale, self).__init__(random_state=True)
        else:
            super(Scale, self).__init__(random_state=False)
            self.scale = scale
    def _transform(self, x):
        # assert x.ndim == 3
        h, w = x.shape[:2]
        size = (int(h*self.scale), int(w*self.scale))
        if size == (h,w):
            return x
        interp = cv2.INTER_LINEAR if np.issubdtype(x.dtype, np.floating) else cv2.INTER_NEAREST
        return cv2.resize(x, size, interpolation=interp)
    def _set_rand_param(self):
        self.scale = uniform(*self.scale_range)
        
class DiscreteScale(Scale):
    def __init__(self, bins=(0.5, 0.75), keep_prob=0.5):
        super(DiscreteScale, self).__init__(scale=(min(bins), 1.0))
        self.bins = bins
        self.keep_prob = keep_prob
    def _set_rand_param(self):
        self.scale = 1.0 if rand()<self.keep_prob else choice(self.bins)


class Flip(Transform):
    # Flip or rotate
    _directions = ('ud', 'lr', 'no', '90', '180', '270')
    def __init__(self, direction=None):
        super(Flip, self).__init__(random_state=(direction is None))
        self.direction = direction
        if direction is not None: assert direction in self._directions
    def _transform(self, x):
        if self.direction == 'ud':
            ## Current torch version doesn't support negative stride of numpy arrays
            return np.ascontiguousarray(x[::-1])
        elif self.direction == 'lr':
            return np.ascontiguousarray(x[:,::-1])
        elif self.direction == 'no':
            return x
        elif self.direction == '90':
            # Clockwise
            return np.ascontiguousarray(self._T(x)[:,::-1])
        elif self.direction == '180':
            return np.ascontiguousarray(x[::-1,::-1])
        elif self.direction == '270':
            return np.ascontiguousarray(self._T(x)[::-1])
        else:
            raise ValueError('invalid flipping direction')

    def _set_rand_param(self):
        self.direction = choice(self._directions)

    @staticmethod
    def _T(x):
        return np.swapaxes(x, 0, 1)
        

class HorizontalFlip(Flip):
    _directions = ('lr', 'no')
    def __init__(self, flip=None):
        if flip is not None: flip = self._directions[~flip]
        super(HorizontalFlip, self).__init__(direction=flip)
    

class VerticalFlip(Flip):
    _directions = ('ud', 'no')
    def __init__(self, flip=None):
        if flip is not None: flip = self._directions[~flip]
        super(VerticalFlip, self).__init__(direction=flip)
        

class Crop(Transform):
    _inner_bounds = ('bl', 'br', 'tl', 'tr', 't', 'b', 'l', 'r')
    def __init__(self, crop_size=None, bounds=None):
        __no_bounds = (bounds is None)
        super(Crop, self).__init__(random_state=__no_bounds)
        if __no_bounds:
            assert crop_size is not None
        else:
            if not((_istuple(bounds) and len(bounds)==4) or (isinstance(bounds, str) and bounds in self._inner_bounds)):
                raise ValueError('invalid bounds')
        self.bounds = bounds
        self.crop_size = crop_size if _istuple(crop_size) else (crop_size, crop_size)
    def _transform(self, x):
        h, w = x.shape[:2]
        if self.bounds == 'bl':
            return x[h//2:,:w//2]
        elif self.bounds == 'br':
            return x[h//2:,w//2:]
        elif self.bounds == 'tl':
            return x[:h//2,:w//2]
        elif self.bounds == 'tr':
            return x[:h//2,w//2:]
        elif self.bounds == 't':
            return x[:h//2]
        elif self.bounds == 'b':
            return x[h//2:]
        elif self.bounds == 'l':
            return x[:,:w//2]
        elif self.bounds == 'r':
            return x[:,w//2:]
        elif len(self.bounds) == 2:
            assert self.crop_size < (h, w)
            ch, cw = self.crop_size
            cx, cy = int((w-cw+1)*self.bounds[0]), int((h-ch+1)*self.bounds[1])
            return x[cy:cy+ch, cx:cx+cw]
        else:
            left, top, right, lower = self.bounds
            return x[top:lower, left:right]
    def _set_rand_param(self):
        self.bounds = (rand(), rand())
   

class MSCrop(Crop):
    def __init__(self, scale, crop_size=None):
        super(MSCrop, self).__init__(crop_size)
        self.scale = scale  # Scale factor

    def __call__(self, lr, hr):
        if self.random_state:
            self._set_rand_param()
        # I've noticed that random scaling bounds may cause pixel misalignment
        # between the lr-hr pair, which significantly damages the training
        # effect, thus the quadruple mode is desired
        left, top, cw, ch = self._get_quad(*lr.shape[:2])
        self._set_quad(left, top, cw, ch)
        lr_crop = self._transform(lr)
        left, top, cw, ch = [int(it*self.scale) for it in (left, top, cw, ch)]
        self._set_quad(left, top, cw, ch)
        hr_crop = self._transform(hr)

        return lr_crop, hr_crop

    def _get_quad(self, h, w):
        ch, cw = self.crop_size
        cx, cy = int((w-cw+1)*self.bounds[0]), int((h-ch+1)*self.bounds[1])
        return cx, cy, cw, ch

    def _set_quad(self, left, top, cw, ch):
        self.bounds = (left, top, left+cw, top+ch)


# Color jittering and transformation
# The followings partially refer to https://github.com/albu/albumentations/
class _ValueTransform(Transform):
    def __init__(self, rs, limit=(0, 255)):
        super().__init__(rs)
        self.limit = limit
        self.limit_range = limit[1] - limit[0]
    @staticmethod
    def keep_range(tf):
        @wraps(tf)
        def wrapper(obj, x):
            # # Make a copy
            # x = x.copy()
            x = tf(obj, np.clip(x, *obj.limit))
            return np.clip(x, *obj.limit)
        return wrapper
        

class ColorJitter(_ValueTransform):
    _channel = (0,1,2)
    def __init__(self, shift=((-20,20), (-20,20), (-20,20)), limit=(0,255)):
        super().__init__(False, limit)
        _nc = len(self._channel)
        if _nc == 1:
            if _istuple(shift):
                rs = True
                self.shift = self.range = shift
            else:
                rs = False
                self.shift = (shift,)
                self.range = (shift, shift)
        else:
            if _istuple(shift):
                if len(shift) != _nc:
                    raise ValueError("specify the shift value (or range) for every channel")
                rs = all(_istuple(s) for s in shift)
                self.shift = self.range = shift
            else:
                rs = False
                self.shift = [shift for _ in range(_nc)]
                self.range = [(shift, shift) for _ in range(_nc)]
                
        self.random_state = rs
        
        def _(x):
            return x, ()
        self.convert_to = _
        self.convert_back = _
    
    @_ValueTransform.keep_range
    def _transform(self, x):
        # CAUTION! 
        # Type conversion here
        x, params = self.convert_to(x)
        for i, c in enumerate(self._channel):
            x[...,c] += self.shift[i]
            x[...,c] = self._clip(x[...,c])
        x, _ = self.convert_back(x, *params)
        return x
        
    def _clip(self, x):
        return np.clip(x, *self.limit)
        
    def _set_rand_param(self):
        if len(self._channel) == 1:
            self.shift = [uniform(*self.range)]
        else:
            self.shift = [uniform(*r) for r in self.range]


class HSVShift(ColorJitter):
    def __init__(self, shift, limit):
        super().__init__(shift, limit)
        def _convert_to(x):
            type_x = x.dtype
            x = x.astype(np.float32)
            # Normalize to [0,1]
            x -= self.limit[0]
            x /= self.limit_range
            x = cv2.cvtColor(x, code=cv2.COLOR_RGB2HSV)
            return x, (type_x,)
        def _convert_back(x, type_x):
            x = cv2.cvtColor(x.astype(np.float32), code=cv2.COLOR_HSV2RGB)
            return x.astype(type_x) * self.limit_range + self.limit[0], ()
        # Pack conversion methods
        self.convert_to = _convert_to
        self.convert_back = _convert_back
        

class HueShift(HSVShift):
    _channel = (0,)
    def __init__(self, shift=(-20, 20), limit=(0, 255)):
        super().__init__(shift, limit)
    def _clip(self, x):
        # Circular
        # Note that this works in Opencv 3.4.3, not yet tested under other versions
        x[x<0] += 360
        x[x>360] -= 360
        return x
        

class SaturationShift(HSVShift):    
    _channel = (1,)
    def __init__(self, shift=(-30, 30), limit=(0, 255)):
        super().__init__(shift, limit)
        self.range = tuple(r / self.limit_range for r in self.range)
    def _clip(self, x):
        return np.clip(x, 0, 1.0)
        

class RGBShift(ColorJitter):
    def __init__(self, shift=((-20,20), (-20,20), (-20,20)), limit=(0, 255)):
        super().__init__(shift, limit)        


class RShift(RGBShift):
    _channel = (0,)
    def __init__(self, shift=(-20,20), limit=(0, 255)):
        super().__init__(shift, limit)


class GShift(RGBShift):
    _channel = (1,)
    def __init__(self, shift=(-20,20), limit=(0, 255)):
        super().__init__(shift, limit)


class BShift(RGBShift):
    _channel = (2,)
    def __init__(self, shift=(-20,20), limit=(0, 255)):
        super().__init__(shift, limit)


class PCAJitter(_ValueTransform):
    def __init__(self, sigma=0.3, limit=(0, 255)):
        # For RGB only
        super().__init__(True, limit)
        self.sigma = sigma
        
    @_ValueTransform.keep_range
    def _transform(self, x):
        old_shape = x.shape
        x = np.reshape(x, (-1,3), order='F')   # For RGB
        x_mean = np.mean(x, 0)
        x -= x_mean
        cov_x = np.cov(x, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eig(np.mat(cov_x))
        # The eigen vectors are already unit "length"
        noise = (eig_vals * self.alpha) * eig_vecs
        x += np.asarray(noise)
        return np.reshape(x+x_mean, old_shape, order='F')
    
    def _set_rand_param(self):
        self.alpha = [gauss(0, self.sigma) for _ in range(3)]
        

class ContraBrightScale(_ValueTransform):
    def __init__(self, alpha=(-0.2, 0.2), beta=(-0.2, 0.2), limit=(0, 255)):
        super().__init__(_istuple(alpha) or _istuple(beta), limit)
        self.alpha = alpha
        self.alpha_range = alpha if _istuple(alpha) else (alpha, alpha)
        self.beta = beta
        self.beta_range = beta if _istuple(beta) else (beta, beta)
    
    @_ValueTransform.keep_range
    def _transform(self, x):
        if self.alpha != 1:
            x *= self.alpha
        if self.beta != 0:
            x += self.beta*np.mean(x)
        return x
    
    def _set_rand_param(self):
        self.alpha = uniform(*self.alpha_range)
        self.beta = uniform(*self.beta_range)


class ContrastScale(ContraBrightScale):
    def __init__(self, alpha=(0.2, 0.8), limit=(0,255)):
        super().__init__(alpha=alpha, beta=0, limit=limit)
        

class BrightnessScale(ContraBrightScale):
    def __init__(self, beta=(-0.2, 0.2), limit=(0,255)):
        super().__init__(alpha=1, beta=beta, limit=limit)


class _AddNoise(_ValueTransform):
    def __init__(self, limit):
        super().__init__(True, limit)
        self._im_shape = (0, 0)
        
    @_ValueTransform.keep_range
    def _transform(self, x):
        return x + self.noise_map
        
    def __call__(self, *args):
        shape = args[0].shape
        if any(im.shape != shape for im in args):
            raise ValueError("the input images should be of same size")
        self._im_shape = shape
        return super().__call__(*args)
        

class AddGaussNoise(_AddNoise):
    def __init__(self, mu=0.0, sigma=0.1, limit=(0, 255)):
        super().__init__(limit)
        self.mu = mu
        self.sigma = sigma
    def _set_rand_param(self):
        self.noise_map = np.random.randn(*self._im_shape)*self.sigma + self.mu
        

def __test():
    a = np.arange(12).reshape((2,2,3)).astype(np.float64)
    tf = Compose(BrightnessScale(), AddGaussNoise(), HueShift())
    print(a[...,0])
    c = tf(a)
    print(c[...,0])
    print(a[...,0])
    
    
if __name__ == '__main__':
    __test()
