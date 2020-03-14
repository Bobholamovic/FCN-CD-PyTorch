import random
import math
from functools import partial, wraps

import numpy as np
import cv2


__all__ = [
    'Compose', 'Choose', 
    'Scale', 'DiscreteScale', 
    'Flip', 'HorizontalFlip', 'VerticalFlip', 'Rotate', 
    'Crop', 'MSCrop',
    'Shift', 'XShift', 'YShift',
    'HueShift', 'SaturationShift', 'RGBShift', 'RShift', 'GShift', 'BShift',
    'PCAJitter', 
    'ContraBrightScale', 'ContrastScale', 'BrightnessScale',
    'AddGaussNoise'
]


rand = random.random
randi = random.randint
choice = random.choice
uniform = random.uniform
# gauss = random.gauss
gauss = random.normalvariate    # This one is thread-safe

# The transformations treat 2-D or 3-D numpy ndarrays only, with the optional 3rd dim as the channel dim

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


class Choose:
    def __init__(self, *tf):
        assert len(tf) > 1
        self.tfs = tf
    def __call__(self, *x):
        idx = randi(0, len(self.tfs)-1)
        return self.tfs[idx](*x)


class Scale(Transform):
    def __init__(self, scale=(0.5,1.0)):
        if _istuple(scale):
            assert len(scale) == 2
            self.scale_range = tuple(scale) #sorted(scale)
            self.scale = float(scale[0])
            super(Scale, self).__init__(random_state=True)
        else:
            super(Scale, self).__init__(random_state=False)
            self.scale = float(scale)
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
        self.bins = tuple(bins)
        self.keep_prob = float(keep_prob)
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


class Rotate(Flip):
    _directions = ('90', '180', '270', 'no')


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
            assert self.crop_size <= (h, w)
            ch, cw = self.crop_size
            if (ch,cw) == (h,w):
                return x
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


class Shift(Transform):
    def __init__(self, x_shift=(-0.0625, 0.0625), y_shift=(-0.0625, 0.0625), circular=True):
        super(Shift, self).__init__(random_state=_istuple(x_shift) or _istuple(y_shift))

        if _istuple(x_shift):
            self.xshift_range = tuple(x_shift)
            self.xshift = float(x_shift[0])
        else:
            self.xshift = float(x_shift)
            self.xshift_range = (self.xshift, self.xshift)

        if _istuple(y_shift):
            self.yshift_range = tuple(y_shift)
            self.yshift = float(y_shift[0])
        else:
            self.yshift = float(y_shift)
            self.yshift_range = (self.yshift, self.yshift)

        self.circular = circular

    def _transform(self, im):
        h, w = im.shape[:2]
        xsh = -int(self.xshift*w)
        ysh = -int(self.yshift*h)
        if self.circular:
            # Shift along the x-axis
            im_shifted = np.concatenate((im[:, xsh:], im[:, :xsh]), axis=1)
            # Shift along the y-axis
            im_shifted = np.concatenate((im_shifted[ysh:], im_shifted[:ysh]), axis=0)
        else:
            zeros = np.zeros(im.shape)
            im1, im2 = (zeros, im) if xsh < 0 else (im, zeros)
            im_shifted = np.concatenate((im1[:, xsh:], im2[:, :xsh]), axis=1)
            im1, im2 = (zeros, im_shifted) if ysh < 0 else (im_shifted, zeros)
            im_shifted = np.concatenate((im1[ysh:], im2[:ysh]), axis=0)

        return im_shifted
        
    def _set_rand_param(self):
        self.xshift = uniform(*self.xshift_range)
        self.yshift = uniform(*self.yshift_range)


class XShift(Shift):
    def __init__(self, x_shift=(-0.0625, 0.0625), circular=True):
        super(XShift, self).__init__(x_shift, 0.0, circular)


class YShift(Shift):
    def __init__(self, y_shift=(-0.0625, 0.0625), circular=True):
        super(YShift, self).__init__(0.0, y_shift, circular)


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
            dtype = x.dtype
            # The calculations are done with floating type in case of overflow
            # This is a stupid yet simple way
            x = tf(obj, np.clip(x.astype(np.float32), *obj.limit))
            # Convert back to the original type
            return np.clip(x, *obj.limit).astype(dtype)
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
                    raise ValueError("please specify the shift value (or range) for every channel.")
                rs = all(_istuple(s) for s in shift)
                self.shift = self.range = shift
            else:
                rs = False
                self.shift = [shift for _ in range(_nc)]
                self.range = [(shift, shift) for _ in range(_nc)]
                
        self.random_state = rs
        
        def _(x):
            return x
        self.convert_to = _
        self.convert_back = _
    
    @_ValueTransform.keep_range
    def _transform(self, x):
        x = self.convert_to(x)
        for i, c in enumerate(self._channel):
            x[...,c] = self._clip(x[...,c]+float(self.shift[i]))
        x = self.convert_back(x)
        return x
        
    def _clip(self, x):
        return x
        
    def _set_rand_param(self):
        if len(self._channel) == 1:
            self.shift = [uniform(*self.range)]
        else:
            self.shift = [uniform(*r) for r in self.range]


class HSVShift(ColorJitter):
    def __init__(self, shift, limit):
        super().__init__(shift, limit)
        def _convert_to(x):
            x = x.astype(np.float32)
            # Normalize to [0,1]
            x -= self.limit[0]
            x /= self.limit_range
            x = cv2.cvtColor(x, code=cv2.COLOR_RGB2HSV)
            return x
        def _convert_back(x):
            x = cv2.cvtColor(x.astype(np.float32), code=cv2.COLOR_HSV2RGB)
            return x * self.limit_range + self.limit[0]
        # Pack conversion methods
        self.convert_to = _convert_to
        self.convert_back = _convert_back

        def _clip(self, x):
            raise NotImplementedError
        

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
        x = x - x_mean
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
        if not math.isclose(self.alpha, 1.0):
            x *= self.alpha
        if not math.isclose(self.beta, 0.0):
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
            raise ValueError("the input images should be of same size.")
        self._im_shape = shape
        return super().__call__(*args)
        

class AddGaussNoise(_AddNoise):
    def __init__(self, mu=0.0, sigma=0.1, limit=(0, 255)):
        super().__init__(limit)
        self.mu = mu
        self.sigma = sigma
    def _set_rand_param(self):
        self.noise_map = np.random.randn(*self._im_shape)*self.sigma + self.mu