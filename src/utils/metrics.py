from functools import partial

import numpy as np
from sklearn import metrics


class AverageMeter:
    def __init__(self, callback=None):
        super().__init__()
        if callback is not None:
            self.compute = callback
        self.reset()

    def compute(self, *args):
        if len(args) == 1:
            return args[0]
        else:
            raise NotImplementedError

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        for attr in filter(lambda a: not a.startswith('__'), dir(self)):
            obj = getattr(self, attr)
            if isinstance(obj, AverageMeter):
                AverageMeter.reset(obj)

    def update(self, *args, n=1):
        self.val = self.compute(*args)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return 'val: {} avg: {} cnt: {}'.format(self.val, self.avg, self.count)


# These metrics only for numpy arrays
class Metric(AverageMeter):
    __name__ = 'Metric'
    def __init__(self, n_classes=2, mode='separ', reduction='binary'):
        super().__init__(None)
        self._cm = AverageMeter(partial(metrics.confusion_matrix, labels=np.arange(n_classes)))
        assert mode in ('accum', 'separ')
        self.mode = mode
        assert reduction in ('mean', 'none', 'binary')
        if reduction == 'binary' and n_classes != 2:
            raise ValueError("binary reduction only works in 2-class cases")
        self.reduction = reduction
    
    def _compute(self, cm):
        raise NotImplementedError

    def compute(self, cm):
        if self.reduction == 'none':
            # Do not reduce size
            return self._compute(cm)
        elif self.reduction == 'mean':
            # Micro averaging
            return self._compute(cm).mean()
        else:
            # The pos_class be 1
            return self._compute(cm)[1]

    def update(self, pred, true, n=1):
        self._cm.update(true.ravel(), pred.ravel())
        if self.mode == 'accum':
            # Note that accumulation mode is special in that metric.val saves historical information.
            # Therefore, metric.avg IS USUALLY NOT THE "AVERAGE" VALUE YOU WANT!!! 
            # Instead, metric.val is the averaged result in the sense of metric.avg in separ mode, 
            # while metric.avg can be considered as some average of average.
            cm = self._cm.sum
        elif self.mode == 'separ':
            cm = self._cm.val
        else:
            raise NotImplementedError
        super().update(cm, n=n)

    def __repr__(self):
        return self.__name__+' '+super().__repr__()


class Precision(Metric):
    __name__ = 'Prec.'
    def _compute(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=0))


class Recall(Metric):
    __name__ = 'Recall'
    def _compute(self, cm):
        return np.nan_to_num(np.diag(cm)/cm.sum(axis=1))


class Accuracy(Metric):
    __name__ = 'OA'
    def __init__(self, n_classes=2, mode='separ'):
        super().__init__(n_classes=n_classes, mode=mode, reduction='none')
    def _compute(self, cm):
        return np.nan_to_num(np.diag(cm).sum()/cm.sum())


class F1Score(Metric):
    __name__ = 'F1'
    def _compute(self, cm):
        prec = np.nan_to_num(np.diag(cm)/cm.sum(axis=0))
        recall = np.nan_to_num(np.diag(cm)/cm.sum(axis=1))
        return np.nan_to_num(2*(prec*recall) / (prec+recall))