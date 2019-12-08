from sklearn import metrics


class AverageMeter:
    def __init__(self, callback=None):
        super().__init__()
        self.callback = callback
        self.reset()

    def compute(self, *args):
        if self.callback is not None:
            return self.callback(*args) 
        elif len(args) == 1:
            return args[0]
        else:
            raise NotImplementedError

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, *args, n=1):
        self.val = self.compute(*args)
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count


class Metric(AverageMeter):
    __name__ = 'Metric'
    def __init__(self, callback, **configs):
        super().__init__(callback)
        self.configs = configs
    
    def compute(self, pred, true):
        return self.callback(true.ravel(), pred.ravel(), **self.configs)


class Precision(Metric):
    __name__ = 'Prec.'
    def __init__(self, **configs):
        super().__init__(metrics.precision_score, **configs)


class Recall(Metric):
    __name__ = 'Recall'
    def __init__(self, **configs):
        super().__init__(metrics.recall_score, **configs)


class Accuracy(Metric):
    __name__ = 'OA'
    def __init__(self, **configs):
        super().__init__(metrics.accuracy_score, **configs)


class F1Score(Metric):
    __name__ = 'F1'
    def __init__(self, **configs):
        super().__init__(metrics.f1_score, **configs)