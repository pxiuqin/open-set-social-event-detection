import numpy as np
from sklearn.metrics import f1_score,accuracy_score

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0

    def __call__(self, outputs, target, loss):
        self.correct = accuracy_score(target,outputs)
        return self.value()

    def reset(self):
        self.correct = 0

    def value(self):
        return self.correct

    def name(self):
        return 'Accuracy'
    

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return float(self.correct) / self.total

    def name(self):
        return 'AccumulatedAccuracy'
    

class MacroF1Metric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.f1 = 0

    def __call__(self, outputs, target, loss):
        self.f1 = f1_score(target,outputs,average='macro')
        return self.value()

    def reset(self):
        self.f1 = 0

    def value(self):
        return self.f1

    def name(self):
        return 'Macro F1'
    

class MicroF1Metric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.f1 = 0

    def __call__(self, outputs, target, loss):
        self.f1 = f1_score(target,outputs,average='micro')
        return self.value()

    def reset(self):
        self.f1 = 0

    def value(self):
        return self.f1

    def name(self):
        return 'Micro F1'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'