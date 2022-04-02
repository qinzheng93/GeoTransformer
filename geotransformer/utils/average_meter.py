import numpy as np


class AverageMeter:
    def __init__(self, last_n=None):
        self._records = []
        self.last_n = last_n

    def update(self, result):
        if isinstance(result, (list, tuple)):
            self._records += result
        else:
            self._records.append(result)

    def reset(self):
        self._records.clear()

    @property
    def records(self):
        if self.last_n is not None:
            return self._records[-self.last_n :]
        else:
            return self._records

    def sum(self):
        return np.sum(self.records)

    def mean(self):
        return np.mean(self.records)

    def std(self):
        return np.std(self.records)

    def median(self):
        return np.median(self.records)
