import time

import numpy as np

from .python_utils import safe_divide


class StatisticsMeter:
    def __init__(self):
        self.records = []

    def update(self, result):
        if isinstance(result, list) or isinstance(result, tuple):
            self.records += result
        else:
            self.records.append(result)

    def reset(self):
        self.records.clear()

    def sum(self):
        return np.sum(self.records)

    def mean(self):
        return np.mean(self.records)

    def std(self):
        return np.std(self.records)

    def median(self):
        return np.median(self.records)


class StatisticsDictMeter:
    def __init__(self):
        self.meter_dict = {}

    def register_meter(self, key):
        self.meter_dict[key] = StatisticsMeter()

    def reset_meter(self, key):
        self.meter_dict[key].reset()

    def check_key(self, key):
        if key not in self.meter_dict:
            raise KeyError('No meter for key "{}".'.format(key))

    def update(self, key, value):
        self.check_key(key)
        self.meter_dict[key].update(value)

    def update_from_result_dict(self, result_dict):
        if not isinstance(result_dict, dict):
            raise TypeError('`result_dict` must be a dict, but {} is used.'.format(type(result_dict)))
        for key, value in result_dict.items():
            if key in self.meter_dict:
                self.meter_dict[key].update(value)

    def sum(self, key):
        self.check_key(key)
        return self.meter_dict[key].sum()

    def mean(self, key):
        self.check_key(key)
        return self.meter_dict[key].mean()

    def std(self, key):
        self.check_key(key)
        return self.meter_dict[key].std()

    def median(self, key):
        self.check_key(key)
        return self.meter_dict[key].median()

    def summary(self):
        items = ['{}: {:.3f}'.format(key, meter.mean()) for key, meter in self.meter_dict.items()]
        summary = items[0]
        for item in items[1:]:
            summary += ', {}'.format(item)
        return summary


class Timer:
    def __init__(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.num_prepare_time = 0
        self.num_process_time = 0
        self.last_time = time.time()

    def reset_stats(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.num_prepare_time = 0
        self.num_process_time = 0

    def reset_time(self):
        self.last_time = time.time()

    def add_prepare_time(self):
        current_time = time.time()
        self.total_prepare_time += current_time - self.last_time
        self.num_prepare_time += 1
        self.last_time = current_time

    def add_process_time(self):
        current_time = time.time()
        self.total_process_time += current_time - self.last_time
        self.num_process_time += 1
        self.last_time = current_time

    def get_prepare_time(self):
        return self.total_prepare_time / self.num_prepare_time

    def get_process_time(self):
        return self.total_process_time / self.num_process_time
