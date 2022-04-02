import time


class Timer:
    def __init__(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.count_prepare_time = 0
        self.count_process_time = 0
        self.last_time = time.time()

    def reset(self):
        self.total_prepare_time = 0
        self.total_process_time = 0
        self.count_prepare_time = 0
        self.count_process_time = 0
        self.last_time = time.time()

    def record_time(self):
        self.last_time = time.time()

    def add_prepare_time(self):
        current_time = time.time()
        self.total_prepare_time += current_time - self.last_time
        self.count_prepare_time += 1
        self.last_time = current_time

    def add_process_time(self):
        current_time = time.time()
        self.total_process_time += current_time - self.last_time
        self.count_process_time += 1
        self.last_time = current_time

    def get_prepare_time(self):
        return self.total_prepare_time / (self.count_prepare_time + 1e-12)

    def get_process_time(self):
        return self.total_process_time / (self.count_process_time + 1e-12)

    def tostring(self):
        summary = 'time: '
        if self.count_prepare_time > 0:
            summary += '{:.3f}s/'.format(self.get_prepare_time())
        summary += '{:.3f}s'.format(self.get_process_time())
        return summary


class TimerDict:
    def __init__(self):
        self.total_time = {}
        self.count_time = {}
        self.last_time = {}
        self.timer_keys = []

    def add_timer(self, key):
        self.total_time[key] = 0.0
        self.count_time[key] = 0
        self.last_time[key] = 0.0
        self.timer_keys.append(key)

    def tic(self, key):
        if key not in self.timer_keys:
            self.add_timer(key)
        self.last_time[key] = time.time()

    def toc(self, key):
        assert key in self.timer_keys
        duration = time.time() - self.last_time[key]
        self.total_time[key] += duration
        self.count_time[key] += 1

    def get_time(self, key):
        assert key in self.timer_keys
        return self.total_time[key] / (float(self.count_time[key]) + 1e-12)

    def summary(self, keys):
        summary = 'time: '
        summary += '/'.join(['{:.3f}s'.format(self.get_time(key)) for key in keys])
        return summary
