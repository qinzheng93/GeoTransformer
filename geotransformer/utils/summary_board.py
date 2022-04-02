from typing import Optional, List

from geotransformer.utils.average_meter import AverageMeter
from geotransformer.utils.common import get_print_format


class SummaryBoard:
    r"""Summary board."""

    def __init__(self, names: Optional[List[str]] = None, last_n: Optional[int] = None, adaptive=False):
        r"""Instantiate a SummaryBoard.

        Args:
            names (List[str]=None): create AverageMeter with the names.
            last_n (int=None): only the last n records are used.
            adaptive (bool=False): whether register basic meters automatically on the fly.
        """
        self.meter_dict = {}
        self.meter_names = []
        self.last_n = last_n
        self.adaptive = adaptive

        if names is not None:
            self.register_all(names)

    def register_meter(self, name):
        self.meter_dict[name] = AverageMeter(last_n=self.last_n)
        self.meter_names.append(name)

    def register_all(self, names):
        for name in names:
            self.register_meter(name)

    def reset_meter(self, name):
        self.meter_dict[name].reset()

    def reset_all(self):
        for name in self.meter_names:
            self.reset_meter(name)

    def check_name(self, name):
        if name not in self.meter_names:
            if self.adaptive:
                self.register_meter(name)
            else:
                raise KeyError('No meter for key "{}".'.format(name))

    def update(self, name, value):
        self.check_name(name)
        self.meter_dict[name].update(value)

    def update_from_result_dict(self, result_dict):
        if not isinstance(result_dict, dict):
            raise TypeError('`result_dict` must be a dict: {}.'.format(type(result_dict)))
        for key, value in result_dict.items():
            if key not in self.meter_names and self.adaptive:
                self.register_meter(key)
            if key in self.meter_names:
                self.meter_dict[key].update(value)

    def sum(self, name):
        self.check_name(name)
        return self.meter_dict[name].sum()

    def mean(self, name):
        self.check_name(name)
        return self.meter_dict[name].mean()

    def std(self, name):
        self.check_name(name)
        return self.meter_dict[name].std()

    def median(self, name):
        self.check_name(name)
        return self.meter_dict[name].median()

    def tostring(self, names=None):
        if names is None:
            names = self.meter_names
        items = []
        for name in names:
            value = self.meter_dict[name].mean()
            fmt = get_print_format(value)
            format_string = '{}: {:' + fmt + '}'
            items.append(format_string.format(name, value))
        summary = ', '.join(items)
        return summary

    def summary(self, names=None):
        if names is None:
            names = self.meter_names
        summary_dict = {name: self.meter_dict[name].mean() for name in names}
        return summary_dict
