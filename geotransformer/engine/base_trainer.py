import sys
import argparse
import os.path as osp
import time
import json
import abc
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
import ipdb

from geotransformer.utils.summary_board import SummaryBoard
from geotransformer.utils.timer import Timer
from geotransformer.utils.torch import all_reduce_tensors, release_cuda, initialize
from geotransformer.engine.logger import Logger


def inject_default_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--snapshot', default=None, help='load from snapshot')
    parser.add_argument('--epoch', type=int, default=None, help='load epoch')
    parser.add_argument('--log_steps', type=int, default=10, help='logging steps')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for ddp')
    return parser


class BaseTrainer(abc.ABC):
    def __init__(
        self,
        cfg,
        parser=None,
        cudnn_deterministic=True,
        autograd_anomaly_detection=False,
        save_all_snapshots=True,
        run_grad_check=False,
        grad_acc_steps=1,
    ):
        # parser
        parser = inject_default_parser(parser)
        self.args = parser.parse_args()

        # logger
        log_file = osp.join(cfg.log_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
        self.logger = Logger(log_file=log_file, local_rank=self.args.local_rank)

        # command executed
        message = 'Command executed: ' + ' '.join(sys.argv)
        self.logger.info(message)

        # print config
        message = 'Configs:\n' + json.dumps(cfg, indent=4)
        self.logger.info(message)

        # tensorboard
        self.writer = SummaryWriter(log_dir=cfg.event_dir)
        self.logger.info(f'Tensorboard is enabled. Write events to {cfg.event_dir}.')

        # cuda and distributed
        if not torch.cuda.is_available():
            raise RuntimeError('No CUDA devices available.')
        self.distributed = self.args.local_rank != -1
        if self.distributed:
            torch.cuda.set_device(self.args.local_rank)
            dist.init_process_group(backend='nccl')
            self.world_size = dist.get_world_size()
            self.local_rank = self.args.local_rank
            self.logger.info(f'Using DistributedDataParallel mode (world_size: {self.world_size})')
        else:
            if torch.cuda.device_count() > 1:
                self.logger.warning('DataParallel is deprecated. Use DistributedDataParallel instead.')
            self.world_size = 1
            self.local_rank = 0
            self.logger.info('Using Single-GPU mode.')
        self.cudnn_deterministic = cudnn_deterministic
        self.autograd_anomaly_detection = autograd_anomaly_detection
        self.seed = cfg.seed + self.local_rank
        initialize(
            seed=self.seed,
            cudnn_deterministic=self.cudnn_deterministic,
            autograd_anomaly_detection=self.autograd_anomaly_detection,
        )

        # basic config
        self.snapshot_dir = cfg.snapshot_dir
        self.log_steps = self.args.log_steps
        self.run_grad_check = run_grad_check
        self.save_all_snapshots = save_all_snapshots

        # state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.iteration = 0
        self.inner_iteration = 0

        self.train_loader = None
        self.val_loader = None
        self.summary_board = SummaryBoard(last_n=self.log_steps, adaptive=True)
        self.timer = Timer()
        self.saved_states = {}

        # training config
        self.training = True
        self.grad_acc_steps = grad_acc_steps

    def save_snapshot(self, filename):
        if self.local_rank != 0:
            return

        model_state_dict = self.model.state_dict()
        # Remove '.module' prefix in DistributedDataParallel mode.
        if self.distributed:
            model_state_dict = OrderedDict([(key[7:], value) for key, value in model_state_dict.items()])

        # save model
        filename = osp.join(self.snapshot_dir, filename)
        state_dict = {
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model': model_state_dict,
        }
        torch.save(state_dict, filename)
        self.logger.info('Model saved to "{}"'.format(filename))

        # save snapshot
        snapshot_filename = osp.join(self.snapshot_dir, 'snapshot.pth.tar')
        state_dict['optimizer'] = self.optimizer.state_dict()
        if self.scheduler is not None:
            state_dict['scheduler'] = self.scheduler.state_dict()
        torch.save(state_dict, snapshot_filename)
        self.logger.info('Snapshot saved to "{}"'.format(snapshot_filename))

    def load_snapshot(self, snapshot, fix_prefix=True):
        self.logger.info('Loading from "{}".'.format(snapshot))
        state_dict = torch.load(snapshot, map_location=torch.device('cpu'))

        # Load model
        model_dict = state_dict['model']
        if fix_prefix and self.distributed:
            model_dict = OrderedDict([('module.' + key, value) for key, value in model_dict.items()])
        self.model.load_state_dict(model_dict, strict=False)

        # log missing keys and unexpected keys
        snapshot_keys = set(model_dict.keys())
        model_keys = set(self.model.model_dict().keys())
        missing_keys = model_keys - snapshot_keys
        unexpected_keys = snapshot_keys - model_keys
        if self.distributed:
            missing_keys = set([missing_key[7:] for missing_key in missing_keys])
            unexpected_keys = set([unexpected_key[7:] for unexpected_key in unexpected_keys])
        if len(missing_keys) > 0:
            message = f'Missing keys: {missing_keys}'
            self.logger.warning(message)
        if len(unexpected_keys) > 0:
            message = f'Unexpected keys: {unexpected_keys}'
            self.logger.warning(message)
        self.logger.info('Model has been loaded.')

        # Load other attributes
        if 'epoch' in state_dict:
            self.epoch = state_dict['epoch']
            self.logger.info('Epoch has been loaded: {}.'.format(self.epoch))
        if 'iteration' in state_dict:
            self.iteration = state_dict['iteration']
            self.logger.info('Iteration has been loaded: {}.'.format(self.iteration))
        if 'optimizer' in state_dict and self.optimizer is not None:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.logger.info('Optimizer has been loaded.')
        if 'scheduler' in state_dict and self.scheduler is not None:
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.logger.info('Scheduler has been loaded.')

    def register_model(self, model):
        r"""Register model. DDP is automatically used."""
        if self.distributed:
            local_rank = self.local_rank
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        self.model = model
        message = 'Model description:\n' + str(model)
        self.logger.info(message)
        return model

    def register_optimizer(self, optimizer):
        r"""Register optimizer. DDP is automatically used."""
        if self.distributed:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.world_size
        self.optimizer = optimizer

    def register_scheduler(self, scheduler):
        r"""Register LR scheduler."""
        self.scheduler = scheduler

    def register_loader(self, train_loader, val_loader):
        r"""Register data loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def optimizer_step(self, iteration):
        if iteration % self.grad_acc_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def save_state(self, key, value):
        self.saved_states[key] = release_cuda(value)

    def read_state(self, key):
        return self.saved_states[key]

    def check_invalid_gradients(self):
        for param in self.model.parameters():
            if torch.isnan(param.grad).any():
                self.logger.error('NaN in gradients.')
                return False
            if torch.isinf(param.grad).any():
                self.logger.error('Inf in gradients.')
                return False
        return True

    def release_tensors(self, result_dict):
        r"""All reduce and release tensors."""
        if self.distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self.world_size)
        result_dict = release_cuda(result_dict)
        return result_dict

    def set_train_mode(self):
        self.training = True
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        self.training = False
        self.model.eval()
        torch.set_grad_enabled(False)

    def write_event(self, phase, event_dict, index):
        r"""Write TensorBoard event."""
        if self.local_rank != 0:
            return
        for key, value in event_dict.items():
            self.writer.add_scalar(f'{phase}/{key}', value, index)

    @abc.abstractmethod
    def run(self):
        raise NotImplemented
