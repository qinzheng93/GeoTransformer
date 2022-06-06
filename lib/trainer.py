import os, gc
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from lib.utils import AverageMeter, Logger

class Trainer(object):
    '''

    '''
    def __init__(self, args):
        self.config = args
        ###########################
        # parameters
        ###########################
        self.verbose = args.verbose
        self.verbose_freq = args.verbose_freq
        self.start_epoch = 1
        self.max_epoch = args.max_epoch
        self.training_max_iter = args.training_max_iter
        self.val_max_iter = args.val_max_iter
        self.device = args.device
        self.best_loss = 1e5
        self.best_matching_recall = -1e5
        self.best_local_matching_precision = -1e5

        self.save_dir = args.save_dir
        self.snapshot_dir = args.snapshot_dir

        self.model = args.model.to(self.device)
        self.optimizer = args.optimizer
        self.scheduler = args.scheduler
        self.scheduler_interval = args.scheduler_interval
        self.snapshot_interval = args.snapshot_interval
        self.iter_size = args.iter_size

        self.w_matching_loss = args.w_matching_loss
        self.w_local_matching_loss = args.w_local_matching_loss

        self.writer = SummaryWriter(logdir=args.tboard_dir)
        self.logger = Logger(self.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if args.pretrain != '':
            self._load_pretrain(args.pretrain)

        self.loader = dict()

        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader
        self.desc_loss = args.desc_loss

        with open(f'{args.snapshot_dir}/model.log', 'w') as f:
            f.write(str(self.model))

        f.close()


    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch: index of epoch of current model
        :param name: path to the saving model
        :return: None
        '''
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_matching_recall': self.best_matching_recall,
            'best_local_matching_precision': self.best_local_matching_precision
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        '''
        Load a pretrained model
        :param resume: the path to the pretrained model
        :return: None
        '''
        if os.path.isfile(resume):
            print(f'=> loading checkpoint {resume}')
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']
            self.best_matching_recall = state['best_matching_recall']
            self.best_local_matching_precision = state['best_local_matching_precision']

            self.logger.write(f'Successfully load pretrained model from {resume}!\n')
            self.logger.write(f'Current best loss {self.best_loss}\n')
            self.logger.write(f'Current best matching recall {self.best_matching_recall}\n')
            self.logger.write(f'Current best local matching precision {self.best_local_matching_precision}\n')

        else:
            raise ValueError(f'=> no checkpoint found at {resume}')

    def _get_lr(self, group=0):
        '''
        Get current learning rate
        :param group:
        :return: None
        '''
        return self.optimizer.param_groups[group]['lr']

    def stats_dict(self):
        '''
        Create the dict of all metrics
        :return: stats: the dict containing all metrics
        '''
        stats = dict()
        stats['matching_loss'] = 0.
        stats['matching_recall'] = 0.
        stats['local_matching_loss'] = 0.
        stats['local_matching_precision'] = 0.
        stats['total_loss'] = 0.
        '''
        to be added
        '''
        return stats

    def stats_meter(self):
        '''
        For each metric in stats dict, create an AverageMeter() for update
        :return: meters: dict of AverageMeter()
        '''
        meters = dict()
        stats = self.stats_dict()
        for key, _ in stats.items():
            meters[key] = AverageMeter()
        return meters

    def inference_one_batch(self, input_dict, phase):
        '''

        :param input_dict:
        :param phase:
        :return:
        '''
        assert phase in ['train', 'val', 'test']
        #############################################
        # training
        if (phase == 'train'):
            self.model.train()
            ###############
            # forward pass
            ###############
            scores, local_scores, local_scores_gt = self.model.forward(input_dict)
            matching_mask = input_dict['matching_mask']
            
            ###############
            # get loss
            ###############

            stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
            c_loss = self.w_matching_loss * stats['matching_loss'] +\
                     self.w_local_matching_loss * stats['local_matching_loss']

            c_loss.backward()
        else:
            self.model.eval()
            with torch.no_grad():
                ###############
                # forward pass
                ###############

                matching_mask = input_dict['matching_mask']

                scores, local_scores, local_scores_gt = self.model.forward(input_dict)

                ###############
                # get loss
                ###############

                stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)

        ######################################
        # detach gradients for loss terms
        ######################################
        stats['matching_loss'] = float(stats['matching_loss'].detach())
        stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
        stats['total_loss'] = float(stats['total_loss'].detach())
        stats['matching_recall'] = stats['matching_recall']
        stats['local_matching_precision'] = stats['local_matching_precision']
        return stats

    def inference_one_epoch(self, epoch, phase):
        '''

        :param epoch:
        :param phase:
        :return:
        '''
        gc.collect()
        assert phase in ['train', 'val', 'test']

        #init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()

        for c_iter in tqdm(range(num_iter)):
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v.to(self.device)

            ####################################
            # forward pass
            ####################################
            stats = self.inference_one_batch(inputs, phase)

            ####################################
            # run optimization
            ####################################
            if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
                self.optimizer.step()
                self.optimizer.zero_grad()

            ####################################
            # update to stats_meter
            ####################################
            for key, value in stats.items():
                stats_meter[key].update(value)


            torch.cuda.empty_cache()

            if self.verbose and (c_iter + 1) % self.verbose_freq == 0:
                curr_iter = num_iter * (epoch - 1) + c_iter
                for key, value in stats_meter.items():
                    self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)

                message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
                for key, value in stats_meter.items():
                    message += f'{key}:{value.avg:.2f}\t'

                self.logger.write(message + '\n')

        message = f'{phase} Epoch: {epoch}'
        for key, value in stats_meter.items():
            message += f'{key}: {value.avg:.4f}\t'

        self.logger.write(message + '\n')

        return stats_meter

    def train(self):
        '''
        :return:
        '''
        print('start training...')
        for epoch in range(self.start_epoch, self.max_epoch):
            self.inference_one_epoch(epoch, 'train')
            self.scheduler.step()

            stats_meter = self.inference_one_epoch(epoch, 'val')

            if stats_meter['total_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['total_loss'].avg
                self._snapshot(epoch, 'best_loss')

            if stats_meter['local_matching_precision'].avg > self.best_local_matching_precision:
                self.best_local_matching_precision = stats_meter['local_matching_precision'].avg
                self._snapshot(epoch, 'best_local_matching_precision')

            if stats_meter['matching_recall'].avg > self.best_matching_recall:
                self.best_matching_recall = stats_meter['matching_recall'].avg
                self._snapshot(epoch, 'best_matching_recall')

        # finish all epoch
        print('training finish!')

    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0, 'val')

        for key, value in stats_meter.items():
            print(key, value.avg)



