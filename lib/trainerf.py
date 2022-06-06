import os, gc
from tqdm import tqdm
from tensorboardX import SummaryWriter
import torch
from lib.utils import AverageMeter, Logger

import logging

class Trainer(object):
    def __init__(self, args):
        self.config = args
        ###########################
        # parameters
        ###########################
        self.verbose = args.verbose              # True
        self.verbose_freq = args.verbose_freq    # 500
        self.start_epoch = 0                        # args.start_epoch
        self.max_epoch = args.max_epoch          # 150
        self.training_max_iter = args.training_max_iter   #3500
        self.val_max_iter = args.val_max_iter          #验证集最大迭代500
        # 三个评价标准
        self.best_loss = 1e5
        self.best_matching_recall = -1e5
        self.best_local_matching_precision = -1e5

        self.save_dir = args.save_dir                # 保存的目录snapshot/tdmatch_enc_dec/checkpoints/
        self.snapshot_dir = args.snapshot_dir        # snapshot/tdmatch_enc_dec

        self.model = args.model.cuda()               # RoughMatchingModel(config)
        self.optimizer = args.optimizer              # ADam
        self.scheduler = args.scheduler              # ExpLR
        self.scheduler_interval = args.scheduler_interval     # 学习率间隔 1
        self.snapshot_interval = args.snapshot_interval       # 1
        self.iter_size = args.iter_size                       # optim iter_size: 4

        self.w_matching_loss = args.w_matching_loss           # 1
        self.w_local_matching_loss = args.w_local_matching_loss    # 1

        self.writer = SummaryWriter(logdir=args.tboard_dir)
        self.logger = Logger(self.snapshot_dir)
        self.logger.write(f'#parameters {sum([x.nelement() for x in self.model.parameters()]) / 1000000.} M\n')

        if args.pretrain != '':               # 初始化默认为''
            self._load_pretrain(args.pretrain)

        # create dataset and dataloader
        self.loader = dict()
        self.loader['train'] = args.train_loader
        self.loader['val'] = args.val_loader
        self.loader['test'] = args.test_loader
        # create evaluation metrics
        self.desc_loss = args.desc_loss

        with open(f'{args.snapshot_dir}/model.log', 'w') as f:
            f.write(str(self.model))

        f.close()

    def train(self):
        print('start training...')
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        fmt = '%(asctime)s-%(levelname)s-%(message)s' #日志输出的格式
        formatter = logging.Formatter(fmt)  #设置格式
        fileHandler = logging.FileHandler(filename='train_log')#未指定handler日志级别，使用logger级别
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        logger.info('Start training...')
        global_epoch = 0
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, self.max_epoch))
            self.inference_one_epoch(epoch, 'train')
#             gc.collect()    #清除内存
#             #init stats meter
#             #stats_meter = self.stats_meter()
#             meters = dict()
#             #stats = self.stats_dict()
#             stats = dict()
#             stats['matching_loss'] = 0.
#             stats['matching_recall'] = 0.
#             stats['local_matching_loss'] = 0.
#             stats['local_matching_precision'] = 0.
#             stats['total_loss'] = 0.
#
#             for key, _ in stats.items():
#                 meters[key] = AverageMeter()
#             stats_meter = meters
#
#             num_iter = int(len(self.loader['train'].dataset) // self.loader['train'].batch_size)
#             c_loader_iter = self.loader['train'].__iter__()
#
#             self.optimizer.zero_grad()      #初始化更新参数
#             for c_iter in tqdm(range(num_iter)):  #在 batch中传入数据
#                 inputs = c_loader_iter.next()
#                 for k, v in inputs.items():
#                     if type(v) == list:
#                         inputs[k] = [item.cuda() for item in v]
#                     else:
#                         inputs[k] = v.cuda()
#                 # forward pass
#                 # stats = self.inference_one_batch(inputs, 'train')
#                 self.model.train()
#
#                     # forward pass
#                 scores, local_scores, local_scores_gt = self.model.forward(inputs)
#                 matching_mask = inputs['matching_mask']
#
#                     # get loss
#                 stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
#                 c_loss = self.w_matching_loss * stats['matching_loss'] + \
#                          self.w_local_matching_loss * stats['local_matching_loss']
#
#                 c_loss.backward()
#                     # detach gradients for loss terms
#                 stats['matching_loss'] = float(stats['matching_loss'].detach())
#                 stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
#                 stats['total_loss'] = float(stats['total_loss'].detach())
#                 stats['matching_recall'] = stats['matching_recall']
#                 stats['local_matching_precision'] = stats['local_matching_precision']
#
#
#                 # run optimization
#                 if (c_iter + 1) % self.iter_size == 0 :
#                     self.optimizer.step()
#                     self.optimizer.zero_grad()
#                 # update to stats_meter
#                 for key, value in stats.items():
#                     stats_meter[key].update(value)
#
#                 torch.cuda.empty_cache()
#                 # if self.verbose and (c_iter + 1) % self.verbose_freq * 20 == 0:
#                 #     curr_iter = num_iter * (epoch - 1) + c_iter
#                 #     for key, value in stats_meter.items():
#                 #         self.writer.add_scalar(f'{"train"}/{key}', value.avg, curr_iter)
#                 #
#                 #     message = f'{"train"} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
#                 #     for key, value in stats_meter.items():
#                 #         message += f'{key}:{value.avg:.2f}\t'
#                 #
#                 #     self.logger.write(message + '\n')
#             message = f'{"train"} Epoch: {epoch}'
#             for key, value in stats_meter.items():
#                 message += f'{key}: {value.avg:.4f}\t'
#
#             self.logger.write(message + '\n')
#
# #2222222222222222222222222222222222
            self.scheduler.step()    #根据梯度更新网络参数

            stats_meter = self.inference_one_epoch(epoch, 'val')
#             phase = 'val'
#             gc.collect()    #清除内存
#             # assert phase in ['train', 'val', 'test']   #选择阶段
#
#             #init stats meter
#             #stats_meter = self.stats_meter()
#             meters = dict()
#             #stats = self.stats_dict()
#             stats = dict()
#             stats['matching_loss'] = 0.
#             stats['matching_recall'] = 0.
#             stats['local_matching_loss'] = 0.
#             stats['local_matching_precision'] = 0.
#             stats['total_loss'] = 0.
#
#             for key, _ in stats.items():
#                 meters[key] = AverageMeter()
#             stats_meter = meters
#
#             num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
#             c_loader_iter = self.loader[phase].__iter__()
#
#             self.optimizer.zero_grad()      #初始化更新参数
#
#             for c_iter in tqdm(range(num_iter)):
#                 inputs = c_loader_iter.next()
#                 for k, v in inputs.items():
#                     if type(v) == list:
#                         inputs[k] = [item.cuda() for item in v]
#                     else:
#                         inputs[k] = v.cuda()
#
#                 ####################################
#                 # forward pass
#                 ####################################
#                 #stats = self.inference_one_batch(inputs, phase)
#                 self.model.eval()
#                 with torch.no_grad():
#                     ###############
#                     # forward pass
#                     ###############
#
#                     matching_mask = input['matching_mask']
#
#                     scores, local_scores, local_scores_gt = self.model.forward(input)
#
#                     ###############
#                     # get loss
#                     ###############
#
#                     stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
#                 stats['matching_loss'] = float(stats['matching_loss'].detach())
#                 stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
#                 stats['total_loss'] = float(stats['total_loss'].detach())
#                 stats['matching_recall'] = stats['matching_recall']
#                 stats['local_matching_precision'] = stats['local_matching_precision']
#
#                 ####################################
#                 # run optimization
#                 ####################################
#                 if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
#                     self.optimizer.step()
#                     self.optimizer.zero_grad()
#
#                 ####################################
#                 # update to stats_meter
#                 ####################################
#                 for key, value in stats.items():
#                     stats_meter[key].update(value)
#
#
#                 torch.cuda.empty_cache()
#
#                 if self.verbose and (c_iter + 1) % self.verbose_freq * 20 == 0:
#                     curr_iter = num_iter * (epoch - 1) + c_iter
#                     for key, value in stats_meter.items():
#                         self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
#
#                     message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
#                     for key, value in stats_meter.items():
#                         message += f'{key}:{value.avg:.2f}\t'
#
#                     self.logger.write(message + '\n')
#
#             message = f'{phase} Epoch: {epoch}'
#             for key, value in stats_meter.items():
#                 message += f'{key}: {value.avg:.4f}\t'
#
#             self.logger.write(message + '\n')
#             logger.info('Train total_loss: %f' % self.best_loss)
# #-------------------------------------------------------------------------------            '''


            if stats_meter['local_matching_precision'].avg > self.best_local_matching_precision:
                self.best_local_matching_precision = stats_meter['local_matching_precision'].avg
                logger.info('Save model_local_matching_precision ...')
                self._snapshot(epoch, 'best_local_matching_precision')
            logger.info('Test local_matching_precision: %f '% (stats_meter['local_matching_precision'].avg))
            logger.info('Best local_matching_precision: %f '% (self.best_local_matching_precision))


            if stats_meter['matching_recall'].avg > self.best_matching_recall:
                self.best_matching_recall = stats_meter['matching_recall'].avg
                logger.info('Save model_matching_recall ...')
                self._snapshot(epoch, 'best_matching_recall')
            logger.info('Test matching_recall: %f '% (stats_meter['matching_recall'].avg))
            logger.info('Best matching_recall: %f '% (self.best_matching_recall))

            if stats_meter['total_loss'].avg < self.best_loss:
                self.best_loss = stats_meter['total_loss'].avg
                logger.info('Save model_best_loss ...')
                self._snapshot(epoch, 'best_loss')
                # state = {
                #     'epoch': epoch,
                #     'state_dict': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                #     'scheduler': self.scheduler.state_dict(),
                #     'best_loss': self.best_loss,
                #     'best_matching_recall': self.best_matching_recall,
                #     'best_local_matching_precision': self.best_local_matching_precision
                # }
                # name = 'best_loss'
                # # if name is None:
                # #     filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
                # # else:
                # #     filename = os.path.join(self.save_dir, f'model_{name}.pth')
                # filename = os.path.join(self.save_dir, f'model_{epoch}_{name}.pth')

                # print(f'Save model to {filename}')
                # self.logger.write(f'Save model to {filename}\n')
                # torch.save(state, filename)
            logger.info('Test total_loss: %f '% (stats_meter['total_loss'].avg))
            logger.info('Best total_loss: %f '% (self.best_loss))

            if stats_meter['total_loss'].avg > self.best_loss and stats_meter['matching_recall'].avg < self.best_matching_recall and stats_meter['local_matching_precision'].avg < self.best_local_matching_precision :

                logger.info('Save model usual ...')
                self._snapshot(epoch, 'usual')
            logger.info('No best loss ')


            global_epoch += 1
        # finish all epoch
        print('training finish!')
        logger.info('End of training...')


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
            print(f'=> no checkpoint found at {resume}')
            #raise ValueError(f'=> no checkpoint found at {resume}')

    def _snapshot(self, epoch, name=None):
        '''
        Save a trained model
        :param epoch: index of epoch of current model
        :param name: path to the saving model
        :return: None
        '''
        state = {'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'best_matching_recall': self.best_matching_recall,
            'best_local_matching_precision': self.best_local_matching_precision
        }
        # if name is None:
        #     filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        # else:
        #     filename = os.path.join(self.save_dir, f'model_{name}.pth')
        Name = name
        filename = os.path.join(self.save_dir, f'model_{epoch}_{Name}.pth')

        print(f'Save model to {filename}')
        self.logger.write(f'Save model to {filename}\n')
        torch.save(state, filename)

    def eval(self):
        print('Start to evaluate on validation datasets...')
        stats_meter = self.inference_one_epoch(0, 'val')

        for key, value in stats_meter.items():
            print(key, value.avg)

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
            c_loss = self.w_matching_loss * stats['matching_loss'] + \
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
        gc.collect()    #清除内存
        assert phase in ['train', 'val', 'test']   #选择阶段

        #init stats meter
        stats_meter = self.stats_meter()

        num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
        c_loader_iter = self.loader[phase].__iter__()

        self.optimizer.zero_grad()      #初始化更新参数

        for c_iter in tqdm(range(num_iter)):
            inputs = c_loader_iter.next()
            for k, v in inputs.items():
                if type(v) == list:
                    inputs[k] = [item.cuda() for item in v]
                else:
                    inputs[k] = v.cuda()

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

            if self.verbose and (c_iter + 1) % self.verbose_freq  == 0:
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

    # def stats_dict(self):
    #     '''
    #     Create the dict of all metrics
    #     :return: stats: the dict containing all metrics
    #     '''
    #     stats = dict()
    #     stats['matching_loss'] = 0.
    #     stats['matching_recall'] = 0.
    #     stats['local_matching_loss'] = 0.
    #     stats['local_matching_precision'] = 0.
    #     stats['total_loss'] = 0.
    #     '''
    #     to be added
    #     '''
    #     return stats
    #
    # def stats_meter(self):
    #     '''
    #     For each metric in stats dict, create an AverageMeter() for update
    #     :return: meters: dict of AverageMeter()
    #     '''
    #     meters = dict()
    #     stats = self.stats_dict()
    #     for key, _ in stats.items():
    #         meters[key] = AverageMeter()
    #     return meters

    # def inference_one_batch(self, input_dict, phase):
    #     '''
    #
    #     :param input_dict:
    #     :param phase:
    #     :return:
    #     '''
    #     assert phase in ['train', 'val', 'test']
    #     #############################################
    #     # training
    #     if (phase == 'train'):
    #         self.model.train()
    #         ###############
    #         # forward pass
    #         ###############
    #         scores, local_scores, local_scores_gt = self.model.forward(input_dict)
    #         matching_mask = input_dict['matching_mask']
    #
    #         ###############
    #         # get loss
    #         ###############
    #
    #         stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
    #         c_loss = self.w_matching_loss * stats['matching_loss'] + \
    #                  self.w_local_matching_loss * stats['local_matching_loss']
    #
    #         c_loss.backward()
    #     else:
    #         self.model.eval()
    #         with torch.no_grad():
    #             ###############
    #             # forward pass
    #             ###############
    #
    #             matching_mask = input_dict['matching_mask']
    #
    #             scores, local_scores, local_scores_gt = self.model.forward(input_dict)
    #
    #             ###############
    #             # get loss
    #             ###############
    #
    #             stats = self.desc_loss(scores[0], matching_mask, local_scores, local_scores_gt)
    #
    #     ######################################
    #     # detach gradients for loss terms
    #     ######################################
    #     stats['matching_loss'] = float(stats['matching_loss'].detach())
    #     stats['local_matching_loss'] = float(stats['local_matching_loss'].detach())
    #     stats['total_loss'] = float(stats['total_loss'].detach())
    #     stats['matching_recall'] = stats['matching_recall']
    #     stats['local_matching_precision'] = stats['local_matching_precision']
    #     return stats
    # def inference_one_epoch(self, epoch, phase):
    #     '''
    #
    #     :param epoch:
    #     :param phase:
    #     :return:
    #     '''
    #     gc.collect()    #清除内存
    #     assert phase in ['train', 'val', 'test']   #选择阶段
    #
    #     #init stats meter
    #     stats_meter = self.stats_meter()
    #
    #     num_iter = int(len(self.loader[phase].dataset) // self.loader[phase].batch_size)
    #     c_loader_iter = self.loader[phase].__iter__()
    #
    #     self.optimizer.zero_grad()      #初始化更新参数
    #
    #     for c_iter in tqdm(range(num_iter)):
    #         inputs = c_loader_iter.next()
    #         for k, v in inputs.items():
    #             if type(v) == list:
    #                 inputs[k] = [item.cuda() for item in v]
    #             else:
    #                 inputs[k] = v.cuda()
    #
    #         ####################################
    #         # forward pass
    #         ####################################
    #         stats = self.inference_one_batch(inputs, phase)
    #
    #         ####################################
    #         # run optimization
    #         ####################################
    #         if (c_iter + 1) % self.iter_size == 0 and phase == 'train':
    #             self.optimizer.step()
    #             self.optimizer.zero_grad()
    #
    #         ####################################
    #         # update to stats_meter
    #         ####################################
    #         for key, value in stats.items():
    #             stats_meter[key].update(value)
    #
    #
    #         torch.cuda.empty_cache()
    #
    #         if self.verbose and (c_iter + 1) % self.verbose_freq * 20 == 0:
    #             curr_iter = num_iter * (epoch - 1) + c_iter
    #             for key, value in stats_meter.items():
    #                 self.writer.add_scalar(f'{phase}/{key}', value.avg, curr_iter)
    #
    #             message = f'{phase} Epoch: {epoch} [{c_iter + 1:4d}/{num_iter}]'
    #             for key, value in stats_meter.items():
    #                 message += f'{key}:{value.avg:.2f}\t'
    #
    #             self.logger.write(message + '\n')
    #
    #     message = f'{phase} Epoch: {epoch}'
    #     for key, value in stats_meter.items():
    #         message += f'{key}: {value.avg:.4f}\t'
    #
    #     self.logger.write(message + '\n')
    #
    #     return stats_meter