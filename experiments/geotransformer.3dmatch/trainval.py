import argparse
import os
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from IPython import embed

from geotransformer.engine import Engine
from geotransformer.utils.metrics import Timer, StatisticsDictMeter
from geotransformer.utils.torch_utils import to_cuda, all_reduce_dict

from config import config
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', metavar='N', type=int, default=10, help='iteration steps for logging')
    return parser


def run_one_epoch(
        engine,
        epoch,
        data_loader,
        model,
        evaluator,
        loss_func=None,
        optimizer=None,
        scheduler=None,
        training=True
):
    if training:
        model.train()
    else:
        model.eval()

    if training:
        loss_meter = StatisticsDictMeter()
        loss_meter.register_meter('loss')
        loss_meter.register_meter('c_loss')
        loss_meter.register_meter('f_loss')

    result_meter = StatisticsDictMeter()
    result_meter.register_meter('PIR')
    result_meter.register_meter('IR')
    result_meter.register_meter('RRE')
    result_meter.register_meter('RTE')
    result_meter.register_meter('RR')
    timer = Timer()

    num_iter_per_epoch = len(data_loader)
    for i, data_dict in enumerate(data_loader):
        data_dict = to_cuda(data_dict)

        timer.add_prepare_time()

        if training:
            output_dict = model(data_dict)
            loss_dict = loss_func(output_dict, data_dict)
            result_dict = evaluator(output_dict, data_dict)
        else:
            with torch.no_grad():
                output_dict = model(data_dict)
                result_dict = evaluator(output_dict, data_dict)

        result_dict = {key: value.item() for key, value in result_dict.items()}
        result_meter.update_from_result_dict(result_dict)
        accepted = result_dict['RRE'] < 15. and result_dict['RTE'] < 0.3
        result_meter.update('RR', float(accepted))

        if training:
            loss = loss_dict['loss']

            loss_dict = {key: value.item() for key, value in loss_dict.items()}
            loss_meter.update_from_result_dict(loss_dict)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        timer.add_process_time()

        if (i + 1) % engine.args.steps == 0:
            message = 'Epoch {}/{}, '.format(epoch + 1, config.max_epoch) + \
                      'iter {}/{}, '.format(i + 1, num_iter_per_epoch)
            if training:
                message += 'loss: {:.3f}, '.format(loss_dict['loss']) + \
                           'c_loss: {:.3f}, '.format(loss_dict['c_loss']) + \
                           'f_loss: {:.3f}, '.format(loss_dict['f_loss'])
            message += 'PIR: {:.3f}, '.format(result_dict['PIR']) + \
                       'IR: {:.3f}, '.format(result_dict['IR']) + \
                       'RRE: {:.3f}, '.format(result_dict['RRE']) + \
                       'RTE: {:.3f}, '.format(result_dict['RTE'])
            if training:
                message += 'lr: {:.3e}, '.format(scheduler.get_last_lr()[0])
            message += 'time: {:.3f}s/{:.3f}s'.format(timer.get_prepare_time(), timer.get_process_time())
            if not training:
                message = '[Eval] ' + message
            engine.logger.info(message)

        if training:
            engine.step()
        torch.cuda.empty_cache()

    message = 'Epoch {}, '.format(epoch + 1)
    if training:
        message += '{}, '.format(loss_meter.summary)
    message += '{}'.format(result_meter.summary())
    if not training:
        message = '[Eval] ' + message
    engine.logger.critical(message)

    if training:
        engine.register_state(epoch=epoch)
        snapshot = osp.join(config.snapshot_dir, 'epoch-{}.pth.tar'.format(epoch))
        engine.save_snapshot(snapshot)
        scheduler.step()


def main():
    parser = make_parser()
    log_file = osp.join(config.logs_dir, 'train-{}.log'.format(time.strftime('%Y%m%d-%H%M%S')))
    with Engine(log_file=log_file, default_parser=parser, seed=config.seed) as engine:
        start_time = time.time()
        train_loader, valid_loader, neighborhood_limits = train_valid_data_loader(config)
        loading_time = time.time() - start_time
        message = 'Neighborhood limits: {}.'.format(neighborhood_limits)
        engine.logger.info(message)
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        engine.logger.info(message)

        model = create_model(config).cuda()
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate * engine.world_size,
            weight_decay=config.weight_decay
        )
        loss_func = OverallLoss(config).cuda()
        evaluator = Evaluator(config).cuda()

        engine.register_state(model=model, optimizer=optimizer)

        if engine.args.snapshot is not None:
            engine.load_snapshot(engine.args.snapshot)

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, config.gamma, last_epoch=engine.state.epoch)

        for epoch in range(engine.state.epoch + 1, config.max_epoch):
            run_one_epoch(
                engine, epoch, train_loader, model, evaluator, loss_func=loss_func, optimizer=optimizer,
                scheduler=scheduler, training=True
            )
            run_one_epoch(
                engine, epoch, valid_loader, model, evaluator, training=False
            )


if __name__ == '__main__':
    main()
