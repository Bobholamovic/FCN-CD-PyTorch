#!/usr/bin/env python3
import argparse
import os
import shutil
import random
import ast
from os.path import basename, exists, splitext

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import yaml

from core.trainers import CDTrainer
from utils.misc import OutPathGetter, Logger, register


def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    return cfg or {}


def parse_config(cfg_name, cfg):
    # Parse the name of config file
    sp = splitext(cfg_name)[0].split('_')
    if len(sp) >= 2:
        cfg.setdefault('tag', sp[1])
        cfg.setdefault('suffix', '_'.join(sp[2:]))
    
    return cfg


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('cmd', choices=['train', 'val'])

    # Data
    # Common
    group_data = parser.add_argument_group('data')
    group_data.add_argument('-d', '--dataset', type=str, default='OSCD')
    group_data.add_argument('-p', '--crop-size', type=int, default=256, metavar='P', 
                        help='patch size (default: %(default)s)')
    group_data.add_argument('--num-workers', type=int, default=8)
    group_data.add_argument('--repeats', type=int, default=100)

    # Optimizer
    group_optim = parser.add_argument_group('optimizer')
    group_optim.add_argument('--optimizer', type=str, default='Adam')
    group_optim.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: %(default)s)')
    group_optim.add_argument('--lr-mode', type=str, default='const')
    group_optim.add_argument('--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: %(default)s)')
    group_optim.add_argument('--step', type=int, default=200)

    # Training related
    group_train = parser.add_argument_group('training related')
    group_train.add_argument('--batch-size', type=int, default=8, metavar='B',
                        help='input batch size for training (default: %(default)s)')
    group_train.add_argument('--num-epochs', type=int, default=1000, metavar='NE',
                        help='number of epochs to train (default: %(default)s)')
    group_train.add_argument('--load-optim', action='store_true')
    group_train.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint')
    group_train.add_argument('--anew', action='store_true',
                        help='clear history and start from epoch 0 with the checkpoint loaded')
    group_train.add_argument('--trace-freq', type=int, default=50)
    group_train.add_argument('--device', type=str, default='cpu')
    group_train.add_argument('--metrics', type=str, default='F1Score+Accuracy+Recall+Precision')

    # Experiment
    group_exp = parser.add_argument_group('experiment related')
    group_exp.add_argument('--exp-dir', default='../exp/')
    group_exp.add_argument('-o', '--out-dir', default='')
    group_exp.add_argument('--tag', type=str, default='')
    group_exp.add_argument('--suffix', type=str, default='')
    group_exp.add_argument('--exp-config', type=str, default='')
    group_exp.add_argument('--save-on', action='store_true')
    group_exp.add_argument('--log-off', action='store_true')
    group_exp.add_argument('--suffix-off', action='store_true')

    # Criterion
    group_critn = parser.add_argument_group('criterion related')
    group_critn.add_argument('--criterion', type=str, default='NLL')
    group_critn.add_argument('--weights', type=str, default=(1.0, 1.0))

    # Model
    group_model = parser.add_argument_group('model')
    group_model.add_argument('--model', type=str, default='siamunet_conc')
    group_model.add_argument('--num-feats-in', type=int, default=13)

    args = parser.parse_args()

    if exists(args.exp_config):
        cfg = read_config(args.exp_config)
        cfg = parse_config(basename(args.exp_config), cfg)
        # Settings from cfg file overwrite those in args
        # Note that the non-default values will not be affected
        parser.set_defaults(**cfg)  # Reset part of the default values
        args = parser.parse_args()  # Parse again

    # Handle args.weights
    if isinstance(args.weights, str):
        args.weights = ast.literal_eval(args.weights)
    args.weights = tuple(args.weights)

    return args


def set_gpc_and_logger(args):
    gpc = OutPathGetter(
            root=os.path.join(args.exp_dir, args.tag), 
            suffix=args.suffix)

    log_dir = '' if args.log_off else gpc.get_dir('log')
    logger = Logger(
        scrn=True,
        log_dir=log_dir,
        phase=args.cmd
    )

    register('GPC', gpc)
    register('LOGGER', logger)

    return gpc, logger
    

def main():
    args = parse_args()
    gpc, logger = set_gpc_and_logger(args)

    if exists(args.exp_config):
        # Make a copy of the config file
        cfg_path = gpc.get_path('root', basename(args.exp_config), suffix=False)
        shutil.copy(args.exp_config, cfg_path)

    # Set random seed
    RNG_SEED = 1
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)

    cudnn.deterministic = True
    cudnn.benchmark = False

    try:
        trainer = CDTrainer(args.model, args.dataset, args.optimizer, args)
        if args.cmd == 'train':
            trainer.train()
        elif args.cmd == 'val':
            trainer.validate()
        else:
            pass
    except BaseException as e:
        import traceback
        # Catch ALL kinds of exceptions
        logger.error(traceback.format_exc())
        exit(1)

if __name__ == '__main__':
    main()