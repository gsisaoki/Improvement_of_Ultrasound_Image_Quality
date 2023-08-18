import os
import argparse
import torch
import numpy as np
import yaml
import shutil
import datetime
import random
import pandas as pd

from utils import *
from lib.lr_scheduler import *

def load_yaml(root_path, yaml_file):
    """Load yaml file
    Args:
        root_path (str): The path of root dir
        yaml_file (str): The name of yaml file
    Return:
        dic: Setting arguments
    """
    with open(os.path.join(root_path, 'src', 'yaml', yaml_file), 'r') as stream:
        cfgs = yaml.load(stream, Loader=yaml.SafeLoader)
    return cfgs

def save_yaml(cfgs):
    """Save yaml file
    Args:
        cfgs (dic)     : Setting argments
        root_path (str): The path of root dir
    """
    root_path = cfgs['root_path']
    exp_dir = os.path.join(cfgs['target'], cfgs['exp_name'])
    with open(os.path.join(root_path, 'yaml', exp_dir, 'log.yaml'), 'w') as fp:
        yaml.dump(cfgs, fp, default_flow_style=False)

def fix_seed(seed):
    """Fix the seed
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    """
    Note:
        https://pytorch.org/docs/master/notes/randomness.html
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_expname(name):
    """Get the diretory's name to save
    Args:
        name (str): Outline of the experiment
    Returns:
        str: date_name
    """
    return str(datetime.date.today()) + '_' + name

def prepare_exp_dir(cfgs):
    """Prepare dir
    Args:
        cfgs (dic): Setting argments
    """
    root_path = cfgs['root_path']
    exp_dir = os.path.join(cfgs['target'], cfgs['exp_name'])
    if cfgs['mode'] == 'train':
        save_dirs = ['model', 'visualize/train', 'visualize/val', 'visualize/test', 'log', 'yaml', 'result']
    elif cfgs['mode'] == 'test':
        save_dirs = ['result', 'yaml']

    if os.path.exists(os.path.join(root_path, save_dirs[0], exp_dir)):
        is_ok = input('{} will be deleted. If OK, please enter y.\n'.format(os.path.join(root_path, save_dirs[0], exp_dir)))
        if is_ok == 'y':
            for save_dir in save_dirs:
                shutil.rmtree(os.path.join(root_path, save_dir, exp_dir), ignore_errors=True)

    for save_dir in save_dirs:
        os.makedirs(os.path.join(root_path, save_dir, exp_dir))

def load_checkpoint(checkpoint_path, device):
    """Load checkpoint file
    Args:
        checkpoint_path (str): The path of checkpoint file
        device (obj)         : The device id
    Returns:
        dic: checkpoint state
    """
    print('Loading {} checkpoint model'.format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint

def save_checkpoint(cfgs, model, optimizer, epoch):
    """Save checkpoint file
    Args:
        cfgs (dic)     : Setting argments
        model (dic)    : model
        optimizer (dic): optimizer
        epoch (int)    : epoch
    """
    # save_path = os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'], 'epoch' + return_zero_fill(epoch) + '.pth.tar')
    save_path = os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'], 'best.pth.tar')
    torch.save({
        'epoch' : epoch,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict()}, save_path)

def save_gan_checkpoint(cfgs, netG, netD, optimizerG, optimizerD, epoch):
    """Save checkpoint file
    Args:
        cfgs (dic)     : Setting argments
        model (dic)    : model
        optimizer (dic): optimizer
        epoch (int)    : epoch
    """
    save_path = os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'], 'epoch' + return_zero_fill(epoch) + '.pth.tar')
    torch.save({
        'epoch' : epoch,
        'g_state_dict' : netG.state_dict(),
        'd_state_dict' : netD.state_dict(),
        'optimizerG' : optimizerG.state_dict(),
        'optimizerD' : optimizerD.state_dict()}, save_path)

def transform_tensor2numpy(input):
    """transform tensor to numpy
    Args:
        input (float): torch.cuda
    Returns:
        float: numpy
    """
    return input.detach().cpu().numpy()

def select_optimizer(model, cfgs):
    """Choose optimizer to use
    Args:
        model (dic) : CNN model
        cfgs (dic)  : Setting argments
    Returns:
        dir: optimizer
    """
    if cfgs['train']['optimizer'] == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=cfgs['train']['lr'], weight_decay=cfgs['train']['weight_decay'])
    elif cfgs['train']['optimizer'] == 'nag':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfgs['train']['lr'], momentum=0.7, nesterov=True)
    elif cfgs['train']['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfgs['train']['lr'], momentum=0.9)
    elif cfgs['train']['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfgs['train']['lr'], weight_decay=cfgs['train']['weight_decay'])
        # optimizer = torch.optim.Adam(model.parameters())
    elif cfgs['train']['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=cfgs['train']['lr'], weight_decay=cfgs['train']['weight_decay'])
    elif cfgs['train']['optimizer'] == 'sam':
        from models.sam import SAM
        optimizer = SAM(model.parameters(), torch.optim.Adam, lr=cfgs['train']['lr'], weight_decay=cfgs['train']['weight_decay'])
    elif cfgs['train']['optimizer'] == 'nadam':
        optimizer = torch.optim.NAdam(model.parameters(), lr=cfgs['train']['lr'])
    elif cfgs['train']['optimizer'] == 'radam':
        optimizer = torch.optim.RAdam(model.parameters(), lr=cfgs['train']['lr'])
    elif cfgs['train']['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfgs['train']['lr'])

    return optimizer

def select_scheduler(optimizer, cfgs):
    """Choose scheduler to use
    Args:
        optimizer (dic): optimizer
        cfgs (dic)     : Setting argments
    Returns:
        obj: scheduler
    """
    if cfgs['train']['scheduler'] == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
    elif cfgs['train']['scheduler'] == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,20], gamma=0.1)
    elif cfgs['train']['scheduler'] == 'labmdalr':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.1 ** epoch)
    elif cfgs['train']['scheduler'] == 'explr':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.80)
    elif cfgs['train']['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, verbose=True)
    elif cfgs['train']['scheduler'] == 'poly':
        scheduler = PolyScheduler(optimizer, base_lr=cfgs['train']['lr'], max_steps=2800 // cfgs['train']['batch_size'] * cfgs['train']['epochs'], warmup_steps=0, last_epoch=-1)
    return scheduler

# Average Meter
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        #fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} [{avg' + self.fmt + '}]'
        return fmtstr.format(**self.__dict__)

# Progress Meter
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

#Print Model
def print_model(x, model):
    '''torchsummaryでモデルをプリントする関数
    x : (N,C,H,W)
        '''
    summary(model, x.shape[1:], device='cpu')

# Pandas関連
def load_csv(csv_path, header=None, sep=','):
    data = pd.read_csv(csv_path, header=header, sep=sep).values
    return data

def make_df(cols):
    df = pd.DataFrame(index=[], columns=cols)
    return df

def write_df(df, *data):
    data = list(data)
    record = pd.Series(data, index=df.columns)
    df = df.append(record, ignore_index=True)
    return df

def save_df(save_path, df):
    df.to_csv(save_path)
