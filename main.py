import os
import argparse
from tkinter import Y
import torch
from torch.utils.data import DataLoader
import traceback
from solver import Solver
from dataloader import *
from cnn_lib import *
from tuner import Tuner

def main(cfgs):
    train_scan, val_scan, test_scan = cfgs['breast_scan']['train'], cfgs['breast_scan']['val'], cfgs['breast_scan']['test']
    scan_num = train_scan + val_scan + test_scan
    scan_list = random.sample(list(range(scan_num)), scan_num)
    train_list, val_list, test_list = scan_list[:train_scan], scan_list[train_scan:train_scan+val_scan], scan_list[train_scan+val_scan:]
    cfgs['train_list'], cfgs['val_list'], cfgs['test_list'] = train_list, val_list, test_list
    

    if args.mode == 'train':
        train_dataset = UltrasoundDataset('train', cfgs)
        val_dataset = UltrasoundDataset('val', cfgs)
        breast_test_dataset = UltrasoundDataset('test', cfgs, 'breast')
        eval_test_dataset = UltrasoundDataset('test', cfgs, 'eval')
        invivo_test_dataset = UltrasoundDataset('test', cfgs, 'invivo')
        print('Training   Data:', len(train_dataset))
        print('Validation Data:', len(val_dataset))
        print('Evaluation Data:', len(breast_test_dataset) + len(eval_test_dataset) + len(invivo_test_dataset))
        trainloader = DataLoader(train_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=True, drop_last=True, worker_init_fn=worker_init_fn)
        valloader = DataLoader(val_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        breast_testloader = DataLoader(breast_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        eval_testloader = DataLoader(eval_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        invivo_testloader = DataLoader(invivo_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        if cfgs['train']['tuning']:
            tuner = Tuner(cfgs)
            tuner.main(trainloader, valloader)
        else:
            solver = Solver(cfgs)
            solver.train(trainloader, valloader)
            cfgs['checkpoint_path'] = sorted(glob.glob(os.path.join(cfgs['root_path'], 'model', cfgs['target'], cfgs['exp_name'], '*')))[-1]
            solver = Solver(cfgs)
            solver.test('test', 'breast', breast_testloader)
            solver.test('test', 'eval', eval_testloader)
            solver.test('test', 'invivo', invivo_testloader)
            #PICMUS
            PICMUS_dataset = PICMUSDataset(cfgs)
            print('PICMUS Data: ', len(PICMUS_dataset))
            PICMUSloader = DataLoader(PICMUS_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
            solver.test('PICMUS', 'PICMUS', PICMUSloader)
    elif args.mode == 'test':
        breast_test_dataset = UltrasoundDataset('test', cfgs, 'breast')
        eval_test_dataset = UltrasoundDataset('test', cfgs, 'eval')
        invivo_test_dataset = UltrasoundDataset('test', cfgs, 'invivo')
        PICMUS_dataset = PICMUSDataset(cfgs)
        print('Evaluation Data: ', len(breast_test_dataset) + len(eval_test_dataset) + len(invivo_test_dataset))
        print('PICMUS Data: ', len(PICMUS_dataset))
        breast_testloader = DataLoader(breast_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        eval_testloader = DataLoader(eval_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        invivo_testloader = DataLoader(invivo_test_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        PICMUSloader = DataLoader(PICMUS_dataset, batch_size=cfgs['train']['batch_size'], num_workers=cfgs['num_workers'], shuffle=False)
        if not cfgs['checkpoint_path']:
            cfgs['checkpoint_path'] = sorted(glob.glob(os.path.join(cfgs['root_path'], 'model', cfgs['target'], '*_' + cfgs['exp_name'][11:], '*')))[-1]
        solver = Solver(cfgs)
        solver.test('test', 'breast', breast_testloader)
        solver.test('test', 'eval', eval_testloader)
        solver.test('test', 'invivo', invivo_testloader)
        solver.test('PICMUS', 'PICMUS', PICMUSloader)
    return cfgs

def get_arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', '-d', type=str, default='0')
    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--num_workers', '-w', type=int, default=1)
    parser.add_argument('--exp', '-e', type=str)
    parser.add_argument('--yaml', '-y', type=str, default='train.yaml')
    parser.add_argument('--target', '-t', type=str, default='debug')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    root_path = '/home/jaxa/shidara/PWI/'
    args = get_arg_parse()
    exp_name = get_expname(args.exp)

    # Load yaml file
    cfgs = load_yaml(root_path, args.yaml)

    torch.cuda.set_device(int(args.device))
    cfgs['root_path'] = root_path
    cfgs['device'] = torch.device('cuda:' + args.device)
    cfgs['mode'] = args.mode
    cfgs['num_workers'] = args.num_workers
    cfgs['target'] = args.target
    cfgs['exp_name'] = exp_name
    cfgs['root_path'] = root_path

    # Fix the experimental seed
    fix_seed(cfgs['seed'])

    # Prepare dir about experiments
    prepare_exp_dir(cfgs)
    
    print(cfgs['train']['beta_Envelope'])
    print(cfgs['train']['beta_Fourier'])
    print(cfgs['train']['alpha_amp'])
    print(cfgs['train']['alpha_phase'])

    cfgs = main(cfgs)
    save_yaml(cfgs)