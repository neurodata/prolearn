import torch
from torch.utils.data import DataLoader
import numpy as np

def wif(id):
    """
    Used to fix randomization bug for pytorch dataloader + numpy
    Code from https://github.com/pytorch/pytorch/issues/5059
    """
    process_seed = torch.initial_seed()
    # Back out the base_seed so we can use all the bits.
    base_seed = process_seed - id
    ss = np.random.SeedSequence([id, base_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

def get_dataloader(dataset, batchsize, train=True):
    loader_kwargs = {
            'worker_init_fn': wif,
            'pin_memory': True,
            'num_workers': 4,
            'multiprocessing_context':'fork'
    }
    loader = DataLoader(
                dataset, 
                batch_size=batchsize,
                shuffle=train,
                **loader_kwargs
            )
    return loader

def log_exp_config(log, args, params):
    log.info(f'Dataset details:')
    log.info(f'dataset: {args.dataset}')
    log.info(f'task: {args.task}')
    log.info(f'indices file: {args.indices_file}')
    log.info('\n')

    log.info(f'Experiment details:')
    log.info(f'method: {args.method}')
    log.info(f'N: {args.N}')
    log.info(f't: {args.t}')
    log.info(f'T: {args.T}')
    log.info(f'seed: {args.seed}')
    log.info(f'outer reps: {args.outer_reps}')
    log.info(f'inner reps: {args.reps}')
    log.info(f'device: {args.device}')
    log.info('\n')

    log.info(f'Method details:')
    if args.method in ['proformer', 'conv_proformer', 'timecnn', 'timeresnet']:
        log.info(f'{params[args.method]}') 
    log.info('\n')

    log.info(f'Training details:')
    log.info(f'lr: {args.lr}')
    log.info(f'batchsize: {args.batchsize}')
    log.info(f'epochs: {args.epochs}')
    log.info(f'augment: {args.augment}')
    log.info(f'verbose: {args.verbose}')
    log.info('\n')