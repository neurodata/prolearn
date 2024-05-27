'''
Vision covariate shift exps
'''
import importlib
import torch
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
from hydra.utils import get_original_cwd, to_absolute_path
import logging
import math

from prol.process import (
    get_torch_dataset,
    get_multi_indices_and_map,
    get_multi_cycle
)

import pathlib

class SetParams:
    def __init__(self, dict) -> None:
        for k, v in dict.items():
            setattr(self, k, v)

def get_modules(name):
    try: 
        module1 = importlib.import_module(f"prol.models.{name}")
        module2 = importlib.import_module(f"prol.datahandlers.{name}_handle")
    except ImportError:
        print(f"Module {name} not found")
    return module1, module2

log = logging.getLogger(__name__)

@hydra.main(config_path=".", config_name="config_mnist")
def main(cfg):
    cwd = pathlib.Path(get_original_cwd())

    # load the saved indicies
    indices_file = cwd / f'indices/{cfg.indices_file}.pkl'
    with open(indices_file, 'rb') as f:
        total_indices = pickle.load(f)

    # input parameters
    params = {
        # dataset
        "dataset": total_indices["dataset"],
        "task": total_indices["task"],    # task specification
        "indices_file": cfg.indices_file, # 'mnist_00-51-47', 'cifar-10_02-12-13'

        # experiment
        "method": cfg.method,         # select from {proformer, cnn, mlp, timecnn}
        "t": 200,
        "N": total_indices["N"],                     # time between two task switches                   
        "T": total_indices["T"],                   # future time horizon
        "seed": 1996,   
        "device": cfg.device,          # device
        "reps": total_indices["inner_reps"],                 # number of test reps
        "outer_reps": total_indices["outer_reps"],         
              
        # training params
        "lr": 1e-3,    
        "ft_lr": 5e-4,     
        "batchsize": cfg.batchsize,
        "epochs": cfg.epochs,
        "ft_epochs": 50, # 50 for MNIST, 100 for CIFAR
        "augment": cfg.augment,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

    # max number of classes
    max_num_classes = max([len(task) for task in args.task])

    # get source dataset
    root = '/home/ubuntu/ProL/data'
    torch_dataset, augment_transform, vanilla_transform = get_torch_dataset(root, name=args.dataset)
    test_transform = vanilla_transform
    if args.augment: 
        train_transform = augment_transform
    else:
        train_transform = vanilla_transform
    
    # get indices for each task
    _, mapdict, torch_dataset = get_multi_indices_and_map(
        tasks=args.task,
        dataset=torch_dataset
    )
    maplab = lambda lab : mapdict[lab]

    # unit = get_multi_cycle(args.N, len(args.task))
    # full_pattern = np.array((unit * math.ceil(args.T/(len(unit))))[:args.T]).astype("int")

    # get full task pattern
    full_pattern_list = total_indices['full_pattern']

    # get the module for the specified method
    method, datahandler = get_modules(args.method)

    t_list = [0,200,500,700,1000,1200,1500,1700,2000,2500,3000,4000]
    risk_list = [[] for _ in range(args.outer_reps)]

    raw_metrics = {}
    for outer_rep in range(args.outer_reps):
        log.info(f"rep : {outer_rep}")

        model_kwargs = method.model_defaults(args.dataset)
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=max_num_classes,
            **model_kwargs
        )

        prev_t = 0

        for t in t_list:
            log.info(f"time : {t}")

            args.t = t
            # get a training sequence
            if prev_t == 0 and t == 0:
                train_SeqInd = []
                train_dataset = []
            elif prev_t == 0 and t > 0:
                # use all the data to train from scratch
                train_SeqInd = total_indices[t][outer_rep]['train']
            else:
                # use only the newly available data to finetune
                train_SeqInd = total_indices[t][outer_rep]['train'][:prev_t]

            # sample a bunch of test sequences
            test_seqInds = total_indices[t][outer_rep]['test']

            # form the training dataset
            if t > 0:
                data_kwargs = {
                        "dataset": torch_dataset, 
                        "transform": train_transform,
                        "seqInd": train_SeqInd, 
                        "maplab": maplab
                    }
                train_dataset = datahandler.VisionSequentialDataset(args, **data_kwargs)

                if prev_t == 0:
                    # train from scratch
                    trainer = method.Trainer(model, train_dataset, args)
                    trainer.fit(log)
                    model = trainer.model
                else:
                    # finetune
                    args.lr = args.ft_lr
                    args.epochs = args.ft_epochs
                    log.info(f'Finetuning with lr = {args.lr} for {args.epochs} epochs..')
                    trainer = method.Trainer(model, train_dataset, args)
                    trainer.fit(log)
                    model = trainer.model

                prev_t = t
            else:
                trainer = method.Trainer(model, train_dataset, args)

            # evaluate
            preds = []
            truths = []
            for i in tqdm(range(args.reps)):
                # form a test dataset for each test sequence
                test_kwargs = {
                    "dataset": torch_dataset, 
                    "transform": test_transform,
                    "train_seqInd": train_SeqInd, 
                    "test_seqInd": test_seqInds[i], 
                    "maplab": maplab
                }
                test_dataset = datahandler.VisionSequentialTestDataset(args, **test_kwargs)
                preds_rep, truths_rep = trainer.evaluate(test_dataset)
                preds.append(preds_rep)
                truths.append(truths_rep)
            preds = np.array(preds)
            truths = np.array(truths)

            # compute metrics
            # instantaneous_risk = np.mean(preds != truths, axis=0).squeeze()
            # std_error = np.std(preds != truths, axis=0).squeeze()
            # ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

            time_averaged_risk = np.mean(preds != truths)
            log.info(f"error = {time_averaged_risk:.4f}")
            risk_list[outer_rep].append(time_averaged_risk)

    risks = np.mean(risk_list, axis=0)
    ci_risks = np.std(risk_list, axis=0) * 1.96/np.sqrt(args.outer_reps).squeeze()
    log.info(f"time : {t_list}")
    log.info(f"risks : {risks}")
    
    outputs = {
        "t_list": t_list,
        "args": params,
        "risk": risks,
        "ci_risk": ci_risks
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()