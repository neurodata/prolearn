'''
Indepedent (case 2) Exp
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

@hydra.main(config_path=".", config_name="config_cifar10")
def main(cfg):
    cwd = pathlib.Path(get_original_cwd())

    # input parameters
    params = {
        # dataset
        "dataset": cfg.dataset,
        "task": cfg.task,    # task specification
        "indices_file": cfg.indices_file, # 'mnist_00-51-47', 'cifar-10_02-12-13'

        # experiment
        "method": cfg.method,         # select from {proformer, cnn, mlp, timecnn}
        "N": 10,                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": 5000,                   # future time horizon
        "seed": 1996,   
        "device": cfg.device,          # device
        "reps": 100,                 # number of test reps
        "outer_reps": 3,         
              
        # training params
        "lr": 1e-3,         
        "batchsize": cfg.batchsize,
        "epochs": cfg.epochs,
        "augment": cfg.augment,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

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

    # load the saved indicies
    indices_file = cwd / f'indices/{args.indices_file}.pkl'
    with open(indices_file, 'rb') as f:
        total_indices = pickle.load(f)

    # get full task pattern
    full_pattern = total_indices['full_pattern']

    risk_list = []
    for outer_rep in range(args.outer_reps):
        log.info(" ")
        
        # get a training sequence
        train_SeqInd = total_indices[args.t][outer_rep]['train']

        # sample a bunch of test sequences
        test_seqInds = total_indices[args.t][outer_rep]['test']

        # get the module for the specified method
        method, datahandler = get_modules(args.method)

        # form the train dataset
        if args.t > 0:
            train_dataset_list = []
            pattern = full_pattern[:args.t]
            for i, task in enumerate(args.task):
                data_kwargs = {
                    "dataset": torch_dataset, 
                    "transform": train_transform,
                    "seqInd": train_SeqInd[pattern == i], 
                    "maplab": maplab
                }
                train_dataset = datahandler.VisionSequentialDataset(args, **data_kwargs)
                train_dataset_list.append(train_dataset)
        else:
            train_dataset_list = [[] for _ in args.task] 

        # model
        model_list = []
        for i, task in enumerate(args.task):
            model_kwargs = method.model_defaults(args.dataset)
            log.info(f'{model_kwargs}')
            model = method.Model(
                num_classes=len(task),
                **model_kwargs
            )
            model_list.append(model)
        
        # train
        trainer_list = [
            method.Trainer(model, train_dataset, args) 
            for model, train_dataset in zip(model_list, train_dataset_list)
        ]
        if args.t > 0:
            for i, trainer in enumerate(trainer_list):
                log.info(f'training an individual model for task {i}...')
                trainer.fit(log)

        for task_id, trainer in enumerate(trainer_list):
            log.info(f'model {task_id} trained? {trainer.istrained}')

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

            pattern = full_pattern[args.t:]
            preds_rep = np.zeros(args.T - args.t)
            for task_id, trainer in enumerate(trainer_list):
                pred, truths_rep = trainer.evaluate(test_dataset)
                preds_rep[pattern == task_id] = pred[pattern == task_id]

            preds.append(preds_rep)
            truths.append(truths_rep)
        preds = np.array(preds)
        truths = np.array(truths)

        # compute metrics
        instantaneous_risk = np.mean(preds != truths, axis=0).squeeze()
        std_error = np.std(preds != truths, axis=0).squeeze()
        ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

        time_averaged_risk = np.mean(preds != truths)
        log.info(f"error = {time_averaged_risk:.4f}")
        risk_list.append(time_averaged_risk)

    risk = np.mean(risk_list)
    ci_risk = np.std(risk_list) * 1.96/np.sqrt(args.outer_reps).squeeze()
    log.info(f"risk at t = {args.t} : {risk:.4f}")
    
    outputs = {
        "args": params,
        "risk": risk,
        "ci_risk": ci_risk, 
        "inst_risk": instantaneous_risk,
        "ci": ci
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)

    # save last model
    torch.save(trainer.model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()