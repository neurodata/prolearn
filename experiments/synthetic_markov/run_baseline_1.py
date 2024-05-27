'''
Indepedent (case 2) Exp
'''
import importlib
import torch
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
from hydra.utils import get_original_cwd
import logging

from prol.process import (
    get_synthetic_data,
    get_cycle
)
import math
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

@hydra.main(config_path=".", config_name="config")
def main(cfg):

    # input parameters
    params = {
        # experiment params
        "dataset": "synthetic",
        "method": cfg.method,
        "N": 20,                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": 5000,                   # future time horizon
        "seed": 1996,
        "device": cfg.device,
        "reps": 100,                 # number of test reps
        "outer_reps": 3,

        # training params             
        "lr": 1e-3,         
        "batchsize": cfg.batchsize,
        "epochs": cfg.epochs,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

    # get the task patterns fromt the Markov chain
    cwd = pathlib.Path(get_original_cwd()) 
    fname = cwd / 'synthetic.pkl'
    with open(fname, 'rb') as f:
        saved = pickle.load(f)
    full_pattern_list = saved['pattern']

    risk_list = []
    raw_metrics = {
        "t": args.t,
        "preds": [],
        "truths": []
    }
    for outer_rep in range(args.outer_reps):
        log.info(" ")

        full_pattern = full_pattern_list[outer_rep]
        
        # get a training sequence
        seed = args.seed * outer_rep * 2357
        if args.t > 0:
            x_train, y_train = get_synthetic_data(
                N=args.N,
                total_time_steps=args.t,
                seed=seed,
                markov=True,
                pattern=full_pattern
            )
        else:
            x_train, y_train = [], []

        # sample a bunch of test sequences
        test_data = [
            get_synthetic_data(
                args.N, 
                args.T, 
                seed=seed+1000*(inner_rep+1),
                markov=True,
                pattern=full_pattern
            )
            for inner_rep in range(args.reps)
        ]

        # get the module for the specified method
        method, datahandler = get_modules(args.method)

        # form the train dataset
        if args.t > 0:
            train_dataset_list = []
            pattern = full_pattern[:args.t]
            for i in range(2):
                train_dataset = datahandler.SyntheticSequentialDataset(
                    args, 
                    x_train[pattern==i], 
                    y_train[pattern==i]
                )
                train_dataset_list.append(train_dataset)
        else:
            train_dataset_list = [[] for _ in range(2)] 

        # model
        model_list = []
        for i in range(2):
            model_kwargs = method.model_defaults(args.dataset)
            log.info(f'{model_kwargs}')
            model = method.Model(
                num_classes=2,
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
            x_test, y_test = test_data[i]
            test_dataset = datahandler.SyntheticSequentialTestDataset(args, x_train, y_train, x_test, y_test)

            pattern = full_pattern[args.t:]
            preds_rep = np.zeros(args.T - args.t)
            for task_id, trainer in enumerate(trainer_list):
                pred, truths_rep = trainer.evaluate(test_dataset)
                preds_rep[pattern == task_id] = pred[pattern == task_id]

            preds.append(preds_rep)
            truths.append(truths_rep)
        preds = np.array(preds)
        truths = np.array(truths)

        # store raw predictions and truths
        raw_metrics['preds'].append(preds)
        raw_metrics['truths'].append(truths)

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
        "ci": ci,
        "raw_metrics": raw_metrics
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)

    # save last model
    torch.save(trainer.model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()