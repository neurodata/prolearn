'''
Synthetic label swap exps
'''

import importlib
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
from hydra.utils import get_original_cwd
import logging

from prol.process import (
    get_synthetic_data,
    get_markov_chain
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

@hydra.main(config_path=".", config_name="config")
def main(cfg):

    cwd = pathlib.Path(get_original_cwd())

    # load the saved indicies
    patterns_file = cwd / f'{cfg.patterns_file}.pkl'
    with open(patterns_file, 'rb') as f:
        patterns_dict = pickle.load(f)

    # input parameters
    params = {
        # experiment params
        "dataset": "synthetic",
        "seq_file": cfg.patterns_file,
        "method": cfg.method,
        "N": patterns_dict["N"],                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": patterns_dict["T"],                   # future time horizon
        "seed": 1996,
        "device": cfg.device,
        "reps": 20,                 # number of test reps
        "outer_reps": patterns_dict["outer_reps"],

        # proformer
        "proformer" : {
            "contextlength": 60 if cfg.t < 500 else 200, 
            "encoding_type": 'freq',      
            "multihop": cfg.multihop
        },

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
    fname = cwd / f'{args.seq_file}.pkl'
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
            x_train, y_train = get_synthetic_data(
                N=args.N,
                total_time_steps=300,
                seed=seed
            ) # dummy training data for t = 0 case
            idx = np.random.permutation(len(y_train))
            x_train, y_train = x_train[idx], y_train[idx]

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
            train_dataset = datahandler.SyntheticSequentialDataset(args, x_train, y_train)
        else:
            train_dataset = [] 

        # model
        model_kwargs = method.model_defaults(args.dataset)
        if args.method == 'proformer':
            model_kwargs['encoding_type'] = params[args.method]['encoding_type']
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=2,
            **model_kwargs
        )
        
        # train
        trainer = method.Trainer(model, train_dataset, args)
        if args.t > 0:
            trainer.fit(log)

        # evaluate
        preds = []
        truths = []
        for i in tqdm(range(args.reps)):
            # form a test dataset for each test sequence
            x_test, y_test = test_data[i]
            test_dataset = datahandler.SyntheticSequentialTestDataset(args, x_train, y_train, x_test, y_test)
            preds_rep, truths_rep = trainer.evaluate(test_dataset)
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


if __name__ == "__main__":
    main()