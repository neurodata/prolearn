'''
Run Markov (case 3) exps
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
    get_torch_dataset,
    get_multi_indices_and_map
)
from prol.utils import log_exp_config
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
        "indices_file": cfg.indices_file,

        # experiment
        "method": cfg.method,         # select from {proformer, cnn, mlp, timecnn}
        "N": total_indices["N"],                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": total_indices["T"],                   # future time horizon
        "seed": 1996,   
        "device": cfg.device,          # device
        "reps": total_indices["inner_reps"],                 # number of test reps
        "outer_reps": total_indices["outer_reps"],         
       
        # proformer
        "proformer" : {
            "contextlength": 60 if cfg.t < 500 else 200, 
            "encoding_type": 'freq',      
            "multihop": cfg.multihop
        },

        # conv_proformer
        "conv_proformer" : {
            "contextlength": 60 if cfg.t < 500 else 200, 
            "encoding_type": 'freq',      
            "multihop": cfg.multihop
        },

        # timecnn
        "timecnn": {
            "encoding_type": 'freq', 
        },

        # timeresnet
        "timeresnet": {
            "encoding_type": 'freq', 
        },
              
        # training params
        "lr": 1e-3,         
        "batchsize": cfg.batchsize,
        "epochs": cfg.epochs,
        "augment": cfg.augment,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')
    log_exp_config(log, args, params)

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

    risk_list = []
    raw_metrics = {
        "t": args.t,
        "preds": [],
        "truths": []
    }
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
            data_kwargs = {
                "dataset": torch_dataset, 
                "transform": train_transform,
                "seqInd": train_SeqInd, 
                "maplab": maplab
            }
            train_dataset = datahandler.VisionSequentialDataset(args, **data_kwargs)
        else:
            # when the model doesn't see any training data at all
            if args.method in ['proformer', 'conv_proformer']:
                # set a random context
                train_SeqInd = test_seqInds[np.random.randint(args.reps)][
                    np.random.randint(args.T, size=params[args.method]['contextlength'])
                ]
            train_dataset = [] 

        # model
        model_kwargs = method.model_defaults(args.dataset)
        if args.method in ['proformer', 'conv_proformer', 'timecnn', 'timeresnet']:
            model_kwargs['encoding_type'] = params[args.method]['encoding_type']
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=max_num_classes,
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