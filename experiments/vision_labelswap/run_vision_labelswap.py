'''
Vision label swap exps
'''

import importlib
import numpy as np
from tqdm.auto import tqdm
import pickle

import hydra
import logging

from prol.process import (
    get_cycle,
    get_torch_dataset,
    get_task_indicies_and_map,
    get_sequence_indices
)
from prol.utils import get_dataloader

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
        # dataset
        "dataset": "cifar-10",
        "task": [[0, 1]],    # task specification

        # experiment
        "method": "timeresnet",         # select from {proformer, cnn, mlp, timecnn}
        "N": 20,                     # time between two task switches                   
        "t": cfg.t,                  # training time
        "T": 5000,                   # future time horizon
        "seed": 1996,   
        "device": "cuda:1",          # device
        "reps": 100,                 # number of test reps
        "outer_reps": 3,         
       
        # proformer
        "proformer" : {
            "contextlength": 200, 
            "encoding_type": 'freq',      
            "multihop": True
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
        "batchsize": 64,
        "epochs": 2000,
        "verbose": True
    }
    args = SetParams(params)
    log.info(f'{params}')

    # get source dataset
    root = 'path/to/data_folder'
    torch_dataset = get_torch_dataset(root, name=args.dataset)
    
    # get indices for each task
    taskInd, maplab, dummylab = get_task_indicies_and_map(
        tasks=args.task,
        y=torch_dataset.targets.numpy(),
        type='label-swap'
    )

    # modify the source dataset (assign dummy labels to half of the task labels)
    torch_dataset.targets[taskInd[1]] = torch_dataset.targets[taskInd[1]].apply_(dummylab)

    risk_list = []
    for outer_rep in range(args.outer_reps):
        log.info(" ")
        
        # get a training sequence
        seed = args.seed * outer_rep * 2357
        train_SeqInd, updated_taskInd = get_sequence_indices(
            N=args.N, 
            total_time_steps=args.t, 
            tasklib=taskInd, 
            seed=seed,
            remove_train_samples=True
        )

        # sample a bunch of test sequences
        test_seqInds = [
            get_sequence_indices(args.N, args.T, updated_taskInd, seed=seed+1000*(inner_rep+1))
            for inner_rep in range(args.reps)
        ]

        # get the module for the specified method
        method, datahandler = get_modules(args.method)

        # form the train dataset
        data_kwargs = {
            "dataset": torch_dataset, 
            "seqInd": train_SeqInd, 
            "maplab": maplab
        }
        train_dataset = datahandler.VisionSequentialDataset(args, **data_kwargs)

        # model
        model_kwargs = method.model_defaults(args.dataset)
        if args.method == 'proformer':
            model_kwargs['encoding_type'] = args.proformer['encoding_type']
        elif args.method == 'timecnn':
            model_kwargs['encoding_type'] = args.timecnn['encoding_type']
        elif args.method == 'timeresnet':
            model_kwargs['encoding_type'] = args.timecnn['encoding_type']
        log.info(f'{model_kwargs}')
        model = method.Model(
            num_classes=len(args.task[0]),
            **model_kwargs
        )
        
        # train
        trainer = method.Trainer(model, train_dataset, args)
        trainer.fit(log)

        # evaluate
        preds = []
        truths = []
        for i in tqdm(range(args.reps)):
            # form a test dataset for each test sequence
            test_kwargs = {
            "dataset": torch_dataset, 
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
        instantaneous_risk = np.mean(preds != truths, axis=0).squeeze()
        std_error = np.std(preds != truths, axis=0).squeeze()
        ci = std_error * 1.96/np.sqrt(args.reps).squeeze()

        time_averaged_risk = np.mean(preds != truths)
        log.info(f"error = {time_averaged_risk:.4f}")
        risk_list.append(time_averaged_risk)

    risk = np.mean(risk_list)
    log.info(f"risk at t = {args.t} : {risk:.4f}")
    
    outputs = {
        "args": params,
        "risk": risk,
        "inst_risk": instantaneous_risk,
        "ci": ci
    }
    with open('outputs.pkl', 'wb') as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    main()