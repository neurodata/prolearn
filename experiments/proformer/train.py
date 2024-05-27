import hydra
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from utils.init import *
from utils.run_net import *
from models.transformer import *

def get_data(cfg):
    if cfg.dataset == 'periodic-switching':
        from datahandlers.periodic_switching import PeriodicSwitchingDataset, sample
        dataset = PeriodicSwitchingDataset(cfg)
        sample_from_process = lambda t : sample(t, cfg.process.N)
    else:
        raise NotImplementedError
    return dataset, sample_from_process

def get_model(cfg):
    if cfg.architecture == 'pro-transformer':
        model = TransformerClassifier(
            cfg,
            input_size=1,
            num_classes=2
        )
    else:
        raise NotImplementedError
    return model 

@hydra.main(config_path="./config", config_name="conf.yaml")
def main(cfg):
    name = f"{cfg.exp}_t_{cfg.process.t}_posenc_{cfg.net.encoder_type}_aggr_{cfg.net.aggregate_type}"
    init_wandb(cfg, project_name="prol", run_name=name)

    logs_path = f"exps/{cfg.exp}/logs/"
    models_path = f"exps/{cfg.exp}/models/"
    results_path = f"exps/{cfg.exp}/results/"

    Path(logs_path).mkdir(parents=True, exist_ok=True)
    Path(models_path).mkdir(parents=True, exist_ok=True)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    fp = open_log(cfg, logs_path)

    dataset, sample = get_data(cfg)
    trainloader = DataLoader(dataset, batch_size=cfg.train.batch_size)
    model = get_model(cfg)
    model = train(cfg, model, trainloader)
    torch.save(model.state_dict(), models_path + f"/model_t{cfg.process.t}_posenc_{cfg.net.encoder_type}_aggr_{cfg.net.aggregate_type}.pt")

    preds, truths = evaluate(cfg, model, dataset, sample)

    t_vs_avgerr = np.mean(preds != truths, axis=0).squeeze()
    t_vs_stderr = np.std(preds != truths, axis=0).squeeze()

    err = np.mean(preds != truths)
    std = np.std(preds != truths)
    
    info = {
        "avg_err": np.round(err, 4),
        "std_err": np.round(std, 4)
    }
    if cfg.deploy: 
        wandb.log(info)

    info = {
        "times": np.arange(cfg.process.t, cfg.eval.T, 1).tolist(),
        "t_vs_avgerr": t_vs_avgerr.tolist(),
        "t_vs_stderr": t_vs_stderr.tolist(),
    }
    df = pd.DataFrame.from_dict(info)
    tbl = wandb.Table(data=df)
    if cfg.deploy: 
        wandb.log({"results": tbl})
    df.to_csv(results_path + f"t_{cfg.process.t}_posenc_{cfg.net.encoder_type}_aggr_{cfg.net.aggregate_type}.csv")

    cleanup(cfg, fp)


if __name__ == "__main__":
    main()