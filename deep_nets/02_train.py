import hydra 
import torch
import os
import numpy as np
import pickle
from utils.init import init_wandb, set_seed, open_log
from utils.data import create_dataloader
from utils.net import create_net
from utils.runner import train
from utils.bgd import BGD


def create_optimizer(cfg, net):
    if 'mnist' or 'cifar' in cfg.data.path:
        lr = 0.01
    else:
        lr = 0.1 # for synthetic

    if cfg.bgd:
        params = [{'params': [p]} for p in net.parameters()]
        optimizer = BGD(params, std_init=0.01)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=0.9, nesterov=True,
                                    weight_decay=0.00001)
    return optimizer


def cache_dataload(cfg):
    # Load data
    with open(cfg.data.path, 'rb') as fp:
        data = pickle.load(fp)
        data['x'] = torch.FloatTensor(data['x'])
        data['y'] = torch.LongTensor(data['y'])
        data['t'] = torch.FloatTensor(data['t'])
    return data


@hydra.main(config_path="./config/train", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(cfg, project_name="prospective")
    set_seed(cfg.seed)
    open_log(cfg)

    data = cache_dataload(cfg)

    online = cfg.fine_tune is not None

    if online:
        tskip = 1
    else:
        tskip = cfg.tskip
        
    allerrs_t = []
    for seed in range(cfg.numseeds):
        # Restart net, optimizer if online method
        all_errs = []

        if online:
            net = create_net(cfg)
            opt = create_optimizer(cfg, net)

        for t in range(cfg.tstart, cfg.tend, tskip):

            # Create dataset
            loaders = create_dataloader(cfg, t, seed, data)

            # Create new network if not fine-tuning
            if not online:
                net = create_net(cfg)
                net.to(cfg.dev)
                opt = create_optimizer(cfg, net)
                run_eval = True
            else:
                run_eval = ((t - cfg.tstart) % cfg.tskip) == 0

            # Run and evaluate network
            errs = train(cfg, net, opt, loaders, run_eval)

            if run_eval:
                all_errs.append((t, errs))
                print("Time %d, Seed %d, Error: %.4f" % (t, seed, np.mean(errs)))

        allerrs_t.append(all_errs)

    # Save errs
    fdir = os.path.join('data/checkpoints', cfg.tag)
    os.makedirs(fdir, exist_ok=True)
    with open(os.path.join(fdir, cfg.name + "_errs.pkl"), 'wb') as fp:
        pickle.dump(allerrs_t, fp)


if __name__ == "__main__":
    main()

