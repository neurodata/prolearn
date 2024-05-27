# Prospective Learning

This repo contains code for prospective learning experiments.

## Usage

* Dependendies: CUDA Toolkit 12.1, pytorch 2.3.0
* Set up the conda environment:

    ```
    conda env create -f environment.yml
    ```
* Install the `prol` package
    ```
    pip install .
    ```

## Directory Structure

The `prol` folder contains scripts for implementing all the models, sampling from processes, and datahandling. It is organized as follows.

```
prol
├── __init__.py
├── datahandlers                 # data handlers for all the models
│   ├── __init__.py
│   ├── cnn_handle.py
│   ├── conv_proformer_handle.py
│   ├── mlp_handle.py
│   ├── proformer_handle.py
│   ├── resnet_handle.py
│   ├── timecnn_handle.py
│   ├── timemlp_handle.py
│   └── timeresnet_handle.py
├── models                       # contains the model architectures
│   ├── __init__.py
│   ├── base_trainer.py          # trainer class for fitting/evaluating models
│   ├── cnn.py
│   ├── conv_proformer.py
│   ├── masked_proformer.py
│   ├── mlp.py
│   ├── proformer.py
│   ├── resnet.py
│   ├── timecnn.py
│   ├── timemlp.py
│   └── timeresnet.py
├── process.py                   # methods to draw data from processes
└── utils.py                     # misc. utils
```

The `experiments` folder has the scripts to executing the experiments considered in the draft.

```
experiments
├── proformer                       # time-embedding comparison
│   ├── config_mnist.yaml
│   ├── precompute_indices.py
│   ├── run_proformer.py
│   ├── script.sh
│   └── train.py
├── synthetic                       # synthetic tasks (deterministic Markov)
│   ├── config.yaml
│   ├── run_baseline_1.py
│   ├── run_synthetic.py
│   └── script.sh
├── synthetic_markov                # synthetic tasks (prob. Markov)
│   ├── config.yaml
│   ├── precompute_task_sequence.py
│   ├── run_baseline_1.py
│   ├── run_synthetic_markov.py
│   └── scripts.sh
├── vision_markov                   # vision tasks (prob. Markov)
│   ├── config_cifar10.yaml
│   ├── config_mnist.yaml
│   ├── precompute_indices.py
│   ├── run_baseline_1.py
│   ├── run_baseline_2.py
│   ├── run_vision_markov.py
│   └── script.sh
└── vision_multi                    # vision tasks (deterministic Markov)
    ├── config_cifar10.yaml
    ├── config_mnist.yaml
    ├── precompute_indices.py
    ├── run_baseline_1.py
    ├── run_baseline_2.py
    ├── run_vision_multi.py
    └── script.sh
```