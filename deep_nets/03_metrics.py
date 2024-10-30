import pickle
import os
import numpy as np


def create_metrics(fnames, fout, model_names):
    infos = []
    for fname in fnames:
        with open(fname, "rb") as fp:
            info = pickle.load(fp)
        infos.append(info)
    plot_metrics = {}


    for model in range(len(infos)):

        seed_errs = []
        times = []

        for seed in range(len(infos[model])):
            info = infos[model][seed]

            errs = []

            for row in info:
                t, arr = row
                if seed == 0:
                    times.append(t)

                errs.append(np.mean(arr[t:]))

            seed_errs.append(errs)

        seed_errs = np.array(seed_errs)

        mean = np.mean(seed_errs, axis=0)
        std = np.std(seed_errs, axis=0)
        plot_metrics[model_names[model]] = np.array([mean, std, times])

    with open(fout, "wb") as fp:
        pickle.dump(plot_metrics, fp)


if __name__ == "__main__":
    model_names4 = ["ERM", "Prospective", "Online-SGD", "Bayesian GD"]
    model_names2 = ["ERM", "Prospective"]

    os.makedirs("data/metrics", exist_ok=True)

    ##### Synthetic data - Scenario 2
    fnames_syn_s2 = ["./data/checkpoints/scenario2/mlp_erm_errs.pkl",
                     "./data/checkpoints/scenario2/mlp_prospective_errs.pkl",
                     "./data/checkpoints/scenario2/mlp_ft1_errs.pkl",
                     "./data/checkpoints/scenario2/mlp_bgd_errs.pkl"]
    fout_syn_s2 = "data/metrics/syn_scenario2.pkl"

    ##### Synthetic data - Scenario 3
    fnames_syn_s3 = ["./data/checkpoints/scenario3/erm_mlp_errs.pkl",
                     "./data/checkpoints/scenario3/prospective_mlp_errs.pkl",
                     "./data/checkpoints/scenario3/mlp_ft1_errs.pkl",
                     "./data/checkpoints/scenario3/mlp_bgd_errs.pkl",
                     ] 
    fout_syn_s3 = "data/metrics/syn_scenario3.pkl"

    ##### Synthetic data - Scenario 3 Markov 2
    fnames_syn_s3_m2 = ["./data/checkpoints/scenario3_markov2/erm_mlp_errs.pkl",
                        "./data/checkpoints/scenario3_markov2/prospective_mlp_errs.pkl",
                        ]
    fout_syn_s3_m2 = "data/metrics/syn_scenario3_markov2.pkl"


    ##### MNIST - Scenario 2
    fnames_mnist_s2 = ["./data/checkpoints/mnist_s2/erm_mlp_errs.pkl",
                       "./data/checkpoints/mnist_s2/prospective_mlp_errs.pkl",
                       "./data/checkpoints/mnist_s2/mlp_ft1_errs.pkl",
                       "./data/checkpoints/mnist_s2/mlp_bgd_errs.pkl"]      
    fout_mnist_s2 = "./data/metrics/mnist_scenario2.pkl"


    # MNIST - Scenario 3
    fnames_mnist_s3 = ["./data/checkpoints/mnist_s3/erm_mlp_errs.pkl",
                       "./data/checkpoints/mnist_s3/prospective_mlp_errs.pkl",
                       "./data/checkpoints/mnist_s3/mlp_ft1_errs.pkl",
                       "./data/checkpoints/mnist_s3/mlp_bgd_errs.pkl",
                       ]
    fout_mnist_s3 = "data/metrics/mnist_scenario3.pkl"

    # MNIST - Scenario 3 Markov 2
    fnames_mnist_s3_m2 = ["./data/checkpoints/mnist_s3_markov2/erm_mlp_errs.pkl", 
                        "./data/checkpoints/mnist_s3_markov2/prospective_mlp_errs.pkl"]
    fout_mnist_s3_m2 = "data/metrics/mnist_scenario3_markov2.pkl"

    ##### CIFAR - Scenario 2
    fnames_cifar_s2 = ["./data/checkpoints/cifar_s2/erm_cnn_errs.pkl",
                       "./data/checkpoints/cifar_s2/prospective_cnn_o_errs.pkl",
                       "./data/checkpoints/cifar_s2/cnn_o_ft1_errs.pkl",
                       "./data/checkpoints/cifar_s2/cnn_o_bgd_errs.pkl",
                     ]
    fout_cifar_s2 = "data/metrics/cifar_scenario2.pkl"

    ## CIFAR - Scenario 3
    fnames_cifar_s3 = ["./data/checkpoints/cifar_s3/erm_cnn_errs.pkl",
                       "./data/checkpoints/cifar_s3/prospective_cnn_o_errs.pkl",
                       "./data/checkpoints/cifar_s3/cnn_o_ft1_errs.pkl",
                       "./data/checkpoints/cifar_s3/cnn_o_bgd_errs.pkl",
                     ]
    fout_cifar_s3 = "data/metrics/cifar_scenario3.pkl"

    # CIFAR - Scenario 3 Markov 2
    fnames_cifar_s3_m2 = ["./data/checkpoints/cifar_s3_markov2/erm_cnn_errs.pkl",
                          "./data/checkpoints/cifar_s3_markov2/prospective_cnn_o_errs.pkl"]
    fout_cifar_s3_m2 = "data/metrics/cifar_scenario3_markov2.pkl"


    create_metrics(fnames_syn_s2, fout_syn_s2, model_names4)
    create_metrics(fnames_syn_s3, fout_syn_s3, model_names4)
    create_metrics(fnames_syn_s3_m2, fout_syn_s3_m2, model_names2)
    
    create_metrics(fnames_mnist_s2, fout_mnist_s2, model_names4)
    create_metrics(fnames_mnist_s3, fout_mnist_s3, model_names4)
    create_metrics(fnames_mnist_s3_m2, fout_mnist_s3_m2, model_names2)
    
    create_metrics(fnames_cifar_s2, fout_cifar_s2, model_names4)
    create_metrics(fnames_cifar_s3, fout_cifar_s3, model_names4)
    create_metrics(fnames_cifar_s3_m2, fout_cifar_s3_m2, model_names2)
