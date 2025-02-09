# Prospective Learning: Principled Extrapolation to the Future

## Overview

In real-world applications, the distribution of the data, and our goals, evolve
over time. The prevailing theoretical framework for studying machine learning,
namely probably approximately correct (PAC) learning, largely ignores time. As a
consequence, existing strategies to address the dynamic nature of data and goals
exhibit poor real-world performance. This paper develops a theoretical framework
called "Prospective Learning" that is tailored for situations when the optimal
hypothesis changes over time. In PAC learning, empirical risk minimization (ERM)
is known to be consistent. We develop a learner called Prospective ERM, which
returns a sequence of predictors that make predictions on future data. We prove that
the risk of prospective ERM converges to the Bayes risk under certain assumptions
on the stochastic process generating the data. Prospective ERM, roughly speaking,
incorporates time as an input in addition to the data. We show that standard ERM
as done in PAC learning, without incorporating time, can result in failure to learn
when distributions are dynamic. Numerical experiments illustrate that prospective
ERM can learn synthetic and visual recognition problems constructed from MNIST
and CIFAR-10.

<p align="center">
    <img src="assets/cartoon.jpg" alt="Alt text" width="50%"/>
</p>

## Dependencies

To setup a mamba (conda) environment, run

```sh
micromamba env create -f environment.yml
```

## Tutorial

We have written a [tutorial](https://github.com/neurodata/prolearn/blob/main/tutorials/tutorial.ipynb) that provides a quick introduction to prospective learning.

## Figures

Run the following to generate the results and figures for the binary examples.

```sh
bash binary/binary_examples.sh
```

To run the neural net experiments, run
```sh
cd deep_nets
bash scripts/generate_data.sh
bash scripts/train_scenario2.sh
bash scripts/train_scenario3.sh
bash scripts/create_plots.sh
```

## Cite us

If you find this code useful consider citing

    @article{desilva2024prospective,
      title={Prospective Learning: Principled Extrapolation to the Future
      author={De Silva*, Ashwin and Ramesh*, Rahul and Yang*, Rubing and Yu, Siyu and Vogelstein*, Joshua T and Chaudhari*, Pratik},
      journal={Advances in neural information processing systems},
      year={2024}
    }

