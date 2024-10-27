# Prospective Learning: Learning for a Dynamic Future

## Overview

In real world applications, the distribution of the data and our goals evolve over time. And we therefore care about performance over time, rather than just instantaneous performance. Yet, the prevailing theoretical framework in artificial intelligence (AI) is probably approximately correct (PAC) learning, which ignores time. Existing strategies (both theoretical and empirical) to address the dynamic nature of distributions and goals have typically assumed that the optimal hypothesis is fixed, rather than dynamic. Here, we enrich PAC learning by allowing the optimal hypothesis to change over time. This generalizes the notion of learning to something we call "prospective learning". We prove that `retrospective' (i.e., canonical) empirical risk minimization cannot solve certain trivially simple prospective learning problems. We then prove that a simple prospective augmentation to empirical risk minimization provably solves certain prospective learning problems. Numerical experiments illustrate that prospective learners can prospectively learn on synthetic  and visual recognition tasks constructed from MNIST and CIFAR, in contrast to their retrospective counterparts. This framework offers a conceptual link towards both (i) improving AI solutions for currently intractable problems, and (ii) better characterizing the naturally intelligent systems that solve them.

## Figure

## Dependencies

* Dependendies: CUDA Toolkit 12.1, pytorch 2.3.0
* Set up the conda environment:

    ```
    conda env create -f environment.yml
    ```

* System requirements:

## Directory Structure

## Demo (executing scripts for generating figures)

## Tutorial (Rice's code organized into a notebook)