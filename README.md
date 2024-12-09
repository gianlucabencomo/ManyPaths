# ManyPaths

This repository is a messy JAX / PyTorch implementation of the experiments for the Many Paths Project (README last updated: December 9th, 2024).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements.txt.

```bash
pip install requirements.txt
```

## Comments on Organization

There are three main folders: archive, gaussians, and numerical. archive contains some code for Bayesian linear regression experiments with fourier basis functions that I was playing around with (and is mostly complete). It also contains some code for doing MAML in JAX (haven't verified if it still runs / compiles). 

gaussians and numerical folders are my code for the Gaussians and Numerical tasks outlined in the Google doc. Both of these experiments are incomplete, but have a portion of the implementation done.