# nnvmc-intro
Neural Network-based Variational Monte Carlo: introduction

## Required libraries
- absl, optax, jax, pyinstrument, ml_collections, os, math, pandas, numpy, matplotlib, chex, typing, kfac_jax, functools

## Recommendations
It is recommended to install the requirements inside a python virtual environment. For more information visit: https://docs.python.org/3/library/venv.html

In order to support GPUs, one should install JAX libraries with GPU support. For more information visit https://github.com/jax-ml/jax

## Example 
Running the program for the harmonic oscillator
> python3.11 main.py \-\-config src/inputs/ho.py
