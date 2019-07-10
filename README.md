[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.2602529.svg)](https://doi.org/10.5281/zenodo.2602529)

GroomRL
=======

This repository contains the code and results presented in
[arXiv:1903.09644](https://arxiv.org/abs/1903.09644 "GroomRL paper").

## About

GroomRL is a reinforcement learning framework to train jet grooming strategies.

## Install GroomRL

GroomRL is tested and supported on 64-bit systems running Linux.

Install GroomRL with Python's pip package manager:
```
git clone https://github.com/JetsGames/groomrl.git
cd groomrl
pip install .
```
To install the package in a specific location, use
the "--target=PREFIX_PATH" flag.

This process will copy the `groomrl` program to your environment python path.

We recommend the installation of the GroomRL package using a `miniconda3`
environment with the
[configuration specified here](https://github.com/JetsGame/groomrl/blob/master/environment.yml).

GroomRL requires the following packages:
- python3
- numpy
- [fastjet](http://fastjet.fr/) (compiled with --enable-pyext)
- gym
- matplotlib
- pandas
- keras
- keras-rl
- tensorflow
- json
- gzip
- argparse
- hyperopt (optional)

## Pre-trained models

The final models presented in
[arXiv:1903.09644](https://arxiv.org/abs/1903.09644 "GroomRL paper")
are stored in:
- results/groomerW_final: GroomRL model trained on W jets.
- results/groomerTop_final: GroomRL model trained on top jets.

## Input data

All data used for the final models can be downloaded from the git-lfs repository
at https://github.com/JetsGame/data.

## Running the code

In order to launch the code run:
```
groomrl <runcard.json> [--output <folder>]
```

This will create a folder containing the result of the fit.

To create diagnostic plots, run with the --plot option or use
```
groomrl --model <folder> --plot [--nev n]
```
on the previous output folder.

The groomer can also be exported with the --cpp option, or by running
```
groomrl --model <folder> --cpp
```
This will create a cpp/model.nnet file that can be imported in a c++ framework.

To apply an existing grooming model to a new data set, you can use
```
groomrl --model <folder> --data <datafile> [--nev n]
```
which will create a new directory in `<folder>` using the datafile name.

## C++ interface

To call models trained with GroomRL in a C++ code, install and use the
[libGroomRL library](https://github.com/JetsGame/libGroomRL).

## References

* S. Carrazza and F. A. Dreyer, "Jet grooming through reinforcement learning,"
  [arXiv:1903.09644](https://arxiv.org/abs/1903.09644 "GroomRL paper")
