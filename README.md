# Symbolic Model Based Reinforcement Learning
This repository contains the code used for the project of the PhD course [*Neural Networks and Deep Learning*](https://retis.sssup.it/~giorgio/courses/neural/nn.html). The project goal was reproducing and deeply understanding the paper *Symbolic Model-Based Reinforcement Learning* by P. A. Karmienny and S. Lamprier.
Thus, this project is *not* an original work, it is only meant for didactic reasons.

## Installation
The dependencies are collected in `environment.yaml` and can be installed, after cloning the repository, using [`mamba`]("https://github.com/mamba-org/mamba"):
```bash
$ mamba env create -f environment.yaml
```

Once the environment is installed and activated, install the library using

```bash
$ pip install .
```

## Usage
To reproduce the results, change the hyperparameters in `config_files.py` and then simply use
```bash
$ python src/symbolic_mbrl/simple1dmdp.py
```

or

```bash
$ python src/symbolic_mbrl/cartpole.py
```