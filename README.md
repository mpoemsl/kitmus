# KITMUS

This repository contains the data and generation code for the `KITMUS` test suite, which is described in the upcoming ACL 2023 paper:

```
The KITMUS Test: Evaluating Knowledge Integration from Multiple Sources
```

A preprint of the paper is available [on ArXiv](https://arxiv.org/abs/2212.08192).


## Setup

Runs on Python 3.8. Required packages can be installed with `pip install -r requirements.txt`.


## Usage

Run `python generate.py` to (re-)generate the KITMUS dataset with default hyperparameters. This will create a folder `kitmus/` which will take up about 4GB of space.

Run `python evaluate.py <PATH-TO-GOLD-CONLL-FILE> <PATH-TO-PREDICTION-FILE>` to evaluate a prediction. The prediction file can be a `jsonlines` or `tsv` file. Predictions for the experiments featured in the paper can be found in `predictions/`.

To learn more about any script and its parameters, run `python <SCRIPT> -h`.
