# KITMUS

This repository contains the data and generation code for the `KITMUS` test suite, which is described in the upcoming ACL 2023 paper:

```
The KITMUS Test: Evaluating Knowledge Integration from Multiple Sources
```

A preprint is available [on ArXiv](https://arxiv.org/abs/2212.08192).

## Setup

Runs on Python 3.8. Required packages can be installed with `pip install -r requirements.txt`.


## Usage

Run `python generate.py` to generate the KITMUS dataset with default hyperparameters. This will create a folder `kitmus/` which will take up about 4GB of space.

Run `python evaluate.py <PATH-TO-GOLD-CONLL-FILE> <PATH-TO-PREDICTED-FILE>` to evaluate a prediction. The predicted file can be a `jsonlines` or `tsv` file.

To learn more about any script and its parameters, run python <script>.py -h.
