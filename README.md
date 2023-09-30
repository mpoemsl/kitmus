# KITMUS+

This repository contains the dataset generation code for the `KITMUS` test suite, which is described in the ACL 2023 paper [The KITMUS Test: Evaluating Knowledge Integration from Multiple Sources](https://aclanthology.org/2023.acl-long.841/).

In addition, this branch contains a variant with pretrain-time entity-specific information (PTE), which can be found in [kitmus/ent-train-background-train/](kitmus/ent-train-background-train/) and was created from the WikiData-derived entities in [resources/wiki_entities.csv](resources/wiki_entities.csv).

If you use the dataset or code in your research, please consider [citing](https://github.com/mpoemsl/kitmus#citation) the paper.

## Content

This repository contains:

* The generated `KITMUS` test suite dataset (`kitmus/`)
* The code to generate the dataset (`generate.py`, `texts.py`, `utils.py`)
* The templates and resources to generate the `KITMUS` test suite dataset (`resources/`)
* The train- and test set predictions from the experiments of the paper (`predictions/`)
* The code to evaluate predictions against gold annotations (`evaluate.py`, `utils.py`)

## Setup

Runs on Python 3.8. Required packages can be installed with `pip install -r requirements.txt`.

## Usage

Main scripts:

* `generate.py`
* `evaluate.py`

To learn more about any script and its parameters, run `python <SCRIPT> -h`. If you run into any issues when running the scripts, please create an issue.

### Generating the KITMUS test dataset

To (re-)generate the KITMUS dataset with default hyperparameters as used in the experiments described in the paper, run:

```
python generate.py
```

This will create a folder `kitmus/` which will take up about 4GB of space in total.

### Evaluating a model prediction

To evaluate a `jsonlines` prediction file as output by e.g. [C2F](https://github.com/kentonl/e2e-coref/), [BERT4Coref](https://github.com/mandarjoshi90/coref) or a `tsv` prediction file as output by e.g. [PeTra](https://github.com/shtoshni/petra), [GREP](https://github.com/sattree/gap), run:

```
python evaluate.py <PATH-TO-GOLD-CONLL-FILE> <PATH-TO-PREDICTION-FILE>
```

Prediction files for the experiments featured in the paper can be found in `predictions/`. For a more detailed explanation of the evaluation metrics, see section `5.3 Evaluation` in the paper.

### Generating a custom dataset

The easiest way to generate a custom dataset is to specify an alternative resource directory to `generate.py` with the command line argument `--resources_dir`. A valid resources directory should have the following file structure:

```
<RESOURCES-DIR>/
├── locations.csv
├── names.csv
├── noise
├── occupations
│   ├── charfict_charfict.csv
│   ├── charfict_real.csv
│   ├── charfict_wordfict.csv
│   ├── real_charfict.csv
│   ├── real_real.csv
│   └── real_wordfict.csv
├── pronouns.json
├── templates
│   ├── background_knowledge_sentence.txt
│   ├── entity_mention_templates.json
│   ├── entspec_knowledge_sentence.txt
│   ├── meet_sentence.txt
│   └── pronoun_sentence.txt
└── vocab.json

```

The directory `<RESOURCES-DIR>/noise/` is not necessary for generating the `background-train-no-noise` variant. Similarly, only `<RESOURCES-DIR>/occupations/real_real.csv` is needed for the `background-train-*` variants. Take a look at the files provided in `resources/` to understand the necessary fields and structure of each kind of file.

If the custom dataset is in a language with a similar morphological structure as English, it should be sufficient to modify only the resources. For other languages, it may be necessary to write custom rules in the functions `create_knowledge_sents` and `create_task_sents` in `texts.py`. An example of a custom rule for the English `a/an` distinction is already present in the code.

## Citation

```
@inproceedings{arodi-etal-2023-kitmus,
    title = "The {KITMUS} Test: Evaluating Knowledge Integration from Multiple Sources",
    author = {Arodi, Akshatha  and
      P{\"o}msl, Martin  and
      Suleman, Kaheer  and
      Trischler, Adam  and
      Olteanu, Alexandra  and
      Cheung, Jackie Chi Kit},
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.841",
    pages = "15088--15108",
}

```

