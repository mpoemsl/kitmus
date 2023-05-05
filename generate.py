import argparse
import os

import numpy as np
import pandas as pd
from texts import generate_texts
from tqdm import tqdm
from utils import check_overlap, export

parser = argparse.ArgumentParser()

parser.add_argument(
    "--base_dir",
    help="Base directory where all variants will be stored",
    default="kitmus/",
    type=str,
)

parser.add_argument(
    "--names_fp",
    help="Path to file containing name resources",
    default="resources/names.csv",
    type=str,
)
parser.add_argument(
    "--locs_fp",
    help="Path to file containing location resources",
    default="resources/locations.csv",
    type=str,
)
parser.add_argument(
    "--occs_dir",
    help="Path to dir containing occupation resources",
    default="resources/occupations/",
    type=str,
)
parser.add_argument(
    "--templates_dir",
    help="Path to dir containing templates",
    default="resources/templates/multi/",
    type=str,
)

parser.add_argument(
    "--n_examples_train", help="Number of examples in train set", default=2000, type=int
)
parser.add_argument(
    "--n_examples_dev", help="Number of examples in development set", default=400, type=int
)
parser.add_argument(
    "--n_examples_test", help="Number of examples in test set", default=2000, type=int
)

parser.add_argument(
    "--random_seed", help="Seed to initialize random number generator", default=42, type=int
)


VARIANTS = [
    "background-train",
    "background-train-no-noise",
    "background-both",
    "background-inference-real-charfict",
    "background-inference-real-wordfict",
    "background-inference-charfict-real",
    "background-inference-charfict-charfict",
    "background-inference-charfict-wordfict",
]


def main(base_dir="", occs_dir="", names_fp="", locs_fp="", **params):
    os.makedirs(base_dir, exist_ok=True)

    names = pd.read_csv(names_fp, usecols=["lastname"])["lastname"].values
    locs = pd.read_csv(locs_fp).values

    for variant_name in VARIANTS:
        if "no-noise" in variant_name:
            add_noise = False
        else:
            add_noise = True

        if variant_name.startswith("background-inference") or variant_name.startswith(
            "background-both"
        ):
            add_background = True
        else:
            add_background = False

        if variant_name.startswith("background-train") or variant_name.startswith(
            "background-both"
        ):
            occ_prefix = "real_real"
        else:
            occ_prefix = "_".join(variant_name.split("-")[-2:])

        make_variant(
            base_dir + variant_name + "/",
            names,
            locs,
            occs_dir + occ_prefix + ".csv",
            add_noise=add_noise,
            add_background=add_background,
            **params
        )


def make_variant(
    data_dir,
    names,
    locs,
    occs_fp,
    n_examples_train=2000,
    n_examples_dev=400,
    n_examples_test=2000,
    random_seed=42,
    templates_dir="",
    **params
):
    # load occupations
    occupations = pd.read_csv(occs_fp).values

    # load templates
    with open(templates_dir + "meet_sents.txt", "r", encoding="utf-8") as fh:
        meet_sents = np.asarray(fh.read().strip().split("\n\n"))

    with open(templates_dir + "pronoun_sents.txt", "r", encoding="utf-8") as fh:
        pronoun_sents = np.asarray(fh.read().strip().split("\n\n"))

    # create data dir
    os.makedirs(data_dir, exist_ok=True)

    # gather split examples numbers
    n_examples_splits = (n_examples_train, n_examples_dev, n_examples_test)

    # create random number generator
    rng = np.random.default_rng(random_seed)

    # for all subtasks
    for n_ents in [2, 3, 4]:
        # create subtask dir
        subtask_dir = data_dir + "subtask_" + str(n_ents) + "_ents/"
        os.makedirs(subtask_dir, exist_ok=True)

        # create empty lists
        knowledge_splits = []
        task_splits = []
        meet_sent_templates = []
        pronoun_sent_templates = []

        # randomly sample disjunct resources for each split
        rng.shuffle(locs)
        rng.shuffle(names)
        rng.shuffle(occupations)
        rng.shuffle(meet_sents)
        rng.shuffle(pronoun_sents)

        loc_splits = np.array_split(locs, 3, axis=0)
        name_splits = np.array_split(names, 3, axis=0)
        occ_splits = np.array_split(occupations, 3, axis=0)

        meet_splits = (
            np.array_split(meet_sents, 3, axis=0) if meet_sents.size >= 3 else [meet_sents] * 3
        )
        pronoun_splits = (
            np.array_split(pronoun_sents, 3, axis=0)
            if pronoun_sents.size >= 3
            else [pronoun_sents] * 3
        )

        # generate examples for each split
        for (
            n_examples,
            split_names,
            split_locs,
            split_occs,
            split_meet_sents,
            split_pronoun_sents,
        ) in zip(
            n_examples_splits, name_splits, loc_splits, occ_splits, meet_splits, pronoun_splits
        ):
            # generate all template combinations
            split_templates = [
                (meet_sent, 'f"{noise}"', pronoun_sent)
                for meet_sent in split_meet_sents
                for pronoun_sent in split_pronoun_sents
            ]

            # generate texts
            knowledge_texts, task_texts = generate_texts(
                rng,
                n_examples,
                n_ents,
                subtask_dir,
                split_templates,
                split_names,
                split_occs,
                split_locs,
                **params
            )

            # append texts
            knowledge_splits.append(knowledge_texts)
            task_splits.append(task_texts)

            # create subdirs
            os.makedirs(subtask_dir + "knowledge-text-only/", exist_ok=True)
            os.makedirs(subtask_dir + "task-text-only/", exist_ok=True)
            os.makedirs(subtask_dir + "full-text/", exist_ok=True)

        # check for overlap in examples between splits
        check_overlap(knowledge_splits)
        check_overlap(task_splits)

        # export all splits
        for split_name, knowledge_split, task_split in tqdm(
            zip(["train", "validation", "test"], knowledge_splits, task_splits),
            desc=subtask_dir,
            total=3,
        ):
            # export knowledge texts
            export(knowledge_split, subtask_dir + "knowledge-text-only/" + split_name)

            # export task texts
            export(task_split, subtask_dir + "task-text-only/" + split_name)

            # export merged texts
            full_split = [
                knowledge_text + task_text
                for knowledge_text, task_text in zip(knowledge_split, task_split)
            ]
            export(full_split, subtask_dir + "full-text/" + split_name)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
