import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from texts import generate_texts
from utils import check_overlap, export, load_resources, load_templates

parser = argparse.ArgumentParser()

parser.add_argument(
    "--export_dir",
    help="Base directory where all variants will be stored",
    default="kitmus/",
    type=str,
)

parser.add_argument(
    "--resources_dir",
    help="Resources directory from which resources will be loaded",
    default="resources/",
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

RESOURCE_SPLIT_ORDER = [
    "locations",
    "names",
    "occupations",
    "meet_sents",
    "pronoun_sents",
    "background_sents",
    "entspec_sents",
]


def main(export_dir: str, resources_dir: str, **params):
    os.makedirs(export_dir, exist_ok=True)
    assert os.path.isdir(resources_dir), f"resources_dir {resources_dir} not found!"

    splittable_resources, vocab, pronouns = load_resources(resources_dir)

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

        occupations = pd.read_csv(
            os.path.join(resources_dir, "occupations", f"{occ_prefix}.csv")
        ).values

        splittable_templates, entity_mention_templates = load_templates(resources_dir)

        splittable_resources.update(
            {
                "occupations": occupations,
                **splittable_templates,
            }
        )

        variant_export_dir = os.path.join(export_dir, variant_name)

        make_variant(
            variant_export_dir,
            splittable_resources,
            entity_mention_templates,
            vocab,
            pronouns,
            add_noise=add_noise,
            add_background=add_background,
            **params,
        )


def make_variant(
    export_dir,
    splittable_resources,
    entity_mention_templates,
    vocab,
    pronouns,
    n_examples_train=2000,
    n_examples_dev=400,
    n_examples_test=2000,
    random_seed=42,
    **params,
):
    # create data dir
    os.makedirs(export_dir, exist_ok=True)

    # gather split examples numbers
    n_examples_splits = (n_examples_train, n_examples_dev, n_examples_test)

    # create random number generator
    rng = np.random.default_rng(random_seed)

    # for all subtasks
    for n_ents in [2, 3, 4]:
        # create subtask dir
        subtask_dir = os.path.join(export_dir, f"subtask_{n_ents}_ents")
        os.makedirs(subtask_dir, exist_ok=True)

        # create empty lists
        knowledge_splits = []
        task_splits = []

        entity_mention_template = entity_mention_templates.get(f"{n_ents}_ent_mention")

        # randomly sample disjunct resources for each split
        resource_splits = dict()

        for resource_name in RESOURCE_SPLIT_ORDER:
            resource_values = splittable_resources.get(resource_name)
            if len(resource_values) >= 3:
                rng.shuffle(resource_values)
                resource_splits[resource_name] = np.array_split(resource_values, 3, axis=0)
            else:
                resource_splits[resource_name] = [resource_values] * 3

        # generate examples for each split
        for split_ix, n_examples in enumerate(n_examples_splits):
            # generate all task text template combinations
            task_text_templates = [
                (meet_sent, "{noise}", pronoun_sent)
                for meet_sent in resource_splits["meet_sents"][split_ix]
                for pronoun_sent in resource_splits["pronoun_sents"][split_ix]
            ]

            # generate texts
            knowledge_texts, task_texts = generate_texts(
                rng,
                n_examples,
                n_ents,
                subtask_dir,
                vocab,
                pronouns,
                task_text_templates,
                entity_mention_template,
                resource_splits["background_sents"][split_ix],
                resource_splits["entspec_sents"][split_ix],
                resource_splits["names"][split_ix],
                resource_splits["occupations"][split_ix],
                resource_splits["locations"][split_ix],
                **params,
            )

            # append texts
            knowledge_splits.append(knowledge_texts)
            task_splits.append(task_texts)

            # create subdirs
            os.makedirs(os.path.join(subtask_dir, "knowledge-text-only"), exist_ok=True)
            os.makedirs(os.path.join(subtask_dir, "task-text-only"), exist_ok=True)
            os.makedirs(os.path.join(subtask_dir, "full-text"), exist_ok=True)

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
            os.makedirs(os.path.join(subtask_dir, "knowledge-text-only", split_name), exist_ok=True)
            export(knowledge_split, os.path.join(subtask_dir, "knowledge-text-only", split_name))

            # export task texts
            os.makedirs(os.path.join(subtask_dir, "task-text-only", split_name), exist_ok=True)
            export(task_split, os.path.join(subtask_dir, "task-text-only", split_name))

            # export merged texts
            full_split = [
                knowledge_text + task_text
                for knowledge_text, task_text in zip(knowledge_split, task_split)
            ]
            os.makedirs(os.path.join(subtask_dir, "full-text", split_name), exist_ok=True)
            export(full_split, os.path.join(subtask_dir, "full-text", split_name))


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
