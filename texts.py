from typing import Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def generate_texts(
    rng: np.random.Generator,
    n_examples: int,
    n_ents: int,
    subtask_dir: str,
    vocab: Dict[str, str],
    pronouns: List[Tuple[float, str, str]],
    task_text_templates: List[str],
    entity_mention_template: str,
    background_sent_templates: List[str],
    entspec_sent_templates: List[str],
    names: Iterable,
    occupations: Iterable,
    locations: Iterable,
    **params,
) -> Tuple[List[Tuple[Tuple[str]]], List[Tuple[Tuple[str]]]]:
    """Create examples with randomized templates."""

    knowledge_texts = []
    task_texts = []

    # create examples
    for ix in tqdm(range(n_examples), desc=f"Generating {subtask_dir} split"):
        # sample common pronoun for all entities in this example
        pronoun, pronoun_be = rng.choice(
            a=[pronoun[1:] for pronoun in pronouns], p=[pronoun[0] for pronoun in pronouns]
        )

        # sample task text template
        task_text_template = rng.choice(task_text_templates)

        # sample entity cluster
        ent_clusters = list(range(1, n_ents + 1))
        pronoun_cluster = rng.choice(ent_clusters)

        # sample names for entities
        orig_names = rng.choice(names, size=n_ents, replace=False).tolist()
        ent_ments = [
            (name, "NNP", "(" + str(cluster) + ")")
            for name, cluster in zip(orig_names, ent_clusters)
        ]

        # sample occupations for entities
        occs, occ_descs = zip(*rng.choice(occupations, size=n_ents, replace=False))
        occs = [
            tuple(zip(occ_name.split(" "), ["NN"] * (occ_name.count(" ") + 1))) for occ_name in occs
        ]

        # create dict linking occupations and descriptions
        occ2desc = dict(zip(occs, map(eval, occ_descs)))

        # convert to list of entity dicts
        entities = [
            {"mention": mention, "occupation": occ, "cluster": cluster}
            for mention, occ, cluster in zip(ent_ments, occs, ent_clusters)
        ]

        # create knowledge text
        rng.shuffle(entities)
        if len(background_sent_templates) > 1:
            background_sent_template = rng.choice(background_sent_templates)
        else:
            background_sent_template = background_sent_templates[0]

        if len(entspec_sent_templates) > 1:
            entspec_sent_template = rng.choice(entspec_sent_templates)
        else:
            entspec_sent_template = entspec_sent_templates[0]

        knowledge_sents = create_knowledge_sents(
            rng,
            entities,
            occ2desc,
            background_sent_template,
            entspec_sent_template,
            vocab,
            **params,
        )

        # sample location for entities to meet
        location_str, noise_fp = rng.choice(locations)
        location = eval(location_str)

        # create task text
        rng.shuffle(entities)
        task_sents = create_task_sents(
            rng,
            task_text_template,
            entity_mention_template,
            vocab,
            entities,
            location,
            pronoun,
            pronoun_be,
            pronoun_cluster,
            occ2desc,
            noise_fp,
            **params,
        )

        # append texts
        knowledge_texts.append(knowledge_sents)
        task_texts.append(task_sents)

    return knowledge_texts, task_texts


def create_knowledge_sents(
    rng: np.random.Generator,
    entities: List[Dict[str, Iterable]],
    occ2desc: Dict[Tuple, Iterable],
    background_sent_template: str,
    entspec_sent_template: str,
    vocab: Dict[str, str],
    add_background: bool = False,
    **kwargs,
) -> Tuple[Tuple[str]]:
    """Create a knowledge text for one example."""

    # create entity-specific knowledge sents linking entities to their occupations
    knowledge_sents = []

    for entity in entities:
        if entity["occupation"][0][0][0] in {"a", "e", "i", "o", "u"}:
            a_an = vocab["an"]
        else:
            a_an = vocab["a"]

        entspec_sent = eval(
            entspec_sent_template.format(
                entity_mention=entity["mention"],
                entity_occupation=str(list(entity["occupation"])).strip("[]"),
                a_an=a_an,
                **vocab,
            )
        )
        knowledge_sents.append(entspec_sent)

    # create background knwoedge sents linking occupations to situations
    if add_background:
        for entity in entities:
            if entity["occupation"][0][0][0] in {"a", "e", "i", "o", "u"}:
                a_an = vocab["an"]
            else:
                a_an = vocab["a"]

            background_sent = eval(
                background_sent_template.format(
                    entity_occupation=str(list(entity["occupation"])).strip("[]"),
                    a_an=a_an,
                    occupation_description=str(occ2desc[entity["occupation"]]).strip("[]"),
                    **vocab,
                )
            )
            knowledge_sents.append(background_sent)

    # capitalize first word of each sentence
    for sent in knowledge_sents:
        sent[0] = tuple([sent[0][0].capitalize()] + list(sent[0][1:]))

    # shuffle all sentences
    rng.shuffle(knowledge_sents)

    # convert to tuples
    knowledge_sents = tuple(map(tuple, knowledge_sents))

    return knowledge_sents


def create_task_sents(
    rng: np.random.Generator,
    task_text_template: str,
    entity_mention_template: str,
    vocab: Dict[str, str],
    entities: List[Dict[str, Iterable]],
    location: str,
    pronoun: str,
    pronoun_be: str,
    pronoun_cluster: int,
    occ2desc: Dict[Tuple, Iterable],
    noise_fp: str,
    add_noise: bool = True,
    **kwargs,
) -> Tuple[Tuple[str]]:
    """Create a task text for one example."""

    mentions = entity_mention_template.format(*[entity["mention"] for entity in entities], **vocab)

    # determine noise sentence
    if add_noise:
        # read noise sentences for this location
        with open(noise_fp, "r") as fh:
            noise_lines = fh.readlines()

        # insert a random noise sent in betweeen
        noise = eval(rng.choice(noise_lines))
    else:
        noise = []

    # determine situation description
    true_entity = max(entities, key=lambda entity: entity["cluster"] == pronoun_cluster)
    situation = occ2desc[true_entity["occupation"]]

    # determine be
    be_with_pos = (pronoun_be, "VBD")

    # determine pronoun mention
    pronoun_mention = (pronoun, "PRP", "(" + str(pronoun_cluster) + ")")

    # fill template
    task_sents = [
        eval(
            sent_template.format(
                mentions=str(mentions).strip("[]"),
                pronoun=pronoun_mention,
                be=be_with_pos,
                noise=noise,
                location=str(location).strip("[]"),
                situation=str(situation).strip("[]"),
                **vocab,
            )
        )
        for sent_template in task_text_template
    ]

    # filter empty noise
    task_sents = list(filter(len, task_sents))

    # capitalize first word of each sent
    for sent in task_sents:
        sent[0] = tuple([sent[0][0].capitalize()] + list(sent[0][1:]))

    # convert to tuples
    task_sents = tuple(map(tuple, task_sents))

    return task_sents
