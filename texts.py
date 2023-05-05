from tqdm import tqdm

# pronouns with associated probabilities
PRONOUNS = {
    "a": ("she", "he", "they", "ze", "ey"),
    "p": (0.4, 0.4, 0.1, 0.05, 0.05),
}

# template phrases with pos tags
AN = ("an", "DT")
IS_A = [("is", "VBZ"), ("a", "DT")]
THE_WORK_OF_A = [("The", "DT"), ("work", "NN"), ("of", "IN"), ("a", "DT")]


def generate_texts(
    rng, n_examples, n_ents, dir_name, templates, names, occupations, locations, **params
):
    """Create examples with randomized templates."""

    knowledge_texts = []
    task_texts = []

    # create examples
    for ix in tqdm(range(n_examples), desc=dir_name):
        # sample common pronoun for all entities in this example
        pronoun = rng.choice(**PRONOUNS)

        # sample task text template
        template = rng.choice(templates)

        # sample entity cluster
        ent_clusters = list(range(1, n_ents + 1))
        pronoun_cluster = rng.choice(ent_clusters)

        # sample names for entities
        orig_names = rng.choice(names, size=n_ents, replace=False).tolist()
        ent_ments = [
            [(name, "NNP", "(" + str(cluster) + ")")]
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
        knowledge_sents = create_knowledge_sents(rng, entities, occ2desc, **params)

        # sample location for entities to meet
        location_str, noise_fp = rng.choice(locations)
        location = eval(location_str)

        # create task text
        rng.shuffle(entities)
        task_sents, ms_temp, ts_temp = create_task_sents(
            rng,
            template,
            entities,
            location,
            pronoun,
            pronoun_cluster,
            occ2desc,
            noise_fp,
            **params
        )

        # append texts
        knowledge_texts.append(knowledge_sents)
        task_texts.append(task_sents)

    return knowledge_texts, task_texts


def create_knowledge_sents(rng, entities, occ2desc, add_background=False, **kwargs):
    """Create a knowledge text for one example."""

    # create is-a fact sents linking entities to their occupations
    knowledge_sents = []

    for entity in entities:
        knowledge_sent = entity["mention"] + IS_A

        # check for a/an
        if entity["occupation"][0][0][0] in {"a", "e", "i", "o", "u"}:
            knowledge_sent[-1] = AN

        knowledge_sent += list(entity["occupation"]) + [(".", ".")]
        knowledge_sents.append(knowledge_sent)

    # add background knowledge
    if add_background:
        for entity in entities:
            knowledge_sent = list(THE_WORK_OF_A)

            # check for a/an
            if entity["occupation"][0][0][0] in {"a", "e", "i", "o", "u"}:
                knowledge_sent[-1] = AN

            knowledge_sent += (
                list(entity["occupation"])
                + [("is", "VBZ")]
                + occ2desc[entity["occupation"]]
                + [(".", ".")]
            )
            knowledge_sents.append(knowledge_sent)

    # capitalize first word of each sent
    for sent in knowledge_sents:
        sent[0] = tuple([sent[0][0].capitalize()] + list(sent[0][1:]))

    # shuffle all sentences
    rng.shuffle(knowledge_sents)

    # convert to tuples
    knowledge_sents = tuple(map(tuple, knowledge_sents))

    return knowledge_sents


def create_task_sents(
    rng,
    template,
    entities,
    location,
    pronoun,
    pronoun_cluster,
    occ2desc,
    noise_fp,
    add_noise=True,
    **kwargs
):
    """Create a task text for one example."""

    # determine mentions
    if len(entities) == 2:
        a, b = entities
        mentions = a["mention"] + [("and", "CC")] + b["mention"]

    elif len(entities) == 3:
        a, b, c = entities
        mentions = (
            a["mention"] + [(",", ",")] + b["mention"] + [(",", ","), ("and", "CC")] + c["mention"]
        )

    elif len(entities) == 4:
        a, b, c, d = entities
        mentions = (
            a["mention"]
            + [(",", ",")]
            + b["mention"]
            + [(",", ",")]
            + c["mention"]
            + [(",", ","), ("and", "CC")]
            + d["mention"]
        )

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
    be = ("were", "VBD") if pronoun == "they" else ("was", "VBD")

    # determine pronoun
    pronoun = (pronoun, "PRP", "(" + str(pronoun_cluster) + ")")

    # fill template
    task_sents = map(eval, map(eval, template))

    # filter empty sents
    task_sents = list(filter(len, task_sents))

    # capitalize first word of each sent
    for sent in task_sents:
        sent[0] = tuple([sent[0][0].capitalize()] + list(sent[0][1:]))

    # convert to tuples
    task_sents = tuple(map(tuple, task_sents))

    return task_sents, template[0], template[2]
