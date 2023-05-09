import json
import os

import numpy as np
import pandas as pd


def load_resources(resources_dir):
    # load names
    names = pd.read_csv(os.path.join(resources_dir, "names.csv"), usecols=["lastname"])[
        "lastname"
    ].values
    # load locations
    locations = pd.read_csv(os.path.join(resources_dir, "locations.csv"))
    locations["noise_fp"] = locations["noise_fp"].apply(lambda p: os.path.join(resources_dir, p))
    locations = locations.values

    # load templates
    templates_dir = os.path.join(resources_dir, "templates")
    with open(os.path.join(templates_dir, "meet_sentence.txt"), "r", encoding="utf-8") as fh:
        meet_sents = np.asarray(fh.read().strip().split("\n\n"))

    with open(os.path.join(templates_dir, "pronoun_sentence.txt"), "r", encoding="utf-8") as fh:
        pronoun_sents = np.asarray(fh.read().strip().split("\n\n"))

    with open(
        os.path.join(templates_dir, "entspec_knowledge_sentence.txt"), "r", encoding="utf-8"
    ) as fh:
        entspec_sents = np.asarray(fh.read().strip().split("\n\n"))

    with open(
        os.path.join(templates_dir, "background_knowledge_sentence.txt"), "r", encoding="utf-8"
    ) as fh:
        background_sents = np.asarray(fh.read().strip().split("\n\n"))

    # load vocab
    with open(os.path.join(resources_dir, "vocab.json"), "r", encoding="utf-8") as fh:
        vocab = {k: str(tuple(v)) for k, v in json.load(fh).items()}

    # load pronouns
    with open(os.path.join(resources_dir, "pronouns.json"), "r", encoding="utf-8") as fh:
        pronouns = [pronoun for pronoun in json.load(fh)["pronouns"]]

    splittable_resources = {
        "locations": locations,
        "names": names,
        "background_sents": background_sents,
        "entspec_sents": entspec_sents,
        "meet_sents": meet_sents,
        "pronoun_sents": pronoun_sents,
    }

    return splittable_resources, vocab, pronouns


def check_overlap(splits):
    """Checks for overlap in examples between splits."""

    train, validation, test = map(set, splits)

    assert len(train.intersection(validation)) == 0, "Overlap between train and validation set!"
    assert len(train.intersection(test)) == 0, "Overlap between train and test set!"
    assert len(validation.intersection(test)) == 0, "Overlap between validation and test set!"


def export(texts, data_path):
    """Export text lists to raw txt, conll, jsonlines, and gap format."""

    raw_lines = create_raw_lines(texts, data_path)
    conll_lines = create_conll_lines(texts, data_path)
    json_lines = create_json_lines(texts, data_path)

    with open(data_path + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(raw_lines))

    with open(data_path + ".v4_gold_conll", "w", encoding="utf-8") as fh:
        fh.write("\n".join(conll_lines))

    with open(data_path + ".jsonlines", "w", encoding="utf-8") as fh:
        fh.write("\n".join(json_lines))

    # create gap format file only if there are annotated pronouns and exactly two antecedents
    if "knowledge-text-only" not in data_path and "2_ents" in data_path:
        gap_df = create_gap_df(texts, data_path)
        gap_df.to_csv(data_path + ".tsv", index=False, sep="\t")


def create_raw_lines(texts, data_path):
    """Create raw lines from text lists."""

    lines = []

    for text_ix, text in enumerate(texts):
        text_id = data_path + ":" + str(text_ix)

        lines.append(text_id)

        tokens = [token_tup[0] for sent in text for token_tup in sent]
        text_str = tokens2str(tokens)[0]

        lines.append(text_str)
        lines.append("")

    return lines


def create_conll_lines(texts, data_path):
    """Create conll lines from text lists."""

    lines = []

    for text_ix, text in enumerate(texts):
        text_id = data_path + ":" + str(text_ix)

        lines.append("#begin document (" + text_id + "); part 000")

        for sent in text:
            for token_ix, token_tup in enumerate(sent):
                if len(token_tup) == 3:
                    token, pos, cluster = token_tup
                elif len(token_tup) == 2:
                    (token, pos), cluster = token_tup, "-"
                else:
                    raise Exception(f"Tuple with wrong length found! {token_tup}")

                line = text_id
                line += (100 - (len(line) + len("0"))) * " " + "0"
                line += (106 - (len(line) + len(str(token_ix)))) * " " + str(token_ix)
                line += (124 - (len(line) + len(token))) * " " + token
                line += (134 - (len(line) + len(pos))) * " " + pos
                line += "    *    -    -    -    Speaker#1    *    "
                line += cluster

                assert (
                    len(list(filter(len, map(lambda s: s.strip(), line.split(" "))))) == 12
                ), f"Line has wrong field number: {line}"

                lines.append(line)

            lines.append("")

        lines.append("#end document")

    return lines


def create_json_lines(texts, data_path):
    """Create json lines from text lists."""

    lines = []

    for text_ix, text in enumerate(texts):
        text_id = data_path + ":" + str(text_ix)

        sentences, speakers, clusters = [], [], {}

        token_ix = 0

        for sent in text:
            sent_tokens, sent_speakers = [], []

            for token_tup in sent:
                sent_tokens.append(token_tup[0])
                sent_speakers.append("Speaker#1")

                if len(token_tup) == 3:
                    cluster = int(token_tup[2].strip("()"))

                    if cluster in clusters:
                        clusters[cluster] += [token_ix]
                    else:
                        clusters[cluster] = [token_ix]

                token_ix += 1

            sentences.append(sent_tokens)
            speakers.append(sent_speakers)

        clusters = [unrange(clusters[cluster]) for cluster in sorted(clusters.keys())]

        line = json.dumps(
            {"doc_key": text_id, "sentences": sentences, "speakers": speakers, "clusters": clusters}
        )

        lines.append(line)

    return lines


def create_gap_df(texts, data_path):
    """Create gap dataframe from text lists."""

    min_sent_ix = data_path2min_sent_ix(data_path)

    rows = []

    for text_ix, text in enumerate(texts):
        text_id = data_path + ":" + str(text_ix)

        tokens = [token_tup[0] for sent in text for token_tup in sent]
        text_str, chars = tokens2str(tokens)

        clusters = dict()
        text_token_ix = 0

        for sent_ix, sent in enumerate(text):
            for token_tup in sent:
                # min_sent_ix is used to avoid accumulating mentions from the knowledge text
                if len(token_tup) == 3 and sent_ix >= min_sent_ix:
                    _, token_pos, cluster = token_tup

                    token_cluster_id = int(cluster.strip("()"))
                    token_is_pronoun = token_pos == "PRP"

                    if token_cluster_id in clusters:
                        clusters[token_cluster_id]["is_pronoun"] += [token_is_pronoun]
                        clusters[token_cluster_id]["token_ics"] += [text_token_ix]
                    else:
                        clusters[token_cluster_id] = {
                            "is_pronoun": [token_is_pronoun],
                            "token_ics": [text_token_ix],
                        }

                text_token_ix += 1

        # calculate mentions from clusters
        mentions = clusters2mentions(clusters)

        # extract pronoun str and start char
        pronoun_start_token_ix, pronoun_end_token_ix = mentions["pronoun"]
        pronoun_str = " ".join(tokens[pronoun_start_token_ix : pronoun_end_token_ix + 1])
        pronoun_start_char = chars[pronoun_start_token_ix]

        # sanity check
        assert text_str[pronoun_start_char:].startswith(
            pronoun_str
        ), "Pronoun start char is not correct!"

        # zip antecedents together with coref truth values
        antecedents = mentions["incorrect_antecedents"] + [mentions["correct_antecedent"]]
        corefs = [False] * len(mentions["incorrect_antecedents"]) + [True]
        antecedents_with_corefs = list(zip(antecedents, corefs))

        # sanity check
        assert (
            len(antecedents_with_corefs) == 2
        ), "The GAP format can only list two antecedents, but more are given!"

        # extract antecedents and coref truth values
        (a_token_ics, a_coref), (b_token_ics, b_coref) = antecedents_with_corefs

        # extract token start and end ics
        a_start_token_ix, a_end_token_ix = a_token_ics
        b_start_token_ix, b_end_token_ix = b_token_ics

        # extract start chars
        a_start_char = chars[a_start_token_ix]
        b_start_char = chars[b_start_token_ix]

        # swap if b occurs earlier than a
        if b_start_char < a_start_char:
            a_coref, b_coref = b_coref, a_coref
            a_start_char, b_start_char = b_start_char, a_start_char
            a_start_token_ix, b_start_token_ix = b_start_token_ix, a_start_token_ix
            a_end_token_ix, b_end_token_ix = b_end_token_ix, a_end_token_ix

        # extract str
        a_str = " ".join(tokens[a_start_token_ix : a_end_token_ix + 1])
        b_str = " ".join(tokens[b_start_token_ix : b_end_token_ix + 1])

        # sanity checks
        assert text_str[a_start_char:].startswith(a_str), "A start char is not correct!"
        assert text_str[b_start_char:].startswith(b_str), "B start char is not correct!"

        # convert bools to strs
        a_coref_str = "TRUE" if a_coref else "FALSE"
        b_coref_str = "TRUE" if b_coref else "FALSE"

        # append example
        rows.append(
            {
                "ID": text_id,
                "Text": text_str,
                "Pronoun": pronoun_str,
                "Pronoun-offset": pronoun_start_char,
                "A": a_str,
                "A-offset": a_start_char,
                "A-coref": a_coref_str,
                "B": b_str,
                "B-offset": b_start_char,
                "B-coref": b_coref_str,
                "URL": "none",
            }
        )

    return pd.DataFrame(rows)


def unrange(token_ics):
    """Bundle consecutive indices together to sublists, then select only first index a and last index b in each sublist. Effectively this is the inverse of range(a, b + 1)."""

    unranged = [[token_ics[0]]]

    for token_ix in token_ics[1:]:
        if token_ix - 1 in unranged[-1]:
            unranged[-1].append(token_ix)
        else:
            unranged.append([token_ix])

    unranged = [[sublist[0], sublist[-1]] for sublist in unranged]

    return unranged


def tokens2str(tokens):
    """Converts list of tokens to a single str line."""

    line = ""
    chars = [0]

    start_char = 0
    open_quote = False

    for token in tokens:
        if (
            token in {".", ",", ")", "/", "!", ";"}
            or token.startswith("'")
            or (token == '"' and open_quote)
        ):
            line = line.rstrip(" ")
            start_char -= 1

        line += token + " "
        start_char += len(token) + 1

        if token in {"(", "/"} or (token == '"' and not open_quote):
            line = line.rstrip(" ")
            start_char -= 1

        chars.append(start_char)

        if token == '"':
            open_quote = not open_quote

    line = line.rstrip(" ")
    chars = chars[:-1]

    return line, chars


def clusters2mentions(clusters):
    """Convert cluster dict with is_pronoun attribute to meaningful mentions dict."""

    clusters_with_pronouns = [
        cluster for cluster in clusters.values() if any(cluster["is_pronoun"])
    ]
    clusters_without_pronouns = [
        cluster for cluster in clusters.values() if not any(cluster["is_pronoun"])
    ]

    assert len(clusters_with_pronouns) == 1, "There is not exactly one cluster involving a pronoun"
    assert (
        sum(clusters_with_pronouns[0]["is_pronoun"]) == 1
    ), "There is more than one pronoun token in the cluster"
    assert clusters_with_pronouns[0]["is_pronoun"][
        -1
    ], "The pronoun is not the last token in the cluster"

    pronoun_cluster_mentions = unrange(clusters_with_pronouns[0]["token_ics"])
    assert (
        len(pronoun_cluster_mentions) == 2
    ), "There are not exactly two mentions in the pronoun cluster"

    correct_antecedent_mention, pronoun_mention = map(tuple, pronoun_cluster_mentions)
    incorrect_antecedent_mentions = [
        tuple(mention)
        for cluster in clusters_without_pronouns
        for mention in unrange(cluster["token_ics"])
    ]

    mentions = {
        "pronoun": pronoun_mention,
        "incorrect_antecedents": incorrect_antecedent_mentions,
        "correct_antecedent": correct_antecedent_mention,
    }

    return mentions


def data_path2min_sent_ix(data_path):
    """Determine first sentence index that belongs to task text in list of sentences."""

    if "full-text" in data_path:
        n_entities = int(data_path.split("_")[1])

        if (
            "background-inference" in data_path
            or "background-both" in data_path
            or "desc" in data_path
        ):
            min_sent_ix = n_entities * 2
        else:
            min_sent_ix = n_entities

    else:
        min_sent_ix = 0

    return min_sent_ix


def char_ics2token_ics(start_char, end_char, chars):
    """Convert character indices to token indices."""

    assert start_char in chars, "Start char not found in gold chars list!"
    start_token_ix = chars.index(start_char)

    end_token_ix = None

    for token_ix, char_ix in enumerate(chars):
        if char_ix > end_char:
            break

        end_token_ix = token_ix

    assert end_token_ix is not None, "End char smaller than all chars in gold chars list!"

    return (start_token_ix, end_token_ix)
