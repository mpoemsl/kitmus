import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import char_ics2token_ics, clusters2mentions, data_path2min_sent_ix, tokens2str

parser = argparse.ArgumentParser(
    description="Evaluates predictions in a .jsonlines or GAP-schema .tsv file against a gold .conll file"
)

parser.add_argument("gold_fp", help="Path to gold .v4_gold_conll file", type=str)
parser.add_argument("prediction_fp", help="Path to predicted .jsonlines or .tsv file", type=str)


def main(gold_fp="", prediction_fp="", print_ids=False, rand=False):
    gold_corefs, text_tokens = read_conll(gold_fp)

    if prediction_fp.endswith(".jsonlines"):
        pred_corefs = read_jsonlines(prediction_fp)
    elif prediction_fp.endswith(".tsv"):
        pred_corefs = read_gap(prediction_fp, text_tokens)
    else:
        raise TypeError("Unknown file ending for file {}!".format(prediction_fp))

    antecedent_f1, precision, recall = calculate_antecedent_f1(pred_corefs, gold_corefs)
    print(
        "Antecedent F1 is {:.3f} with precision {:.3f} and recall {:.3f}.".format(
            antecedent_f1, precision, recall
        )
    )

    text_acc, correct_ids = calculate_text_accuracy(pred_corefs, gold_corefs)
    print("Text accuracy is {:.3f}.".format(text_acc))


def calculate_antecedent_f1(pred_corefs, gold_corefs):
    """Calculate F1 score over boolean prediction of links between and pronoun and each candidate antecedent."""

    antecedent_pronoun_preds = []
    antecedent_pronoun_golds = []

    for text_id, gold_coref in gold_corefs.items():
        assert text_id in pred_corefs, "No prediction found for text id {}!".format(text_id)

        # gather candidate antecedents
        candidate_antecedents = gold_coref["incorrect_antecedents"] + [
            gold_coref["correct_antecedent"]
        ]

        # calculate gold antecedent pronoun coreference booleans
        antecedent_pronoun_golds += [False] * len(gold_coref["incorrect_antecedents"]) + [True]

        # calculate predicted antecedent pronoun coreference booleans
        for candidate_antecedent in candidate_antecedents:
            antecedent_pronoun_pred = False

            for pred_cluster in pred_corefs[text_id]:
                if {candidate_antecedent, gold_coref["pronoun"]} <= set(pred_cluster):
                    antecedent_pronoun_pred = True

            antecedent_pronoun_preds.append(antecedent_pronoun_pred)

    assert len(antecedent_pronoun_preds) > 0 and len(antecedent_pronoun_golds) > 0

    # vectorize booleans
    antecedent_pronoun_preds = np.array(antecedent_pronoun_preds)
    antecedent_pronoun_golds = np.array(antecedent_pronoun_golds)

    # calculate precision, recall, and f1 score
    true_positives = antecedent_pronoun_preds & antecedent_pronoun_golds

    # calculate precision
    denom = antecedent_pronoun_preds.sum() if antecedent_pronoun_preds.sum() > 0 else 1e-8
    precision = true_positives.sum() / denom

    # calculate recall
    denom = antecedent_pronoun_golds.sum() if antecedent_pronoun_golds.sum() > 0 else 1e-8
    recall = true_positives.sum() / denom

    # calculate f1 score
    denom = precision + recall if precision + recall > 0 else 1e-8
    f1_score = (2 * precision * recall) / denom

    return f1_score, precision, recall


def calculate_text_accuracy(pred_corefs, gold_corefs):
    """Calculate accuracy over correctness of prediction for one whole text."""

    text_preds = []
    text_golds = []

    for text_id, gold_coref in gold_corefs.items():
        assert text_id in pred_corefs, "No prediction found for text id {}!".format(text_id)

        # gather candidate antecedents
        candidate_antecedents = gold_coref["incorrect_antecedents"] + [
            gold_coref["correct_antecedent"]
        ]

        # calculate gold vector for this text
        text_gold = np.array([False] * len(gold_coref["incorrect_antecedents"]) + [True])
        text_golds.append(text_gold)

        # calculate predicted vector for this text
        text_pred = []

        for candidate_antecedent in candidate_antecedents:
            antecedent_pronoun_pred = False

            for pred_cluster in pred_corefs[text_id]:
                if {candidate_antecedent, gold_coref["pronoun"]} <= set(pred_cluster):
                    antecedent_pronoun_pred = True

            text_pred.append(antecedent_pronoun_pred)

        text_pred = np.array(text_pred)
        text_preds.append(text_pred)

    # stack texts
    text_preds = np.stack(text_preds, axis=0)
    text_golds = np.stack(text_golds, axis=0)

    # determine correct instances
    correct_text_preds = np.all(text_preds == text_golds, axis=1)

    # calculate accuracy
    accuracy = correct_text_preds.sum() / len(gold_corefs)

    # determine correct ids
    correct_ids = pd.Series(data=gold_corefs.keys(), index=correct_text_preds)

    return accuracy, correct_ids


def read_jsonlines(prediction_fp):
    """Read jsonlines file and determine predicted coreference clusters."""

    with open(prediction_fp, "r") as fh:
        json_lines = fh.readlines()

    pred_corefs = dict()

    for line in tqdm(json_lines, desc=prediction_fp):
        json_dict = json.loads(line)
        text_id = json_dict["doc_key"]

        clusters = [list(map(tuple, cluster)) for cluster in json_dict["clusters"]]

        pred_corefs[text_id] = clusters

    return pred_corefs


def read_conll(gold_fp):
    """Read conll file and determine gold pronoun coreferent."""

    with open(gold_fp, "r") as fh:
        conll_lines = fh.readlines()

    # determine knowledge/task text boundary for merged setting
    min_sent_ix = data_path2min_sent_ix(conll_lines[0])

    # get dictionary of all clusters indices and whether they involve pronouns or not
    text_cluster_ics = dict()

    # get dictionary of all tokens for each text
    text_tokens = dict()

    prev_text_id = None
    text_token_ix, text_sent_ix = 0, 0

    for line in tqdm(conll_lines, desc=gold_fp):
        # skip blank lines
        if len(line) == 0 or line.isspace():
            text_sent_ix += 1
            continue

        # skip document pre- or post-amble lines
        if line.startswith("#"):
            continue

        # split relevant fields
        fields = [
            field.strip() for field in line.split(" ") if len(field) > 0 and not field.isspace()
        ]
        text_id, token_str, token_pos, cluster_str = fields[0], fields[3], fields[4], fields[11]

        # determine text transition
        if text_id == prev_text_id:
            text_token_ix += 1
        else:
            text_token_ix, text_sent_ix = 0, 0
            prev_text_id = text_id

        # collect tokens
        if text_id in text_tokens:
            text_tokens[text_id] += [token_str]
        else:
            text_tokens[text_id] = [token_str]

        # collect clusters if sent_ix is in task text segment
        if text_sent_ix >= min_sent_ix and cluster_str.strip("()").isnumeric():
            token_cluster_id = int(cluster_str.strip("()"))
            token_is_pronoun = token_pos == "PRP"

            if text_id in text_cluster_ics:
                if token_cluster_id in text_cluster_ics[text_id]:
                    text_cluster_ics[text_id][token_cluster_id]["is_pronoun"] += [token_is_pronoun]
                    text_cluster_ics[text_id][token_cluster_id]["token_ics"] += [text_token_ix]
                else:
                    text_cluster_ics[text_id][token_cluster_id] = {
                        "is_pronoun": [token_is_pronoun],
                        "token_ics": [text_token_ix],
                    }

            else:
                text_cluster_ics[text_id] = {
                    token_cluster_id: {
                        "is_pronoun": [token_is_pronoun],
                        "token_ics": [text_token_ix],
                    }
                }

    # get pronoun mention, correct, and incorrect antecedent mentions for each text
    gold_corefs = {
        text_id: clusters2mentions(clusters) for text_id, clusters in text_cluster_ics.items()
    }

    return gold_corefs, text_tokens


def read_gap(prediction_fp, text_tokens):
    """Read GAP-schema tsv file and determine predicted coreference clusters."""

    # load tsv file
    df = pd.read_csv(prediction_fp, sep="\t")

    # extract text id
    df.set_index(df["ID"], drop=True, inplace=True)

    # get char indices from tokens
    text_chars = {text: tokens2str(tokens)[1] for text, tokens in text_tokens.items()}

    pred_corefs = dict()

    # for each text id accumulate predictions
    for text_id, row in df.iterrows():
        # get char ics for this text
        assert text_id in text_chars, "No gold tokens found for prediction {}!".format(text_id)
        chars = text_chars[text_id]

        # get pronoun mention
        pronoun_start_char, pronoun_end_char = row["Pronoun-offset"], row["Pronoun-offset"] + len(
            row["Pronoun"]
        )
        pronoun_mention = char_ics2token_ics(pronoun_start_char, pronoun_end_char, chars)

        a_start_char, a_end_char = row["A-offset"], row["A-offset"] + len(row["A"])
        b_start_char, b_end_char = row["B-offset"], row["B-offset"] + len(row["B"])

        a_mention = char_ics2token_ics(a_start_char, a_end_char, chars)
        b_mention = char_ics2token_ics(b_start_char, b_end_char, chars)

        clusters = list()

        if row["A-coref"]:
            clusters.append([a_mention, pronoun_mention])

        if row["B-coref"]:
            clusters.append([b_mention, pronoun_mention])

        pred_corefs[text_id] = clusters

    return pred_corefs


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
