import numpy as np

SMALL_POSITIVE_CONST = 1e-4

# slightly modified from the original code, for fitting the current codebase
def evaluate_ote(gold_ot, pred_ot):
    """
    evaluate the model performce for the ote task
    :param gold_ot: gold standard ote tags
    :param pred_ot: predicted ote tags
    :return:
    """
    assert len(gold_ot) == len(pred_ot)
    n_samples = len(gold_ot)
    # number of true positive, gold standard, predicted opinion targets
    n_tp_ot, n_gold_ot, n_pred_ot = 0, 0, 0
    for i in range(n_samples):
        n_hit_ot = 0
        g_ot = gold_ot[i]
        p_ot = pred_ot[i]
        # hit number 
        for j, t in enumerate(p_ot):
            # count the number of correctly predicted opinion targets
            if t == g_ot[j] and t == 2: 
                n_hit_ot += 1
        n_tp_ot += n_hit_ot
        # count the number of aspects in g_ot
        n_gold_ot += sum([1 for t in g_ot if t == 2])
        # count the number of aspects in p_ot
        n_pred_ot += sum([1 for t in p_ot if t == 2])
    # add 0.001 for smoothing
    # calculate precision, recall and f1 for ote task
    ot_precision = float(n_tp_ot) / float(n_pred_ot + SMALL_POSITIVE_CONST)
    ot_recall = float(n_tp_ot) / float(n_gold_ot + SMALL_POSITIVE_CONST)
    ot_f1 = 2 * ot_precision * ot_recall / (ot_precision + ot_recall + SMALL_POSITIVE_CONST)
    ote_scores = (ot_precision, ot_recall, ot_f1)
    return ote_scores


def evaluate_ts(gold_ts, pred_ts):
    """
    evaluate the model performance for the ts task
    :param gold_ts: gold standard ts tags
    :param pred_ts: predicted ts tags
    :return:
    """
    assert len(gold_ts) == len(pred_ts)
    n_samples = len(gold_ts)
    # number of true postive, gold standard, predicted targeted sentiment
    n_tp_ts, n_gold_ts, n_pred_ts = np.zeros(3), np.zeros(3), np.zeros(3)
    ts_precision, ts_recall, ts_f1 = np.zeros(3), np.zeros(3), np.zeros(3)

    for i in range(n_samples):
        g_ts = gold_ts[i]
        p_ts = pred_ts[i]
        g_ts_sequence, p_ts_sequence = tag2ts(ts_tag_sequence=g_ts), tag2ts(ts_tag_sequence=p_ts)
        hit_ts_count, gold_ts_count, pred_ts_count = match_ts(gold_ts_sequence=g_ts_sequence,
                                                              pred_ts_sequence=p_ts_sequence)

        n_tp_ts += hit_ts_count
        n_gold_ts += gold_ts_count
        n_pred_ts += pred_ts_count
        # calculate macro-average scores for ts task
    for i in range(3):
        n_ts = n_tp_ts[i]
        n_g_ts = n_gold_ts[i]
        n_p_ts = n_pred_ts[i]
        ts_precision[i] = float(n_ts) / float(n_p_ts + SMALL_POSITIVE_CONST)
        ts_recall[i] = float(n_ts) / float(n_g_ts + SMALL_POSITIVE_CONST)
        ts_f1[i] = 2 * ts_precision[i] * ts_recall[i] / (ts_precision[i] + ts_recall[i] + SMALL_POSITIVE_CONST)

    ts_macro_f1 = ts_f1.mean()

    # calculate micro-average scores for ts task
    n_tp_total = sum(n_tp_ts)
    # total sum of TP and FN
    n_g_total = sum(n_gold_ts)
    # total sum of TP and FP
    n_p_total = sum(n_pred_ts)

    ts_micro_p = float(n_tp_total) / (n_p_total + SMALL_POSITIVE_CONST)
    ts_micro_r = float(n_tp_total) / (n_g_total + SMALL_POSITIVE_CONST)
    ts_micro_f1 = 2 * ts_micro_p * ts_micro_r / (ts_micro_p + ts_micro_r + SMALL_POSITIVE_CONST)
    ts_scores = (ts_macro_f1, ts_micro_p, ts_micro_r, ts_micro_f1)
    return ts_scores


def evaluate(gold_ot, gold_ts, pred_ot, pred_ts):
    """
    evaluate the performance of the predictions
    :param gold_ot: gold standard opinion target tags
    :param gold_ts: gold standard targeted sentiment tags
    :param pred_ot: predicted opinion target tags
    :param pred_ts: predicted targeted sentiment tags
    :return: metric scores of ner and sa
    """
    assert len(gold_ot) == len(gold_ts) == len(pred_ot) == len(pred_ts)
    ote_scores = evaluate_ote(gold_ot=gold_ot, pred_ot=pred_ot)
    ts_scores = evaluate_ts(gold_ts=gold_ts, pred_ts=pred_ts)
    return ote_scores, ts_scores


def match_ot(gold_ote_sequence, pred_ote_sequence):
    """
    calculate the number of correctly predicted opinion target
    :param gold_ote_sequence: gold standard opinion target sequence
    :param pred_ote_sequence: predicted opinion target sequence
    :return: matched number
    """
    n_hit = 0
    for t in pred_ote_sequence:
        if t in gold_ote_sequence:
            n_hit += 1
    return n_hit


def match_ts(gold_ts_sequence, pred_ts_sequence):
    """
    calculate the number of correctly predicted targeted sentiment
    :param gold_ts_sequence: gold standard targeted sentiment sequence
    :param pred_ts_sequence: predicted targeted sentiment sequence
    :return:
    """
    # positive, negative and neutral
    tag2tagid = {'POS': 0, 'NEG': 1, 'NEU': 2}
    hit_count, gold_count, pred_count = np.zeros(3), np.zeros(3), np.zeros(3)
    for t in gold_ts_sequence:
        #print(t)
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        gold_count[tid] += 1
    for t in pred_ts_sequence:
        ts_tag = t[2]
        tid = tag2tagid[ts_tag]
        if t in gold_ts_sequence:
            hit_count[tid] += 1
        pred_count[tid] += 1
    return hit_count, gold_count, pred_count

def tag2ot(ote_tag_sequence):
    """
    transform ote tag sequence to a sequence of opinion target
    :param ote_tag_sequence: tag sequence for ote task
    :return:
    """
    n_tags = len(ote_tag_sequence)
    ot_sequence = []
    beg, end = -1, -1
    for i in range(n_tags):
        tag = ote_tag_sequence[i]
        if tag == 'S':
            ot_sequence.append((i, i))
        elif tag == 'B':
            beg = i
        elif tag == 'E':
            end = i
            if end > beg > -1:
                ot_sequence.append((beg, end))
                beg, end = -1, -1
    return ot_sequence

def tag2ts(ts_tag_sequence):
    """
    transform ts tag sequence to targeted sentiment
    :param ts_tag_sequence: tag sequence for ts task
    :return:
    """
    n_tags = len(ts_tag_sequence)
    ts_sequence, sentiments = [], []
    beg, end = -1, -1
    for i in range(n_tags):
        ts_tag = ts_tag_sequence[i]
        # current position and sentiment
        eles = ts_tag.split('-')
        if len(eles) == 2:
            pos, sentiment = eles
        else:
            pos, sentiment = 'O', 'O'
        if sentiment != 'O':
            # current word is a subjective word
            sentiments.append(sentiment)
        if pos == 'S':
            # singleton
            ts_sequence.append((i, i, sentiment))
            sentiments = []
        elif pos == 'B':
            beg = i
        elif pos == 'E':
            end = i
            # schema1: only the consistent sentiment tags are accepted
            # that is, all of the sentiment tags are the same
            if end > beg > -1 and len(set(sentiments)) == 1:
                ts_sequence.append((beg, end, sentiment))
                sentiments = []
                beg, end = -1, -1
    return ts_sequence