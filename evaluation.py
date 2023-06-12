from thefuzz import process


def dedupe_keyword(keywords, threshold=85):
    return list(process.dedupe(keywords, threshold=threshold))


def calculate_precision_recall_f1(predicted, groundtruth):
    predicted_set = set(predicted)
    true_set = set(groundtruth)

    true_positive = len(predicted_set.intersection(true_set))

    # precision
    if len(predicted_set) == 0:
        precision = 0
    else:
        precision = true_positive / len(predicted_set)

    # recall
    if len(true_set) == 0:
        recall = 0
    else:
        recall = true_positive / len(true_set)

    # f1-score
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    precision = round(precision, 6)
    recall = round(recall, 6)
    f1 = round(f1, 6)

    return (precision, recall, f1)
