import numpy as np


def get_TP(target, prediction, threshold):
    target = 1 - np.clip(target, 0, threshold) / threshold
    prediction = 1 - np.clip(prediction, 0, threshold) / threshold

    tp_array = np.logical_and(target, prediction) * 1.0
    tp = np.sum(tp_array)

    return tp


def get_FP(target, prediction, threshold):
    target = 1 - np.clip(target, 0, threshold) / threshold
    prediction = 1 - np.clip(prediction, 0, threshold) / threshold

    fp_array = np.logical_and(target, prediction) * 1.0
    fp = np.sum(fp_array)

    return fp


def get_FN(target, prediction, threshold):
    target = 1 - np.clip(target, 0, threshold) / threshold
    prediction = 1 - np.clip(prediction, 0, threshold) / threshold

    fn_array = np.logical_and(target, prediction) * 1.0
    fn = np.sum(fn_array)

    return fn


def get_TN(target, prediction, threshold):
    target = 1 - np.clip(target, 0, threshold) / threshold
    prediction = 1 - np.clip(prediction, 0, threshold) / threshold

    tn_array = np.logical_and(target, prediction) * 1.0
    tn = np.sum(tn_array)

    return tn


def get_recall(target, prediction, threshold):
    tp = get_TP(target, prediction, threshold)
    fn = get_FN(target, prediction, threshold)
    print('tp={0}'.format(tp))
    print('fn={0}'.format(fn))
    if tp + fn <= 0.0:
        recall = tp / (tp + fn + 1e-9)
    else:
        recall = tp / (tp + fn)
    return recall


def get_precision(target, prediction, threshold):
    tp = get_TP(target, prediction, threshold)
    fp = get_FP(target, prediction, threshold)
    print('tp={0}'.format(tp))
    print('fp={0}'.format(fp))
    if tp + fp <= 0.0:
        precision = tp / (tp + fp + 1e-9)
    else:
        precision = tp / (tp + fp)
    return precision


def get_F1(target, prediction, threshold):
    recall = get_recall(target, prediction, threshold)
    print(recall)
    precision = get_precision(target, prediction, threshold)
    print(precision)
    if precision == 0.0 or recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1


def get_accuracy(target, prediction, threshold):
    tp = get_TP(target, prediction, threshold)
    tn = get_TN(target, prediction, threshold)
    accuracy = (tp + tn) / target.size
    return accuracy


def get_relative_error(target, prediction):
    return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))


def get_abs_error(target, prediction):
    return np.mean(np.abs(target - prediction))


def get_nde(target, prediction):
    return np.sum((target - prediction) ** 2) / np.sum((target ** 2))


def get_sae(target, prediction, sample_second):
    r = np.sum(target * sample_second * 1.0 / 3600.0)
    rhat = np.sum(prediction * sample_second * 1.0 / 3600.0)
    return np.abs(r - rhat) / np.abs(r)


def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn


def recall_precision_accuracy_f1(pred, ground, threshold):
    pr = np.array([0 if p < threshold else 1 for p in pred])
    gr = np.array([0 if p < threshold else 1 for p in ground])

    tp, tn, fp, fn = tp_tn_fp_fn(pr, gr)
    p = np.sum(pr)
    n = len(pr) - p

    res_recall = recall(tp, fn)
    res_precision = precision(tp, fp)
    res_f1 = f1(res_precision, res_recall)
    res_accuracy = accuracy(tp, tn, p, n)

    return res_recall, res_precision, res_accuracy, res_f1


def relative_error_total_energy(pred, ground):
    E_pred = np.sum(pred)
    E_ground = np.sum(ground)
    return np.abs(E_pred - E_ground) / float(max(E_pred, E_ground))


def mean_absolute_error(pred, ground):
    total_sum = np.sum(np.abs(pred - ground))
    return total_sum / len(pred)


def recall(tp, fn):
    return tp / float(tp + fn)


def precision(tp, fp):
    return tp / float(tp + fp)


def f1(prec, rec):
    return 2 * (prec * rec) / float(prec + rec)


def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)