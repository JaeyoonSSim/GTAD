import torch
import numpy as np

def compute_accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()

    return correct / len(labels)


def confusion(output, labels, ovo=0):
    predict = output.max(1)[1].type_as(labels)
    n_label = output.shape[1]

    if ovo:
        accuracies, precision, specificity, sensitivity, f1_score = [np.zeros((n_label, n_label)) for _ in range(5)]

        for i in range(n_label):
            for j in range(i + 1, n_label):
                p0, p1 = (predict == i), (predict == j)
                l0, l1 = (labels == i), (labels == j)

                tp = torch.sum(p0 & l0).item()
                tn = torch.sum(p1 & l1).item()
                fp = torch.sum(p1 & l0).item()
                fn = torch.sum(p0 & l1).item()

                accuracies[i][j] = (tp + tn) / (tp + fp + tn + fn)
                precision[i][j] = tp / (tp + fp)
                specificity[i][j] = tn / (tn + fp)
                sensitivity[i][j] = tp / (tp + fn)
                f1_score[i][j] = (2 * precision[i][j] * sensitivity[i][j]) / (precision[i][j] + sensitivity[i][j])

        accuracies = accuracies + accuracies.T
        precision = precision + precision.T
        specificity = specificity + specificity.T
        sensitivity = sensitivity + sensitivity.T
        f1_score = f1_score + f1_score.T

        accuracies = accuracies[~np.eye(accuracies.shape[0], dtype=bool)].reshape(accuracies.shape[0], -1)
        precision = precision[~np.eye(precision.shape[0], dtype=bool)].reshape(precision.shape[0], -1)
        specificity = specificity[~np.eye(specificity.shape[0], dtype=bool)].reshape(specificity.shape[0], -1)
        sensitivity = sensitivity[~np.eye(sensitivity.shape[0], dtype=bool)].reshape(sensitivity.shape[0], -1)
        f1_score = f1_score[~np.eye(f1_score.shape[0], dtype=bool)].reshape(f1_score.shape[0], -1)
        
        accuracies = np.nanmean(accuracies, axis=1)
        precision = np.nanmean(precision, axis=1)
        specificity = np.nanmean(specificity, axis=1)
        sensitivity = np.nanmean(sensitivity, axis=1)
        f1_score = np.nanmean(f1_score, axis=1)
    else:
        accuracies, precision, specificity, sensitivity, f1_score = list([] for _ in range(5))

        for i in range(n_label):
            p0, p1 = (predict == i), (predict != i)
            l0, l1 = (labels == i), (labels != i)

            tp = torch.sum(p0 & l0).item()
            tn = torch.sum(p1 & l1).item()
            fp = torch.sum(p1 & l0).item()
            fn = torch.sum(p0 & l1).item()

            accuracies.append((tp + tn) / (tp + fp + tn + fn))

            if tp + fp:
                precision.append(tp / (tp + fp))

            if tn + fp:
                specificity.append(tn / (tn + fp))

            if tp + fn:
                sensitivity.append(tp / (tp + fn))
            
            try:
                f1_score.append((2 * (tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + (tp / (tp + fn))))
            except ZeroDivisionError:
                pass
            
    return np.mean(accuracies), np.mean(precision), np.mean(specificity), np.mean(sensitivity), np.mean(f1_score)