from sklearn import metrics


def accuracy_score(y, y_hat, axis=1):
    if y.ndim > 1:
        acc = []
        for j in range(y.shape[axis]):
            acc.append(metrics.accuracy_score(y[:, j], y_hat))
    else:
        acc = metrics.accuracy_score(y, y_hat)
    return acc
