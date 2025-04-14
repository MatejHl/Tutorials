import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import matthews_corrcoef, confusion_matrix


n_true = [2789, 14385, 6026]

y_true_0 = 0 * numpy.ones(shape = (2789,))
y_true_1 = 1 * numpy.ones(shape = (14385,))
y_true_2 = 2 * numpy.ones(shape = (6026,))

y_true = numpy.concatenate([y_true_0, y_true_1, y_true_2])


n_pred = [3376, 13260, 6564]

y_pred_0_0 = 0 * numpy.ones(shape = (2448,))
y_pred_1_0 = 1 * numpy.ones(shape = (178,))
y_pred_2_0 = 2 * numpy.ones(shape = (163,))

y_pred_0_1 = 0 * numpy.ones(shape = (648,))
y_pred_1_1 = 1 * numpy.ones(shape = (12112,))
y_pred_2_1 = 2 * numpy.ones(shape = (1625,))

y_pred_0_2 = 0 * numpy.ones(shape = (280,))
y_pred_1_2 = 1 * numpy.ones(shape = (970,))
y_pred_2_2 = 2 * numpy.ones(shape = (4776,))

y_pred = numpy.concatenate([y_pred_0_0, y_pred_1_0, y_pred_2_0, 
                            y_pred_0_1, y_pred_1_1, y_pred_2_1,
                            y_pred_0_2, y_pred_1_2, y_pred_2_2])


mcc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)

CM = confusion_matrix(y_true = y_true, y_pred = y_pred)

print(CM)
print(mcc)

n_true = numpy.sum(CM, axis = 1)
y_true = numpy.concatenate([i*numpy.ones(shape = (n)) for i, n in enumerate(n_true)])
tmp = []
for i in range(CM.shape[0]):
    for j in range(CM.shape[1]):
        tmp.append(j*numpy.ones(shape = (CM[i,j],)))
y_pred = numpy.concatenate(tmp)

mcc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)
CM = confusion_matrix(y_true = y_true, y_pred = y_pred)

print(CM)
print(mcc)

# ----------------------------
# ----------------------------
def _test(y_true, y_pred):
    C = confusion_matrix(y_true, y_pred)
    t_sum = C.sum(axis=1, dtype=numpy.float64)
    p_sum = C.sum(axis=0, dtype=numpy.float64)
    n_correct = numpy.trace(C, dtype=numpy.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - numpy.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - numpy.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - numpy.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / numpy.sqrt(cov_ytyt * cov_ypyp)

def _test_with_conf_matrix(C):
    # C = confusion_matrix(y_true, y_pred)
    t_sum = C.sum(axis=1, dtype=numpy.float64)
    p_sum = C.sum(axis=0, dtype=numpy.float64)
    n_correct = numpy.trace(C, dtype=numpy.float64)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - numpy.dot(t_sum, p_sum)
    cov_ypyp = n_samples**2 - numpy.dot(p_sum, p_sum)
    cov_ytyt = n_samples**2 - numpy.dot(t_sum, t_sum)
    if cov_ypyp * cov_ytyt == 0:
        return 0.0
    else:
        return cov_ytyp / numpy.sqrt(cov_ytyt * cov_ypyp)

mcc = _test(y_true = y_true, y_pred = y_pred)
CM = confusion_matrix(y_true = y_true, y_pred = y_pred)

print(y_true)
print(y_pred)

print(CM)
print(mcc)

CM = confusion_matrix(y_true = y_true, y_pred = y_pred)
mcc = _test_with_conf_matrix(CM)

print(CM)
print(mcc)