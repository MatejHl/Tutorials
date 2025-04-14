# Plot MCC as a function of accuracy per class

import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def MCC_per_class_accuracy(acc_0, acc_1, N_0, N_1):
    y_pred=[numpy.zeros(shape = (int(numpy.round(       acc_0 *N_0,)))), 
            numpy.ones( shape = (int(numpy.round((1.0 - acc_0)*N_0,)))), 
            numpy.zeros(shape = (int(numpy.round((1.0 - acc_1)*N_1,)))), 
            numpy.ones( shape = (int(numpy.round(       acc_1 *N_1,)))),
            ]
    
    y_true=[numpy.zeros(shape = (N_0,)), 
            numpy.ones( shape = (N_1,)),
            ]
    y_pred = numpy.concatenate(y_pred)
    y_true = numpy.concatenate(y_true)

    mcc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)
    return mcc

def func(ACC_0, ACC_1):
    mcc_results = []
    n_i, n_j = ACC_0.shape
    for i in range(n_i):
        _res = []
        for j in range(n_j):
            acc_0 = ACC_0[i, j]
            acc_1 = ACC_1[i, j]
            MCC = MCC_per_class_accuracy(acc_0, acc_1, N_0 = N_0, N_1 = N_1)
            # print('Acc 0: {} \tAcc 1: {} \tMCC: {}'.format(acc_0, acc_1, MCC))
            _res.append(MCC)
        mcc_results.append(numpy.array(_res))
    return numpy.stack(mcc_results, axis = 0)

N_0 = 1000
N_1 = 10000
# acc_0 = 0.9
# acc_1 = 0.95

# mcc_results = []
# _acc_0 = numpy.arange(0, 1, 0.05)
# for acc_0 in _acc_0:
#     MCC = MCC_per_class_accuracy(acc_0, acc_1, N_0 = N_0, N_1 = N_1)
#     print('Acc 0: {} \tAcc 1: {} \tMCC: {}'.format(acc_0, acc_1, MCC))
#     mcc_results.append(MCC)
# 
# plt.plot(_acc_0, mcc_results)
# plt.show()

x = numpy.linspace(0.5, 1.0, 80)
y = numpy.linspace(0.5, 1.0, 80)

X, Y = numpy.meshgrid(x, y)
Z = func(X,Y)

fig = plt.figure()
if False:
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 50, cmap='binary')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('3D contour')
else:
    ax = plt.axes()
    ax.contourf(X, Y, Z, levels = 20)
    def fmt(x, y):
        z = numpy.take(MCC_per_class_accuracy(x, y, N_0 = N_0, N_1 = N_1), 0)
        return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)
    ax.format_coord = fmt
plt.show()




# CM = confusion_matrix(y_true = y_true, y_pred = y_pred)
# tn = CM[0,0]
# fp = CM[0,1]
# tp = CM[1,1]
# fn = CM[1,0]
# acc_0 = tn/(tn+fp)
# acc_1 = tp/(tp+fn)
# print('Acc 0: {}, Acc 1: {}'.format(acc_0, acc_1))







