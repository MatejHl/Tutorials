# Plot MCC as a function of accuracy per class for multiclass classification

from this import d
import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import cm
from sklearn.metrics import matthews_corrcoef, confusion_matrix

def one_hot(x, num_classes):
    return numpy.identity(num_classes)[x]

def MCC_per_class_accuracy(acc, N, num_classes = 3):
    y_pred = []
    y_true = []

    for i in range(len(acc)):
        _correct = i*numpy.ones(shape = (int(numpy.round(acc[i]*N[i],))), dtype = int)
        _incorrect = numpy.random.randint(1,3, size=(int(numpy.round((1-acc[i])*N[i],))))
        _incorrect = numpy.mod(_incorrect + i, num_classes)
    
        _y_pred = numpy.concatenate([_correct, _incorrect], axis = 0)
        _y_true = i*numpy.ones(shape = (N[i],), dtype = int)
        y_pred.append(_y_pred)
        y_true.append(_y_true)

    y_pred = numpy.concatenate(y_pred, axis = 0)
    y_true = numpy.concatenate(y_true, axis = 0)

    mcc = matthews_corrcoef(y_true = y_true, y_pred = y_pred)
    return mcc



bla = MCC_per_class_accuracy(acc = [0.9295, 0.8397, 0.7931], N = [40236, 208862, 88763])
print(bla)
print('--------')
bla = MCC_per_class_accuracy(acc = [0.8882, 0.8467, 0.7567], N = [40236, 208862, 88763])
print(bla)
print('--------')
bla = MCC_per_class_accuracy(acc = [0.9018, 0.8573, 0.7471], N = [40236, 208862, 88763])
print(bla)
print('--------')

print('--------')
bla = MCC_per_class_accuracy(acc = [0.9439, 0.8590, 0.8290], N = [40236, 208862, 88763])
print(bla)
print('--------')
bla = MCC_per_class_accuracy(acc = [0.9099, 0.8418, 0.7693], N = [40236, 208862, 88763])
print(bla)
print('--------')


bla = MCC_per_class_accuracy(acc = [0.8818, 0.8135], N = [40236, 208862])
print(bla)
bla = MCC_per_class_accuracy(acc = [0.8818, 0.7135], N = [40236, 208862])
print(bla)
bla = MCC_per_class_accuracy(acc = [0.7818, 0.8135], N = [40236, 208862])
print(bla)
print('--------')

bla = MCC_per_class_accuracy(acc = [0.80, 0.80], N = [40236, 208862])
print(bla)
bla = MCC_per_class_accuracy(acc = [0.80, 0.8 - 5000/208862], N = [40236, 208862])
print(bla)
bla = MCC_per_class_accuracy(acc = [0.80 - 5000/40236, 0.80], N = [40236, 208862])
print(bla)
print('--------')


print('-------- : ---')
bla = MCC_per_class_accuracy(acc = [0.9740906826060086, 0.4946236558607932], N = [24724, 1048], num_classes = 2)
print(bla)

print('-------- : ---')
bla = MCC_per_class_accuracy(acc = [0.8120649651963185, 0.7440670079071938], N = [95091, 22566], num_classes = 2)
print(bla)