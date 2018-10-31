import numpy as np
import pickle

msg = "hello world"
print(msg)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#data1 = unpickle("./cifar-10-batches-py/data_batch_1")
#data2 = unpickle("./cifar-10-batches-py/data_batch_2")
#data3 = unpickle("./cifar-10-batches-py/data_batch_3")
#data4 = unpickle("./cifar-10-batches-py/data_batch_4")
#data5 = unpickle("./cifar-10-batches-py/data_batch_5")

#x_train = np.concatenate((data1[b'data'], data2[b'data'], data3[b'data'], data4[b'data'], data5[b'data']), axis=0)
#y_train = np.concatenate((data1[b'labels'],data2[b'labels'],data3[b'labels'],data4[b'labels'],data5[b'labels']), axis=0)

#np.save('data/cifar10_x_train', x_train)
#np.save('data/cifar10_y_train', y_train)

data6 = unpickle("./cifar-10-batches-py/test_batch")
x_test = data6[b'data']
y_test = data6[b'labels']

np.save('data/cifar10_x_test', x_test)
np.save('data/cifar10_y_test', y_test)