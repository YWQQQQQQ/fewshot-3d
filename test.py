import numpy as np

label = np.zeros(shape=[5], dtype='float32')
l1 = []
l2 = []
l1.append(label)
l2.append(label)

l1[0][1]=1
print(l1,l2)