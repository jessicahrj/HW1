#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from importlib.util import LazyLoader
import os
import struct
import numpy as np
import matplotlib.pyplot as plt 

import twolayer 

f = open('./selection.txt', 'w')

learning_rate_list = [5e-3,1e-3,5e-4,1e-4,5e-5]
hidden_dim_list = [100,200,300,500,1000]
regularization_list = [1e-1,1e-2,1e-3,1e-4,1e-5]


train_image,train_label = twolayer.load_mnist('./data')
test_image,test_label = twolayer.load_mnist('./data','t10k')

for i in range(len(learning_rate_list)):
    for j in range(len(hidden_dim_list)):
        for k in range(len(regularization_list)):
            print(i,j,k)
            np.random.seed(0)

            net = twolayer.TwoLayerNet(train_image.shape[1],hidden_dim_list[j],10)
            train_loss_history,test_loss_history,test_accuracy_history = net.train(train_image,train_label,test_image,test_label,learning_rate=learning_rate_list[i],regulariaztion=regularization_list[k])

            f.write('learning_rate:')
            f.write(str(learning_rate_list[i]))
            f.write('hidden_dim:')
            f.write(str(hidden_dim_list[j]))
            f.write('regularization:')
            f.write(str(regularization_list[k]))
            f.write('Accuracy:')
            f.write(str(max(test_accuracy_history)))
            f.write('\n')

f.close()

