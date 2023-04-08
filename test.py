#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from importlib.util import LazyLoader
import os
import struct
import numpy as np
import matplotlib.pyplot as plt 

import twolayer 


loadnet =twolayer.TwoLayerNet()
test_image,test_label = twolayer.load_mnist('./data','t10k')
filename = './save_model.npz'
loadnet.load_model(filename)

loadnet_accuracy = (loadnet.predict(test_image) == test_label).mean()
print('Accuracy:',loadnet_accuracy)

