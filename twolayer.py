#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import struct
import numpy as np
import matplotlib.pyplot as plt 

def load_mnist(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels-idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

class TwoLayerNet():
    def __init__(self, input_dim=0, hidden_dim=0, output_dim=0, variation=1e-4):
        self.params = {}
        self.params['W1'] = variation * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = variation * np.random.randn(hidden_dim, output_dim)
        self.params['b2'] = np.zeros(output_dim)
    
    def loss(self, feature, label, regularization=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        num, dim = feature.shape

        hidden_output = np.maximum(0,feature.dot(W1)+b1)
        final_output = hidden_output.dot(W2)+b2

        softmax = np.exp(final_output)
        for i in range(0,num):
            softmax[i,:] /= np.sum(softmax[i,:])
        loss = 0
        for i in range(0,num):
            loss += -np.log(softmax[i,label[i]])
        loss = loss/num + regularization*(np.sum(W1*W1)+np.sum(W2*W2))

        gradient = {}
        dl = softmax.copy()
        for i in range(0,num):
            dl[i,label[i]] -= 1
        dl /= num
        gradient['W2'] = hidden_output.T.dot(dl) + 2*regularization*W2
        gradient['b2'] = np.sum(dl,axis=0)

        dh = dl.dot(W2.T)
        dh = (hidden_output>0)*dh
        gradient['W1'] = feature.T.dot(dh) + 2*regularization*W1
        gradient['b1'] = np.sum(dh,axis = 0)

        return loss, gradient
    
    def predict(self,feature):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']

        hidden_output = np.maximum(0,feature.dot(W1)+b1)
        final_output = hidden_output.dot(W2)+b2
        pred = np.argmax(final_output,axis=1)

        return pred


    def train(self, train_feature, train_label, test_feature, test_label,
              learning_rate=5e-3, learning_rate_decay=0.9,decay_steps = 100,
              regulariaztion=1e-3, iteration=1000,batch_size=1000):
        num, dim = train_feature.shape

        train_loss_history = []
        test_loss_history = []
        test_accuracy_history = []

        for i in range(iteration):
            if(i%decay_steps==0):
                learning_rate = learning_rate * learning_rate_decay
            print("iteration:",i)
            index = np.random.choice(num,batch_size,replace =True)
            feature_batch = train_feature[index]
            label_batch = train_label[index]

            loss, gradient = self.loss(feature_batch, label_batch, regulariaztion)
            train_loss_history.append(loss)

            self.params['W2'] += -learning_rate * gradient['W2']
            self.params['b2'] += -learning_rate * gradient['b2']
            self.params['W1'] += -learning_rate * gradient['W1']
            self.params['b1'] += -learning_rate * gradient['b1']

            loss, gradient = self.loss(test_feature, test_label, regulariaztion)
            test_loss_history.append(loss)
            test_accuracy = (self.predict(test_feature) == test_label).mean()
            test_accuracy_history.append(test_accuracy)
        
        return train_loss_history,test_loss_history,test_accuracy_history
    
    def save_model(self,filename):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        np.savez(filename,W1=W1,b1=b1,W2=W2,b2=b2)

    def load_model(self,filename):
        npzfile=np.load(filename)
        self.params['W1'] = npzfile['W1']
        self.params['b1'] = npzfile['b1']
        self.params['W2'] = npzfile['W2']
        self.params['b2'] = npzfile['b2']

    def show_net(self):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W1 = W1.reshape(10, 10, 3, -1)
        W2 = W2.reshape(10, 10, 3, -1)

        space = 1
        (height, width, channel, num) = W1.shape
        grid_size = int(np.ceil(np.sqrt(num)))
        grid_height = height * grid_size + space * (grid_size - 1)
        grid_width = width * grid_size + space * (grid_size - 1)
        grid = np.zeros((grid_height, grid_width, channel))
        
        index = 0
        ystart, yend = 0, height
        for y in range(grid_size):
            xstart, xend = 0, width
            for x in range(grid_size):
                if index < num:
                    img = W1[:,:,:,index]
                    low, high = np.min(img), np.max(img)
                    grid[ystart:yend, xstart:xend] = 255 * (img - low) / (high - low)
                    index += 1
                xstart += width + space
                xend += width + space
            ystart += height + space
            yend += height + space
        plt.figure(2)
        plt.subplot(1, 2, 1)
        plt.imshow(grid.astype('uint8'))
        plt.title('Visualization of W1')

        (height, width, channel, num) = W2.shape
        grid_size = int(np.ceil(np.sqrt(num)))
        grid_height = height * grid_size + space * (grid_size - 1)
        grid_width = width * grid_size + space * (grid_size - 1)
        grid = np.zeros((grid_height, grid_width, channel))
        
        index = 0
        ystart, yend = 0, height
        for y in range(grid_size):
            xstart, xend = 0, width
            for x in range(grid_size):
                if index < num:
                    img = W2[:,:,:,index]
                    low, high = np.min(img), np.max(img)
                    grid[ystart:yend, xstart:xend] = 255 * (img - low) / (high - low)
                    index += 1
                xstart += width + space
                xend += width + space
            ystart += height + space
            yend += height + space
        plt.figure(2)
        plt.subplot(1, 2, 2)
        plt.imshow(grid.astype('uint8'))
        plt.title('Visualization of W2')

        plt.show()


if __name__=="__main__":

    np.random.seed(0)
    train_image,train_label = load_mnist('./data')
    test_image,test_label = load_mnist('./data','t10k')
    print('Train data shape: ', train_image.shape)
    print('Train labels shape: ', train_label.shape)
    print('Test data shape: ', test_image.shape)
    print('Test labels shape: ', test_label.shape)

    net = TwoLayerNet(train_image.shape[1],300,10)
    train_loss_history,test_loss_history,test_accuracy_history = net.train(train_image,train_label,test_image,test_label)

    print(test_accuracy_history[-1])

    net.save_model('./save_model.npz')

    # loadnet = TwoLayerNet()
    # loadnet.load_model('./save_model.npz')
    # loadnet_accuracy = (loadnet.predict(test_image) == test_label).mean()
    # print('load net accuracy:',loadnet_accuracy)

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.plot(train_loss_history)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')

    plt.figure(1)
    plt.subplot(1, 3, 2)
    plt.plot(test_loss_history)
    plt.xlabel('iteration')
    plt.ylabel('testing loss')
    plt.title('Testing Loss history')

    plt.figure(1)
    plt.subplot(1, 3, 3)
    plt.plot(test_accuracy_history)
    plt.xlabel('iteration')
    plt.ylabel('testing accuracy')
    plt.title('Testing Accuracy history')

    plt.show()


    net.show_net()

