# HW1
Two Layer Neural Network

## 模型架构
1. twolayer.py：激活函数、反向传播、loss以及梯度的计算、学习率下降策略、L2正则化、优化器SGD、保存模型。
2. selection.py：对学习率、隐藏层大小、正则化强度超参数的查找。
3. test.py: 测试集训练。


## 模型复现
1. 运行selection.py查找参数。
2. 用确定好的参数值运行twolayer.py，会输出loss及accuracy曲线，同时将最终模型保存到save_model.npz文件。
3. 最后运行test.py，得到测试集分类精度。

## 模型最终的参数选择
learning rate：5e-3               
Hidden layer：300            
L2 Regularization：1e-3                
准确率：97.57%       

