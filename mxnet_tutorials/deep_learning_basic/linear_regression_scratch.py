from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

# plt.rcParams['figure.figsize'] = (3.5, 2.5)
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
plt.show()


# 本函数已保存在 gluonbook 包中方便以后使用。
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i:min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take 函数根据索引返回对应的元素


batch_size = 10

# 初始化参数
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

# 创建参数的梯度
w.attach_grad()
b.attach_grad()


# 定义模型 线性回归
def linreg(X, w, b):
    return nd.dot(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# param[:] = param - lr * param.grad换成param = param - lr * param.grad 报错
# 会重新创建新param，这个是没有attach_grad的
# 定义优化算法
# 这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和，将它除以批量大小来得到平均值
def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

# 在一个epoch中，将完整遍历一遍data_iter函数，
# 并对训练数据集中的所有样本都使用一次（假设样本数能够被批量大小整除）

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)
        # 每个小批量的损失l的形状为(batch_size,1)
        # 由于变量l并不是一个标量，运行l.backward()将对l中元素求和得到新的变量，再求该变量有关模型参数的梯度
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print("epoch %d, loss %f" % ((epoch + 1), train_l.mean().asnumpy()))

print("true_w is ", true_w, " w is ", w)
print("true_b is ", true_b, " b is ", b)
