from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)

batch_size = 10
# 将训练数据的特征和标签组合
dataset = gdata.ArrayDataset(features, labels)
# 随机读取小批量
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

# Sequential 实例可以看作是一个串联各个层的容器
net = nn.Sequential()
# 全连接层是一个Dense实例
net.add(nn.Dense(1))

# 初始化模型参数
# 通过init.Normal(sigma=0.01)指定权重参数每个元素将在初始化时随机采样于均值为 0 标准差为 0.01 的正态分布
# 偏差参数默认初始化为0
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gloss.L2Loss()

# 创建一个Trainer实例，并指定learning rate为0.03的sgd为优化算法
# 该优化算法将用来迭代net实例所有通过add函数嵌套的层所包含的全部参数，这些参数可以通过collect_params函数获取
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 通过调用Trainer实例的step函数来迭代模型函数
# 由于变量l是长度为batch_size的一维NDArray，执行l.backward()等价于执行l.sum().backward()
# 在step函数中指明批量大小，从而对批量中的样本梯度求平均
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print("epoch %d, loss %f" % (epoch, l.mean().asnumpy()))

dense = net[0]
print("true_w is ", true_w, ", w is ", dense.weight.data())
print("true_b is ", true_b, ", b is ", dense.bias.data())
