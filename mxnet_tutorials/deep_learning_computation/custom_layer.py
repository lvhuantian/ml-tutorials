from mxnet import gluon, nd
from mxnet.gluon import nn


# 通过继承Block类自定了一个将输入减掉均值后输出的层
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


layer = CenteredLayer()
print(layer(nd.array([1, 2, 3, 4, 5])))

net = nn.Sequential()
net.add(nn.Dense(128), CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.shape)
print(y.mean())
print(y.mean().asscalar())


class MyDense(nn.Block):
    # units: 输入个数； in_units: 输出个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


dense = MyDense(units=3, in_units=5)
print(dense.params)

dense.initialize()
print(dense(nd.random.uniform(shape=(2, 5))))
