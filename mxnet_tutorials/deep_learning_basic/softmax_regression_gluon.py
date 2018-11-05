from mxnet.gluon import data as gdata
from mxnet import nd, autograd
from mxnet.gluon import loss as gloss, nn
from mxnet import gluon, init
import sys

mnist_train = gdata.vision.FashionMNIST(root="../data", train=True)
mnist_test = gdata.vision.FashionMNIST(root="../data", train=False)

batch_size = 256

transformer = gdata.vision.transforms.ToTensor()
if sys.platform.startswith('win'):
    num_workers = 0  # 0 表示不用额外的进程加速读取数据
else:
    num_workers = 4

train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)

# softmax回归的输出层是一个全连接层
# 添加一个输出个数为10的全连接层，使用均值为0标准差为0.01的正态分布随机初始化模型的权重参数
net = nn.Sequential()
net.add(nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

# 分开定义softmax运算和交叉熵损失函数可能会造成数值不稳定
# Gluon提供一个包含softmax运算和交叉熵损失计算的函数
loss = gloss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})


# 计算准确率
# `y_hat.argmax(axis=1)`返回矩阵`y_hat`每行中最大元素的索引，且返回结果与变量`y`形状相同
# `(y_hat.argmax(axis=1)==y)`是一个值为0或1的NDArray，
# 由于标签为整数，先将变量`y`变换为浮点数再进行相等条件判断
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


# 评价模型`net`在数据集`data_iter`上的准确率
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def train(net, train_iter, test_iter, loss, num_epochs, batch_size,
          params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)

            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print("epoch %d, loss %.4f, train acc %.3f, test acc %.3f"
              % (epoch + 1, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))


num_epochs = 5

train(net, train_iter, test_iter, loss, num_epochs,
      batch_size, None, None, trainer)
