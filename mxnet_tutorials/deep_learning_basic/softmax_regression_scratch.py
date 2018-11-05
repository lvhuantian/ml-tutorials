from mxnet.gluon import data as gdata
from mxnet import nd, autograd
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

# 模型的输入向量的长度是28x28=784，该向量的每个元素对应图像中的每个像素
# 由于图像有10个类别，单层神经网络输出层的输出个数为10
# 权重和偏差分别为784x10和1x10的矩阵
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)

W.attach_grad()
b.attach_grad()


# 可以对同一列（axis=0）或同一行（axis=1）的元素求和，并在结果中保留行和列这两个维度（keepdims=True）

# 先通过`exp`函数对每个元素做指数运算，再对`exp`矩阵同行元素求和，
# 最后令矩阵每行各元素与该行元素之和相除。
# 最后的矩阵每行元素和为1且非负


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 广播


# 测试
X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
print(X_prob, X_prob.sum(axis=1))


# 定义模型
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


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


num_epochs, lr = 5, 0.1


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


train(net, train_iter, test_iter, cross_entropy, num_epochs,
      batch_size, [W, b], lr)
