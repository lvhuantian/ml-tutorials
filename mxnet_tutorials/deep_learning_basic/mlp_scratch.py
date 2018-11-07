from mxnet import autograd, nd
from mxnet.gluon import data as gdata
from mxnet.gluon import loss as gloss
import sys

sys.path.insert(0, '..')
import gluonbook as gb

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

# Fashion-MNIST数据集中图像形状为28x28，类别数为10
# 设置超参数隐藏单元个数为256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# TODO check
# W的初始化mean分布式在0，std=1/sqrt(n)，n为units的个数
# scale为1时太大，导致梯度过大，SGD无法运行

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()


# 使用基础的`maximum`函数来实现ReLU
def relu(X):
    return nd.maximum(X, 0)


def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2


loss = gloss.SoftmaxCrossEntropyLoss()

num_epochs, lr = 5, 0.1

gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params, lr)
