import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
import time
import sys

sys.path.insert(0, '..')
import gluonbook as gb

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小,通道,高,宽)形状的输入转换成
        # (批量大小,通道*高*宽)形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# 构造一个高和宽均为28的单通道数据样本
# 逐层进行前向计算来查看每个层的输出形状
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

batch_size = 256
mnist_train = gdata.vision.FashionMNIST(root="../data", train=True)
mnist_test = gdata.vision.FashionMNIST(root="../data", train=False)
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


def try_gpu4():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx


ctx = try_gpu4()
print(ctx)


def evaluate_accuracy(data_iter, net, ctx):
    acc = nd.array([0], ctx=ctx)
    for X, y in data_iter:
        # 如果ctx是GPU，将数据复制到GPU上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx)
        acc += gb.accuracy(net(X), y)
    return acc.asscalar() / len(data_iter)


def train_ch5(net, train_iter, test_iter, batch_size,
              trainer, ctx, num_epochs):
    print('training on ', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, start = 0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
            train_acc_sum += gb.accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec' % (epoch + 1, train_l_sum / len(train_iter),
                                 train_acc_sum / len(train_iter),
                                 test_acc, time.time() - start))


lr, num_epochs = 0.9, 5
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
