from mxnet import gluon, init, nd
from mxnet.gluon import nn
from mxnet.gluon import data as gdata
import sys

sys.path.insert(0, '..')
import gluonbook as gb

net = nn.Sequential()
# 使用较大的11x11窗口来捕获物体，使用步幅4来较大减少输出高和宽
net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 减小卷积窗口，使用填充为2来使得输入输出高宽一致，且增大输出通道数
        nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 连续三个卷积层，且使用更小的卷积窗口。除最后的卷积层外，进一步增大了输出通道数。
        # 前两个卷积层后不使用池化层来减少输入的高和宽
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
        nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
        nn.MaxPool2D(pool_size=3, strides=2),
        # 使用dropout
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
        # 输出层，由于使用Fashion-MNIST，故类别数为10
        nn.Dense(10))

# 观察每一层的输出
X = nd.random.uniform(shape=(1, 1, 224, 224))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)


# AlexNet使用的ImageNet数据
# 使用`Resize`将Fashion-MNIST的图像扩大到AlexNet使用的图像高和宽224

def load_data_fashion_mnist(batch_size, resize=None):
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)

    mnist_train = gdata.vision.FashionMNIST(root="../data", train=True)
    mnist_test = gdata.vision.FashionMNIST(root="../data", train=False)

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

    return train_iter, test_iter


batch_size = 128
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)

lr, num_epochs, ctx = 0.01, 5, gb.try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
