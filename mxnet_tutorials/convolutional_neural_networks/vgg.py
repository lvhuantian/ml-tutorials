from mxnet import gluon, init, nd
from mxnet.gluon import nn
import sys

sys.path.insert(0, '..')
import gluonbook as gb


def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))


# 实现VGG-11
def vgg(conv_arch):
    net = nn.Sequential()
    # 卷积部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net


net = vgg(conv_arch)

# 构造一个高和宽均为224的单通道数据样本来观察每一层的输出形状
net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)

# VGG-11计算上比AlexNet更加复杂
# 出于测试目的，构造一个通道数更小的网络
ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg(small_conv_arch)

lr, num_epochs, batch_size, ctx = 0.05, 5, 128, gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size, resize=224, root='../data')
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
