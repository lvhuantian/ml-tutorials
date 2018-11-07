from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn
from mxnet.gluon import data as gdata
import sys

sys.path.insert(0, '..')
import gluonbook as gb

net = nn.Sequential()
net.add(nn.Dense(256, activation="relu"),
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

batch_size = 256
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

loss = gloss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.5})
num_epochs = 15
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             None, None, trainer)
