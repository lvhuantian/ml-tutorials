import sys
import matplotlib.pyplot as plt
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

sys.path.insert(0, '..')
import gluonbook as gb

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train, :], labels[n_train:, :]


def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]


# 定义L2范数惩罚项
def l2_penalty(w):
    return (w ** 2).sum() / 2


batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = gb.linreg, gb.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                              batch_size, shuffle=True)


def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # 添加 L2 正则
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            gb.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels)
                        .mean().asscalar())
        test_ls.append(loss(net(test_features, w, b), test_labels)
                       .mean().asscalar())
    gb.semilogy(range(1, num_epochs + 1), train_ls, "epochs", "loss",
                range(1, num_epochs + 1), test_ls, ["train", "test"])
    print("L2 norm of w:", w.norm().asscalar())


# 观察过拟合
fit_and_plot(lambd=0)

# 使用权重衰减
fit_and_plot(lambd=3)


# 在构造`Trainer`实例时通过`wd`参数指定权重衰减超参数
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # 对权重参数衰减，权重名称一般是以 weight 结尾
    trainer_w = gluon.Trainer(net.collect_params(".*weight"),
                              "sgd", {"learning_rate": lr, "wd": wd})
    # 不对偏差参数衰减，偏差名称一般是以 bias 结尾
    trainer_b = gluon.Trainer(net.collect_params(".*bias"),
                              "sgd", {"learning_rate": lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # 对两个`Trainer`实例分别调用`step`函数，从而分别更新权重和偏差
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels)
                        .mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels)
                       .mean().asscalar())
    gb.semilogy(range(1, num_epochs + 1), train_ls, "epochs", "loss",
                range(1, num_epochs + 1), test_ls, ["train", "test"])
    print("L2 norm of w:", net[0].weight.data().norm().asscalar())


fit_and_plot_gluon(0)

fit_and_plot_gluon(3)
