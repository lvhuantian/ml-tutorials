from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import matplotlib.pyplot as plt

# 生成一个人工数据集
# y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + epsilon

n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2), nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)

# 查看生成的数据集的前2个样本
print(features[:2], poly_features[:2], labels[:2])


# 定义作图函数，y轴使用了对数尺度
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals, y2_vals, linestyle=":")
        plt.legend(legend)
        plt.show()


num_epochs, loss = 100, gloss.L2Loss()


def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                                  batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), "sgd", {"learning_rate": 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features), train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())
    print("final epoch: train loss ", train_ls[-1], ", test loss ", test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, "epochs", "loss",
             range(1, num_epochs + 1), test_ls, ["train", "test"])
    print("weight: ", net[0].weight.data().asnumpy(),
          "\nbias: ", net[0].bias.data().asnumpy())


# 使用与数据生成函数同阶的三阶多项式函数拟合，模型的训练误差和测试误差都较低
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])

# 尝试线性函数拟合
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train], labels[n_train:])
