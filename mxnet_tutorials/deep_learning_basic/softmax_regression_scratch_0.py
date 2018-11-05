from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
import sys
import time

mnist_train = gdata.vision.FashionMNIST(root="../data", train=True)
mnist_test = gdata.vision.FashionMNIST(root="../data", train=False)

# 训练集合测试集中的每个类别的图像数分别为6000和1000
# 因为10个类别，所以训练集和测试集的样本数分别为60000和10000
print(len(mnist_train), len(mnist_test))

# 取第一个样本的图像和标签
feature, label = mnist_train[0]

# 变量`feature`对应高和宽均为28像素的图像
# 每个像素的数值为0到255之间8位无符号整数（uint8）
# 使用3维的NDArray存储，最后一维是通道数，因为是灰度图像，所以通道数为1
# (h,w)
print(feature.shape, feature.dtype)

# 图像的标签使用numpy表示，类型为int32
print(label, type(label), label.dtype)


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


# 查看训练集中前9个样本的图像和标签
X, y = mnist_train[0:9]
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 通过`ToTensor`类将图像数据从uint8格式转换成32位浮点格式，
# 并除以255使得所有像素的数值均在0到1之间。
# `ToTensor`类还将图像通道从最后一维移到最前一维来方便卷积计算
# 通过数据集的`transform_first`函数，将`ToTensor`的变换应用在每个数据样本（图像和标签）的第一个元素，即图像之上。
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

# 查看读取一遍训练数据需要的时间
start = time.time()
for X, y in train_iter:
    continue
print("%.2f sec" % (time.time() - start))
