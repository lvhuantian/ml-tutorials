from matplotlib import pyplot as plt
from mxnet import autograd, nd


def xyplot(x_vals, y_vals, name):
    plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    plt.xlabel('x')
    plt.ylabel(name + '(x)')
    plt.show()


# ReLU(rectified linear unit)
# ReLU(x)=max(x,0)

x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()

# 绘制ReLU函数
xyplot(x, y, 'relu')

# 绘制ReLUctant函数的导数
y.backward()
xyplot(x, x.grad, 'grad of relu')

# sigmoid(x)=1/(1+exp(-x))

with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')

# 绘制sigmoid函数的导数
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')

# tanh(x)=(1-exp(-2x))/(1+exp(-2x))

with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')

# 绘制tanh函数的导数
y.backward()
xyplot(x, x.grad, 'grad of tanh')

# 多层感知机在输出层和输入层之间加入了一个或多个全连接隐藏层，
# 并通过激活函数对隐藏层输出进行变换
