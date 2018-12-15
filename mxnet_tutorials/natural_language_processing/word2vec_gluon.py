import collections
import math
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import random
import sys
import time
import zipfile

sys.path.insert(0, '..')
import gluonbook as gb

with zipfile.ZipFile('../data/ptb.zip', 'r') as zin:
    zin.extractall('../data')

with open('../data/ptb/ptb.train.txt', 'r') as f:
    lines = f.readlines()
    raw_dataset = [st.split() for st in lines]

print('# sentences: %d' % len(raw_dataset))

for st in raw_dataset[:3]:
    print('# tokens:', len(st), st[:5])

# 为简化计算，只保留在数据集中至少出现5次的词
counter = collections.Counter([tk for st in raw_dataset for tk in st])
counter = dict(filter(lambda x: x[1] >= 5, counter.items()))

# 将词映射到整数索引
idx_to_token = [tk for tk, _ in counter.items()]
token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] for st in raw_dataset]
num_tokens = sum([len(st) for st in dataset])
print('# tokens: %d' % num_tokens)


# 二次采样
# 定义丢弃方法
def discard(idx):
    return random.uniform(0, 1) < 1 - math.sqrt(
        1e-4 / counter[idx_to_token[idx]] * num_tokens)


subsampled_dataset = [[tk for tk in st if not discard(tk)] for st in dataset]

print('# tokens: %d' % sum([len(st) for st in subsampled_dataset]))


# 比较一个词在采样前后出现在数据集中的次数
def compare_counts(token):
    return '# %s: before=%d, after=%d' % (token,
                                          sum([st.count(token_to_idx[token]) for st in dataset]),
                                          sum([st.count(token_to_idx[token]) for st in subsampled_dataset]))


# 高频词"the"
print(compare_counts('the'))

# 低频词"join"
print(compare_counts('join'))


# 提取出所有中心词和它们的背景词
# 每次在整数1和max_window_size之间均匀随机采样一个整数作为背景窗口大小
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        # 每个句子至少要有 2 个词才可能组成一对“中心词 - 背景词”
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size),
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])  # 将中心词排除在背景词外
    return centers, contexts


# 创建一个人工数据集，设最大背景窗口为2，打印所有的中词词和它们的背景词
tiny_dataset = [list(range(7)), list(range(7, 10))]
print('dataset', tiny_dataset)
for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
    print('center', center, 'has contexts', context)

# 设最大背景窗口大小为5，提取数据集中所有的中心词及其背景词
all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)


# 负采样
# 随机采样K个噪音词，噪音词采样概率设为w词频与总词频之比的0.75次方
def get_negatives(all_contexts, sampling_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            if i == len(neg_candidates):
                # 根据每个词的权重（sampling_weights）随机生成K个词的索引作为噪音词
                # 为了高效计算，可以将K设的稍大一点
                i, neg_candidates = 0, random.choices(
                    population, sampling_weights, k=int(1e5))
            neg, i = neg_candidates[i], i + 1
            # 噪音词不能是背景词
            if neg not in set(contexts):
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


sampling_weights = [counter[w] ** 0.75 for w in idx_to_token]
all_negatives = get_negatives(all_contexts, sampling_weights, 5)


def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels))


batch_size = 512
num_workers = 0 if sys.platform.startswith('win32') else 4
dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
                             batchify_fn=batchify, num_workers=num_workers)
for batch in data_iter:
    for name, data in zip(['centers', 'contexts_negatives', 'masks',
                           'labels'], batch):
        print(name, 'shape:', data.shape)
    break

# 使用嵌入层和小批量乘法实现跳字模型
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
print(embed.weight)

# 输入一个词的索引i，嵌入层返回权重矩阵的第i行作为它的词向量
x = nd.array([[1, 2, 3], [4, 5, 6]])
print(embed(x))

# 小批量乘法
X = nd.ones((2, 1, 4))
Y = nd.ones((2, 4, 6))
print(nd.batch_dot(X, Y).shape)


def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1, 2))
    return pred


loss = gloss.SigmoidBinaryCrossEntropyLoss()
pred = nd.array([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
# 标签变量label中的1和0分别代表背景词和噪声词
label = nd.array([[1, 0, 0, 0], [1, 1, 0, 0]])
mask = nd.array([[1, 1, 1, 1], [1, 1, 1, 0]])  # 掩码变量
print(loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1))


def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))


print('%.7f' % ((sigmd(1.5) + sigmd(-0.3) + sigmd(1) + sigmd(-2)) / 4))
print('%.7f' % ((sigmd(1.1) + sigmd(-0.6) + sigmd(-2.2)) / 3))

embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
        nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))


def train(net, lr, num_epochs):
    ctx = gb.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    for epoch in range(num_epochs):
        start_time, train_l_sum = time.time(), 0
        for batch in data_iter:
            center, context_negative, mask, label = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                # 使用掩码变量mask避免填充项对损失函数计算的影响
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            train_l_sum += l.mean().asscalar()
        print('epoch %d, train loss %.2f, time %.2fs'
              % (epoch + 1, train_l_sum / len(data_iter), time.time() - start_time))


train(net, 0.005, 8)


def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[token_to_idx[query_token]]
    cos = nd.dot(W, x) / nd.sum(W * W, axis=1).sqrt() / nd.sum(x * x).sqrt()
    topk = nd.topk(cos, k=k + 1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # 除去输入词
        print('cosine sim=%.3f: %s' % (cos[i].asscalar(), (idx_to_token[i])))


get_similar_tokens('chip', 3, net[0])
