# -*- coding: utf-8 -*-
# @Time     : 2019/5/26 16:09
# @Author   : buracagyang
# @File     : gen_features.py
# @Software : PyCharm

"""
Describe:
        
"""

import os
import sys
from math import ceil
import random
import collections
import argparse

continuous_features = list(range(1, 4))
categorical_features = list(range(4, 92))

continuous_clip = [200000, 20000, 10800]  # 连续变量的截断

DISTANCE_MIN = 1.0
DISTANCE_MAX = 225864.0
DISTANCE_TS = 200000.0

PRICE_MIN = 200.0
PRICE_MAX = 92300.0
PRICE_TS = 20000

ETA_MIN = 1.0
ETA_MAX = 72992.0
ETA_TS = 10800.0


class ContinuousFeatureGenerator(object):
    def __init__(self, num_feature):
        self.num_feature = num_feature
        self.min = [sys.maxsize] * num_feature
        self.max = [-sys.maxsize] * num_feature

    def build(self, datafile, continuous_features_idx):
        # self.min = [DISTANCE_MIN, PRICE_MIN, ETA_MIN]
        # self.max = [DISTANCE_TS, PRICE_TS, ETA_TS]
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(' ')
                for i in range(0, self.num_feature):
                    val = features[continuous_features_idx[i]]  # 获取连续变量的value
                    if val != '':
                        val = int(ceil(float(val)))
                        if val > continuous_clip[i]:  # 对其做截断处理
                            val = continuous_clip[i]
                        self.min[i] = min(self.min[i], val)
                        self.max[i] = max(self.max[i], val)

    def gen(self, idx, val):
        if val == '':
            return 0.0
        if float(val) > self.max[idx]:
            val = self.max[idx]
        val = float(val)
        return (val - self.min[idx]) / (self.max[idx] - self.min[idx])


class CategoryDictGenerator(object):
    def __init__(self, num_feature):
        self.dicts = []
        self.num_feature = num_feature
        for i in range(0, num_feature):
            self.dicts.append(collections.defaultdict(int))

    def build(self, datafile, categorical_features_idx, cutoff=1):
        with open(datafile, 'r') as f:
            for line in f:
                features = line.rstrip('\n').split(' ')
                for i in range(0, self.num_feature):  # 遍历每一个分类特征
                    if features[categorical_features_idx[i]] != '':
                        self.dicts[i][features[categorical_features_idx[i]]] += 1  # 对于该分类特征下每一个属性值加1
        for i in range(0, self.num_feature):
            # 对于每个分类特征值，筛选出该特征下属性值出现cutoff次以上的分类属性
            self.dicts[i] = filter(lambda x: x[1] >= cutoff, self.dicts[i].items())
            self.dicts[i] = sorted(self.dicts[i], key=lambda x: (-x[1], x[0]))
            vocabs, _ = list(zip(*self.dicts[i]))
            # 对于每个分类特征，编码其分类属性值;定义一个unk属性，编码为0
            self.dicts[i] = dict(zip(vocabs, range(1, len(vocabs) + 1)))
            self.dicts[i]['<unk>'] = 0

    def gen(self, idx, key):
        if key not in self.dicts[idx]:
            res = self.dicts[idx]['<unk>']
        else:
            res = self.dicts[idx][key]
        return res

    def dicts_sizes(self):
        return list(map(len, self.dicts))


def get_features_main(datadir, outdir):
    # pool = ThreadPool(FLAGS.threads)
    dists = ContinuousFeatureGenerator(len(continuous_features))
    dists.build(os.path.join(datadir, 'train.txt'), continuous_features)
    # pool.apply(dists.build, args=(FLAGS.input_dir + 'train.txt', continous_features,))

    dicts = CategoryDictGenerator(len(categorical_features))
    dicts.build(os.path.join(datadir, 'train.txt'), categorical_features, cutoff=FLAGS.cutoff)
    # pool.apply(dicts.build, args=(FLAGS.input_dir + 'train.txt', categorial_features,))

    # pool.close()
    # pool.join()

    output = open(os.path.join(outdir, 'feature_map'), 'w')
    for i in continuous_features:
        output.write("{0} {1}\n".format('I' + str(i), i))
    dict_sizes = dicts.dicts_sizes()
    categorial_feature_offset = [dists.num_feature]
    for i in range(1, len(categorical_features) + 1):
        offset = categorial_feature_offset[i - 1] + dict_sizes[i - 1]
        categorial_feature_offset.append(offset)
        for key, val in dicts.dicts[i - 1].items():
            output.write("{0} {1}\n".format('C' + str(i) + '|' + key, categorial_feature_offset[i - 1] + val + 1))

    random.seed(2019)

    # 90% for training, 10% for validation.
    with open(os.path.join(outdir, 'tr.cleared'), 'w') as out_train:
        with open(os.path.join(outdir, 'va.cleared'), 'w') as out_valid:
            with open(os.path.join(datadir, 'train.txt'), 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split(' ')

                    feat_vals = []
                    for i in range(0, len(continuous_features)):
                        val = dists.gen(i, features[continuous_features[i]])
                        feat_vals.append(
                            str(continuous_features[i]) + ':' + "{0:.10f}".format(val).rstrip('0').rstrip('.'))

                    for i in range(0, len(categorical_features)):
                        val = dicts.gen(i, features[categorical_features[i]]) + categorial_feature_offset[i]
                        feat_vals.append(str(val) + ':1')

                    label = features[0]
                    if random.randint(0, 9999) % 10 != 0:
                        out_train.write("{0} {1}\n".format(label, ' '.join(feat_vals)))
                    else:
                        out_valid.write("{0} {1}\n".format(label, ' '.join(feat_vals)))

    with open(os.path.join(outdir, 'te.cleared'), 'w') as out:
        with open(os.path.join(datadir, 'test.txt'), 'r') as f:
            label = str(0)
            for line in f:
                features = line.rstrip('\n').split(' ')

                # 由于测试文件已经在首列添加了label字段，故处理方式与训练集处理方式一样
                feat_vals = []
                for i in range(0, len(continuous_features)):
                    val = dists.gen(i, features[continuous_features[i]])
                    feat_vals.append(str(continuous_features[i]) + ':' + "{0:.10f}".format(val).rstrip('0').rstrip('.'))

                for i in range(0, len(categorical_features)):
                    val = dicts.gen(i, features[categorical_features[i]]) + categorial_feature_offset[i]
                    feat_vals.append(str(val) + ':1')

                out.write("{0} {1}\n".format(label, ' '.join(feat_vals)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="threads num"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./data/output",
        help="input data dir"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/output",
        help="feature map output dir"
    )
    parser.add_argument(
        "--cutoff",
        type=int,
        default=1,
        help="cutoff long-tailed categorical values"
    )

    FLAGS, unparsed = parser.parse_known_args()
    print('threads ', FLAGS.threads)
    print('input_dir ', FLAGS.input_dir)
    print('output_dir ', FLAGS.output_dir)
    print('cutoff ', FLAGS.cutoff)

    get_features_main(FLAGS.input_dir, FLAGS.output_dir)
