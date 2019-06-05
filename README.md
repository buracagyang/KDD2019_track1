# KDD2019_track1
KDD2019中的常规机器学习赛道： [Context-Aware Multi-Modal Transportation Recommendation](https://dianshi.baidu.com/competition/29/rule).

基于TensorFlow写了一个DeepFM去解决这个问题，除了比赛提供的原始数据集，只加入了对应时间的天气特征，这块儿大家可以尝试做更多的特征工程。Demo很简单，效果跟 https://github.com/yaoxuefeng6/Paddle_baseline_KDD2019 提供的baseline效果差不多，最终线上提交得分在0.68-0.69之间，当然跟TOP级别的还有些差距。



## 环境

`Python3.5` | `Windows10` | `TensorFlow1.11.0`



## 特征工程

```python
python preprocess.py
python gen_features.py
```



## 训练网络

```python
python deep_fm.py --task_type=train  # --embedding_size=10 --learning_rate=1.0 尝试不同的超参数组合
```



## 测试结果

```python
python deep_fm.py --task_type=infer
python build_submit.py
```













