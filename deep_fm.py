# -*- coding: utf-8 -*-
# @Time     : 2019/5/26 17:28
# @Author   : buracagyang
# @File     : deep_fm.py
# @Software : PyCharm


"""
Context-Aware Multi-Modal Transportation Recommendation, KDD2019

Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.

[2] xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems,
    Jianxun Lian, Xiaohuan Zhou, Fuzheng Zhang, Zhongxia Chen, Xing Xie, Guangzhong Sun
"""

import os
import json
import glob
import shutil
import random
from datetime import date, timedelta
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribution mode {0-local, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 2, "Number of threads")
tf.app.flags.DEFINE_integer("feature_size", 46349, "Number of features")
tf.app.flags.DEFINE_integer("field_size", 91, "Number of fields")
tf.app.flags.DEFINE_integer("embedding_size", 8, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 8, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 1000, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '200,200,200', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.9,0.9,0.9', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", True, "perform batch normalization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.995, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", './data/output/', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", './model/saver/', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '', "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")


def input_fn(filename, batch_size=256, num_epochs=1, perform_shuffle=False):
    print('Parsing: ', filename)

    def decode_line(line):
        columns = tf.string_split([line], ' ')
        labels = tf.string_to_number(columns.values[0], out_type=tf.float32)
        splits = tf.string_split(columns.values[1:], ':')
        id_vals = tf.reshape(splits.values, splits.dense_shape)
        feat_ids, feat_vals = tf.split(id_vals, num_or_size_splits=2, axis=1)
        feat_ids = tf.string_to_number(feat_ids, out_type=tf.int32)
        feat_vals = tf.string_to_number(feat_vals, out_type=tf.float32)

        return {"feat_ids": feat_ids, "feat_vals": feat_vals}, labels

    data_set = tf.data.TextLineDataset(filename).map(decode_line, num_parallel_calls=5).prefetch(500000)
    # Randomizes
    if perform_shuffle:
        data_set = data_set.shuffle(buffer_size=256)
    # epochs from blending together.
    data_set = data_set.repeat(num_epochs)
    data_set = data_set.batch(batch_size)

    iterator = data_set.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    # hyper parameters
    field_size = params["field_size"]
    feature_size = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    batch_norm = params['batch_norm']
    # batch_norm_decay = params["batch_norm_decay"]
    optimizer = params["optimizer"]
    layers = list(map(int, params["deep_layers"].split(',')))
    dropout = list(map(float, params["dropout"].split(',')))

    # weights
    fm_bias = tf.get_variable(name='fm_bias', shape=[1], initializer=tf.constant_initializer(0.0))
    fm_weight = tf.get_variable(name='fm_w', shape=[feature_size], initializer=tf.glorot_normal_initializer(seed=2019))
    fm_v = tf.get_variable(name='fm_v', shape=[feature_size, embedding_size],
                           initializer=tf.glorot_normal_initializer(seed=2019))

    # features
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # f(x)
    with tf.variable_scope("First-order"):
        feat_wgts = tf.nn.embedding_lookup(fm_weight, feat_ids)  # None * F(field_size) * 1
        y_w = tf.reduce_sum(tf.multiply(feat_wgts, feat_vals), 1)

    with tf.variable_scope("Second-order"):
        embeddings = tf.nn.embedding_lookup(fm_v, feat_ids)  # None * F(field_size) * K
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        embeddings = tf.multiply(embeddings, feat_vals)  # vij * xi
        sum_square = tf.square(tf.reduce_sum(embeddings, 1))
        square_sum = tf.reduce_sum(tf.square(embeddings), 1)
        y_v = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), 1)  # None * 1

    with tf.variable_scope("Deep-part"):
        if batch_norm:
            # normalizer_fn = tf.contrib.layers.batch_norm
            # normalizer_fn = tf.layers.batch_normalization
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_phase = True
                # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True,
                # 'updates_collections': None, 'is_training': True, 'reuse': None}
            else:
                train_phase = False
                # normalizer_params = {'decay': batch_norm_decay, 'center': True, 'scale': True,
                # 'updates_collections': None, 'is_training': False, 'reuse': True}
        else:
            train_phase = False
            # normalizer_fn = None
            # normalizer_params = None

        deep_inputs = tf.reshape(embeddings, shape=[-1, field_size * embedding_size])  # None * (F*K)
        # build full connected NN
        for i in range(len(layers)):
            # if batch_norm:
            #    deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' %i)
            # normalizer_params.update({'scope': 'bn_%d' %i})
            deep_inputs = tf.contrib.layers.fully_connected(
                inputs=deep_inputs, num_outputs=layers[i],
                weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg), scope='mlp%d' % i)

            if batch_norm:
                # 放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn
                deep_inputs = batch_norm_layer(deep_inputs, train_phase=train_phase, scope_bn='bn_%d' % i)
            if mode == tf.estimator.ModeKeys.TRAIN:
                deep_inputs = tf.nn.dropout(deep_inputs, keep_prob=dropout[i])

        y_deep = tf.contrib.layers.fully_connected(inputs=deep_inputs, num_outputs=1, activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(l2_reg),
                                                   scope='deep_out')
        y_d = tf.reshape(y_deep, shape=[-1])
        # sig_wgts = tf.get_variable(name='sigmoid_weights', shape=[layers[-1]],
        # initializer=tf.glorot_normal_initializer())
        # sig_bias = tf.get_variable(name='sigmoid_bias', shape=[1], initializer=tf.constant_initializer(0.0))
        # deep_out = tf.nn.xw_plus_b(deep_inputs,sig_wgts,sig_bias,name='deep_out')

    with tf.variable_scope("DeepFM-out"):
        y_bias = fm_bias * tf.ones_like(y_d, dtype=tf.float32)  # None * 1
        y = y_bias + y_w + y_v + y_d
        pred = tf.sigmoid(y)

    predictions = {"prob": pred}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            tf.estimator.export.PredictOutput(predictions)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # loss function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=y)) + \
        l2_reg * tf.nn.l2_loss(fm_weight) + \
        l2_reg * tf.nn.l2_loss(fm_v) + \
        tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # Provide an estimator spec for `ModeKeys.EVAL`
    eval_metric_ops = {"auc": tf.metrics.auc(labels, pred)}
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    # optimizer
    if optimizer == 'GD':
        opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif optimizer == 'Adam':
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif optimizer == 'Momentum':
        opt = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    else:
        opt = tf.train.FtrlOptimizer(learning_rate=learning_rate)

    train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
    # trainable_params = tf.trainable_variables()
    # print(trainable_params)
    # gradients = tf.gradients(loss, trainable_params)
    # clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    # train_op = opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=tf.train.get_global_step())

    # Provide an estimator spec for `ModeKeys.TRAIN` modes
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions,
                                          loss=loss,
                                          train_op=train_op)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def set_dist_env():
    if FLAGS.dist_mode == 1:  # 本地分布式测试模式1 chief, 1 ps, 1 evaluator
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # 无worker参数
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)

    elif FLAGS.dist_mode == 2:  # 集群分布式模式
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[2:]  # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)


def main(_):
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-1)).strftime('%Y%m%d')
    FLAGS.model_dir = FLAGS.model_dir + FLAGS.dt_dir

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('batch_norm_decay ', FLAGS.batch_norm_decay)
    print('batch_norm ', FLAGS.batch_norm)
    print('l2_reg ', FLAGS.l2_reg)

    # 初始化
    tr_files = glob.glob("%s/tr*cleared" % FLAGS.data_dir)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)
    va_files = glob.glob("%s/va*cleared" % FLAGS.data_dir)
    print("va_files:", va_files)
    te_files = glob.glob("%s/te*cleared" % FLAGS.data_dir)
    print("te_files:", te_files)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    # 创建网络
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "batch_norm": FLAGS.batch_norm,
        "batch_norm_decay": FLAGS.batch_norm_decay,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "optimizer": FLAGS.optimizer
    }
    config = tf.estimator.RunConfig(tf_random_seed=2019).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    deep_fm = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size), steps=None,
            start_delay_secs=1000, throttle_secs=120)
        tf.estimator.train_and_evaluate(deep_fm, train_spec, eval_spec)

    elif FLAGS.task_type == 'eval':
        deep_fm.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))

    elif FLAGS.task_type == 'infer':
        predict_prob = deep_fm.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),
                                       predict_keys="prob")
        with open(FLAGS.data_dir + "/predict_prob", "w") as fo:
            for prob in predict_prob:
                fo.write("%f\n" % (prob['prob']))

    elif FLAGS.task_type == 'export':
        feature_spec = {
            'feat_ids': tf.placeholder(dtype=tf.int64, shape=[None, FLAGS.field_size], name='feat_ids'),
            'feat_vals': tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        }
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
        deep_fm.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
