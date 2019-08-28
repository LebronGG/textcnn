# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 16  # 词向量维度
    seq_length = 50  # 序列长度
    num_classes = 2  # 类别数
    hidden_dim = 128  # 全连接层神经元
    kernel_size = [2,3,4]  # 卷积核尺寸
    vocab_size = 16399  # 词汇表达小
    pre_training = None  # use vector_char trained by word2vec
    num_filters = 128  # 卷积核数目

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 20  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard

    require_improvement = 3000

class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim])
            # self.embedding = tf.constant(self.config.pre_training, name='embedding', dtype=tf.float32)
            # self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],initializer=tf.constant_initializer(self.config.pre_training))
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            self.conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv1d')
            # self.conv = tf.nn.relu(self.conv)
            gmp = tf.reduce_max(self.conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            self.fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(self.fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
            self.confidence = tf.nn.softmax(self.logits)[0][self.y_pred_cls[0]]


        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


class TextCNN1(object):

    def __init__(self, config):
        self.config = config
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.text_cnn()

    def text_cnn(self):

        with tf.device('/cpu:0'), tf.name_scope('embedding'):

            # self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim])
            # self.embedding = tf.constant(self.config.pre_training, name='embedding', dtype=tf.float32)
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_dim],initializer=tf.constant_initializer(self.config.pre_training))

            embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope('CNN'):
            pooled_outputs = []
            for i, filter_size in enumerate(self.config.filters_size):
                conv = tf.layers.conv1d(embedding_input, self.config.num_filters, filter_size,name='conv1d_{}'.format(filter_size))
                conv = tf.nn.relu(conv)
                gmp = tf.reduce_max(conv, reduction_indices=[1])
                pooled_outputs.extend([gmp])
            self.output = tf.concat(pooled_outputs, axis=-1)

        with tf.name_scope('score'):
            # 全连接层，后面接dropout以及relu激活
            self.fc = tf.layers.dense(self.output, self.config.hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(self.fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)
            self.confidence = tf.nn.softmax(self.logits)[0][self.y_pred_cls[0]]

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度值,变量
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯度*(clip/l2_g),得到新梯度
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1

        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


