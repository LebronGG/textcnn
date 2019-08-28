# coding: utf-8

from __future__ import print_function

import os
import tensorflow as tf
import tensorflow.contrib.keras as kr
import numpy as np
from cnn_model import TCNNConfig, TextCNN
from cnews_loader import read_category, read_vocab
from sub_emoj import emoj_sub,url_sub
try:
    bool(type(unicode))
except NameError:
    unicode = str

base_dir = 'data/cnews'
vocab_dir = os.path.join(base_dir, 'vocab1.txt')
pre_training = os.path.join(base_dir, 'data_vec.npy')

save_dir = 'checkpoints/textcnn'


def load_checkpoint(checkpoint_dir, session, var_list=None):
    print(' [*] Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
    try:
        restorer = tf.train.Saver(var_list)
        restorer.restore(session, ckpt_path)
        print(' [*] Loading successful! Copy variables from % s' % ckpt_path)
        return True
    except:
        print(' [*] No suitable checkpoint!')
        return False


class CnnModel:
    def __init__(self):
        self.config = TCNNConfig()
        self.categories, self.cat_to_id = read_category()
        self.words, self.word_to_id = read_vocab(vocab_dir)
        self.config.vocab_size = len(self.words)
        self.config.pre_training = np.load(pre_training)
        self.model = TextCNN(self.config)

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        if not load_checkpoint(save_dir, self.session):
            exit()

    def predict(self, message):
        # 支持不论在python2还是python3下训练的模型都可以在2或者3的环境下运行
        content = unicode(message)
        content=emoj_sub(content)
        content=url_sub(content)
        data = [self.word_to_id[x] for x in content if x in self.word_to_id]
        feed_dict = {
            self.model.input_x: kr.preprocessing.sequence.pad_sequences([data], self.config.seq_length),
            self.model.keep_prob: 1.0
        }
        y_pred_cls, conf = self.session.run([self.model.y_pred_cls,self.model.confidence], feed_dict=feed_dict)
        return self.categories[y_pred_cls[0]], conf


if __name__ == '__main__':

    cnn_model = CnnModel()
    # a = ['ṬṜṃ➐➒➏片溦', '软  ++名，摸奶，溦[emts]_sys_0016_电到了(R)[/emts]', '悠  ++名，摸奶，溦[emts]_sys_0016_电到了(R)[/emts]']
    a = ['看看盘丝装备', '盘莲生[emts]_sys_0044_偷偷笑(DD)[/emts]', 'vivo登OPPO？']
    for i in a:
        print(cnn_model.predict(i))