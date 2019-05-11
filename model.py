# -*- coding: utf-8 -*-

'''

┌┬┐┌─┐┌─┐┬┌─┌─┐┌┬┐┬┌┐┌┌─┐
│││├─┤└─┐├┴┐│ │ │││││││ ┬
┴ ┴┴ ┴└─┘┴ ┴└─┘─┴┘┴┘└┘└─┘
(c)2019 _drat | v.000000001 | PT. Warung Data Indonesia

Maskoding adalah Artificial Inteligent untuk pembuatan koding autgenerator
pada pemrograman apapun. Menggunakan Deep Learning bisa
dengan Keras maupun Tensorflow.

RNN (Reccurent Neural Network) dengan dataset yang berisi file-file jenis
bahasa pemrograman yang ingin dilatih.

Tiga variabel penting yang perlu diingat di sini adalah vocab_to_int,
int_to_vocab dan disandikan.

Dua yang pertama memungkinkan kita untuk dengan mudah beralih antara karakter dan int dan sebaliknya. Yang terakhir adalah representasi semua dataset kami dalam format encoder.
(Hanya int bukan karakter).

Inisialisasi sel LSTM (http://colah.github.io/posts/2015-08-Understanding-LSTMs/) digunakan untuk membuat RNN. Lapisan output terhubung ke setiap sel.

Operasi yang digunakan untuk mengukur kesalahan model.

Dan akhirnya untuk meminimalisir keasalah pada model digunakan Adam Optimizer.
------

WARNING!!
-
Masih sangat konseptual dan harus diriset dan dikembangkan lebih jauh lagi dengan
konsep pemodelan pada neural network yang lebih konsisten.

'''


import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)


import numpy as np
import json

#
# Class untuk pengambilan dataset dan pembuatan network graph untuk
# Keras ataupun Tensorflow
#
class Model(object):

    PROG_LANG_EXT = ".py"
    DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
    DATASET_FILE = os.path.join(DATASET_FOLDER, "dataset.dt")
    CHECKPOINT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    TENSORBOARD_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs/train")

    #
    # Setting pada TF dilakukan di sini sesuai dengan kebutuhan
    # ingat sistem ini tidak ada GPU!
    #
    SEQUENCE_SIZE = 100
    BATCH_SIZE = 200
    HIDDEN_LAYER_SIZE = 512
    LSTM_SIZE = 2
    LR = 0.0005
    KEEP_PROB = 0.5

    #
    # Input -> training(Boolean), path model(ckpt)
    #
    def __init__(self, training=False, ckpt=None):

        super(Model, self).__init__()

        #
        # Simpan sesi TF di dalam folder checkpoints
        #
        if not os.path.exists(self.CHECKPOINT_FOLDER):
            os.makedirs(self.CHECKPOINT_FOLDER)

        # Mengolah dataset
        print("[-] Mengolah dataset ...")
        self.encoded, self.vocab_to_int, self.int_to_vocab = Model.get_dataset()
        print("[-] Dataset siap diolah.")

        #
        # io_size adalah jumlah kemungkinan input yang disertakan dan jumlah output
        # neuron pada model yang akan dibuat
        #
        self.io_size = len(self.vocab_to_int)

        #
        # Apakah model ini digunakan untuk training atau dipakai untuk test?
        #
        if training == True:
            self.sequence_size = self.SEQUENCE_SIZE
            self.batch_size = self.BATCH_SIZE
        else:
            self.sequence_size = 1
            self.batch_size = 1

        #
        # Membangung graph network dengan LSTM
        # silahkan lihat di http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        #

        # placeholders
        self.inputs, self.targets, self.keep_prob = self.build_inputs() # Placeholders
        cell_output, self.initial_state, self.final_state = self.build_lstm(self.inputs, self.keep_prob) # LSTM
        self.softmax, logits = self.build_output(cell_output) # LSTM output
        self.loss = self.build_loss(logits, self.targets) # Cost function
        self.optimizer = self.build_optimizer(self.loss) # Optimizer

        # Membangun sessi
        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.session = tf.Session()

        self.session.run(init)

        # Ini untuk Tensorboard
        self.tensorboard = tf.summary.merge_all()
        model_param = "bs_%s_hs_%s_stem_size_%s_lr_%s" % (self.BATCH_SIZE, self.HIDDEN_LAYER_SIZE, self.LSTM_SIZE, self.LR)
        self.train_writer = tf.summary.FileWriter(os.path.join(self.TENSORBOARD_LOG, model_param), self.session.graph)
        self.train_writer_it = 0

        self.model_it = 0

        # Jika isi checkpoints tidak kosong
        if ckpt is not None:
            self.saver.restore(self.session, ckpt)
            print("[-] model checkpoints berhasil dipasang..")


    def get_new_state(self):
        return self.session.run(self.initial_state)

    #
    # Optimasi model dan mengembalikan nilai loss pada batch ini
    # kemudian cell terakhir akan menjadi patokan pada RNN untuk input :
    # *x (Numpy Array)
    # *y (Numpy Array)
    #
    def optimize_model(self, x, y, state):
        summary, batch_loss, n_state, _ = self.session.run([self.tensorboard, self.loss, self.final_state, self.optimizer], feed_dict={self.inputs: x, self.targets: y, self.keep_prob: self.KEEP_PROB, self.initial_state: state})
        self.train_writer.add_summary(summary, self.train_writer_it)
        self.train_writer_it += 1
        return batch_loss, n_state, _

    #
    # Simpan model yang berhasil dibuat dengan loss diatur
    #
    def save_model(self):
        self.model_it += 1
        print("[*] Simpan model sukses!")
        self.saver.save(self.session, "checkpoints/dump_model_%s.ckpt" % self.model_it)
        print('[*] Model berhasil disimpan!')

    #
    # ini adalah tambahan method di dalam class untuk pengambilan dataset file
    #
    @staticmethod
    def get_dataset():

        if os.path.isfile(Model.DATASET_FILE):
            with open(Model.DATASET_FILE, "r") as d:
                d = json.loads(d.read())
                return np.array(d["encoded"]), d["vocab_to_int"], d["int_to_vocab"]

        #
        # Ambil seluruh file pada folder dataset
        # dan pastikan file tersebut python (*.py)
        #
        all_file = os.listdir(Model.DATASET_FOLDER)
        all_file_name = np.array([f for f in all_file if f.find(PROG_LANG_EXT) != -1])

        # buka folder
        content = ""
        for name in all_file_name:
            with open(os.path.join(Model.DATASET_FOLDER, name), "r") as f:
                content += f.read() + "\n"

        # Convert string kedalam int pada list
        vocab = set(content)
        vocab_to_int = {c: i for i, c in enumerate(vocab)}
        int_to_vocab = dict(enumerate(vocab))
        encoded = np.array([vocab_to_int[c] for c in content], dtype=np.int32)

        #
        # Buat file dataset dengan isi vocal dan int
        #
        with open(Model.DATASET_FILE, "w+") as d:
            json_file = json.dumps({"encoded" : [int(i) for i in encoded], "vocab_to_int" : vocab_to_int, "int_to_vocab" : int_to_vocab})
            d.write(json_file)
            d.close()

        return encoded, vocab_to_int, int_to_vocab


    #
    # Membuat RNN model dengan LSTM cell dimana cell neuronya di-null terlebih dahulu
    # kemudian dikembalikan kedalam operasi tensorflow
    #
    def build_lstm(self, inputs, keep_prob):
        with tf.name_scope("LSTM"):
            def create_cell():
                lstm = tf.contrib.rnn.BasicLSTMCell(self.HIDDEN_LAYER_SIZE)
                drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                return drop

            cell = tf.contrib.rnn.MultiRNNCell([create_cell() for _ in range(self.LSTM_SIZE)])
            initial_state = cell.zero_state(self.batch_size, tf.float32)

            x_one_hot = tf.one_hot(inputs, self.io_size)
            cell_outputs, final_state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)

        return cell_outputs, initial_state, final_state

    #
    # Output layer pada RNN degan softmax dan logits
    #
    def build_output(self, cell_outputs):
        with tf.name_scope("graph_outputs"):
            #seq_output = tf.concat(cell_outputs, axis=1, name="concat_all_cell_outputs")
            x = tf.reshape(cell_outputs, [-1, self.HIDDEN_LAYER_SIZE], name="reshape_x")

            with tf.name_scope('output_layer'):
                w = tf.Variable(tf.truncated_normal((self.HIDDEN_LAYER_SIZE, self.io_size), stddev=0.1), name="weights")
                b = tf.Variable(tf.zeros(self.io_size), name="bias")
                tf.summary.histogram("weights", w)
                tf.summary.histogram("bias", b)

            logits = tf.add(tf.matmul(x , w), b, name= "logits")
            softmax = tf.nn.softmax(logits, name='predictions')
            tf.summary.histogram("softmax", softmax)

        return softmax, logits

    #
    # Definisikan terlebih dahulu loss yang akan terjadi untuk di reshaping
    #
    def build_loss(self, logits, targets):

        with tf.name_scope("cost"):
            y_one_hot = tf.one_hot(targets, self.io_size, name="y_to_one_hot")
            y_reshaped = tf.reshape(y_one_hot, logits.get_shape(), name="reshape_one_hot")
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
            loss = tf.reduce_mean(loss)
            tf.summary.scalar('loss', loss)

        return loss

    #
    # Meminimalisir kemungkinan loss dengan Adam Optimizer
    # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
    #
    def build_optimizer(self, loss):
        with tf.name_scope("train"):
            adam = tf.train.AdamOptimizer(self.LR)
            optimizer = adam.minimize(loss)

        return optimizer

    #
    # Placeholder untuk TF neurons
    #
    def build_inputs(self):
        with tf.name_scope("graph_inputs"):
            inputs = tf.placeholder(tf.int32, [None, self.sequence_size], name='placeholder_inputs')
            targets = tf.placeholder(tf.int32, [None, self.sequence_size], name='placeholder_targets')
            keep_prob = tf.placeholder(tf.float32, name='placeholder_keep_prob')

        return inputs, targets, keep_prob




