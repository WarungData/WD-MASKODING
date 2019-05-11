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
import time
import matplotlib.pyplot as plt
from model import Model


#
# Class untuk melakukan pelatihan pada model
# dengan meng-inisiasikan bacthes yang ditentukan di init (bawah)
#
class Train(Model):

    def __init__(self):
        Model.__init__(self, training=True)

        print("[-] Inisiasi batches...")
        self.init_dataset()
        print("[-] Batches siap dilatih.")
        print("\n")

    #
    # Mengambil contoh-contoh file-file python yang ada di dalam folder dataset
    # dan melakukan proses batches pada setiap file
    # serta memisahkan antara encode dan jumlah sequence
    #
    def init_dataset(self):
        """
            Init the dataset and store it inside the DataContainer class
        """
        self.batch_inputs = []
        self.batch_targets = []
        encoded = self.encoded

        batch_size = self.BATCH_SIZE * self.SEQUENCE_SIZE
        n_batches = len(self.encoded) // batch_size
        encoded = encoded[:n_batches * batch_size]
        encoded = encoded.reshape((self.BATCH_SIZE, -1))

        for n in range(0, encoded.shape[1], self.SEQUENCE_SIZE):

            x = encoded[:, n:n+self.SEQUENCE_SIZE]
            y = np.zeros_like(x)
            y[:, :-1], y[:, -1] = x[:, 1:], x[:, 0]

            self.batch_inputs.append(x)
            self.batch_targets.append(y)

    #
    # Memulai pelatihan model
    # dengan input -> *epoch (int) adalah jumlah epoch
    #
    def train_model(self, epochs):

        for e in range(epochs):

            # Pelatihan dimulai
            # untuk network yang ditentukan loss nya [2,10]
            #
            n_state = self.get_new_state()

            for inputs, targets in zip(self.batch_inputs, self.batch_targets):
                t = time.time()
                batch_loss, n_state, _ = self.optimize_model(inputs, targets, n_state)
                print("[-] Epoch > %s, Training loss : %2f, sec/batch : %2f" % (e, batch_loss, (time.time() - t)))

            # Simpan model pada setiap epoch yang dihasilkan
            print("[-] Simpan model ...")
            self.save_model()



#
# Mulai init pelatihan model
#
if __name__ == '__main__':
    tr = Train()

    # masukan jumlah epoch
    tr.train_model(20)

