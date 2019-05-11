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

=============
untuk docopt
=============

Usage:
  maskoding_v1.py <ckpt> <numchar>

Options:
  -h --help                 Show this help.
  <ckpt>                    Path to the ckpt file to use
  <numchar>                 Number of characters in the file

'''
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)

import numpy as np
from model import Model
from docopt import docopt

print("\n============================================")
print(" MASKODING - v1 | Auto Python Code Generator")
print("============================================\n")


#
# Class untuk pembuatan file python dengan model
# pilih terlebih dahulu model (ckpt) yang diinginkan
#
class Use(Model):
    def __init__(self, ckpt):
        Model.__init__(self, training=False, ckpt=ckpt)

        self.code = [c for c in "# -*- coding: utf-8 -*- \n\n ## Percobaan Auto Coding python - (c)2019 _drat ## \n\n\n"]


    #
    # Berdasarkan softmax, tambahkan huruf yang menjadi dasar coding
    #
    def add_to_code(self, softmax):
        softmax = np.squeeze(softmax)
        softmax[np.argsort(softmax)[:-4]] = 0
        softmax = softmax / np.sum(softmax)
        c = np.random.choice(len(self.int_to_vocab), 1, p=softmax)[0]
        self.code.append(self.int_to_vocab[str(c)])
        return c

    #
    # Pembuatan kode ajaib di sini hehehehehehehe
    # *size(int) adalah jumlah karakter kode yang ingin dibuat
    # return pastinya str
    #
    def create_code(self, size):
        previous_state = self.session.run(self.initial_state)

        for c in self.code:
            x = np.array([[self.vocab_to_int[c]]])
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: previous_state}
            softmax, previous_state = self.session.run([self.softmax, self.final_state], feed_dict=feed)

        c = self.add_to_code(softmax)

        for i in range(size):
            x[0,0] = c
            feed = {self.inputs: x, self.keep_prob: 1., self.initial_state: previous_state}
            softmax, previous_state = self.session.run([self.softmax, self.final_state], feed_dict=feed)
            c = self.add_to_code(softmax)

        return ''.join(self.code)

#
# Mulai
#
if __name__ == '__main__':
    arguments = docopt(__doc__)

    use_model = Use(arguments["<ckpt>"])
    code = use_model.create_code(int(arguments["<numchar>"]))
    print("[-] Nama Model : ", arguments["<ckpt>"])
    print("[-] Jumlah Karakter : ", arguments["<numchar>"])

    with open("program_robot.py", "w+") as s:
        s.write(code)
        s.close()

    print("[*] File maskoding berhasil dibuat!\n")




