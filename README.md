# WD-MASKODING
PT. Warung Data Indonesia

![]image/1111.jpg

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

# WARNING!!

Masih sangat konseptual dan harus diriset dan dikembangkan lebih jauh lagi dengan
konsep pemodelan pada neural network yang lebih konsisten.

