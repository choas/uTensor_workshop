# uTensor Workshop

[Why Machine Learning on The Edge?](https://towardsdatascience.com/why-machine-learning-on-the-edge-92fac32105e6)

## Teil 1 - Keras Überblick und erstes ML Modell

```sh
git clone https://github.com/choas/uTensor_workshop.git
cd uTensor_workshop
```


### Ziel Jupyter Notebooks installieren

__Python 2.7__ sollte bereits installiert sein, ansonsten: [Download Python 2.7.16](https://www.python.org/downloads/release/python-2716/)

#### [Virtualenv](https://virtualenv.pypa.io/en/latest/) installieren

```sh
pip install virtualenv
```

```sh
cd python
virtualenv . -p <PYTHON2.7>
source bin/activate
python --version
```

#### noch mehr Python installieren

TensorFlow, Keras, Jupyter und TensorFlow.js installieren:

```sh
pip install -r requirement.txt
```

#### Warum die TensorFlow und Keras Versionen?

1. Keras + TensorFlow = 🤬
2. [Welche Version passt mit welcher Version zusammen?](https://docs.floydhub.com/guides/environments/)
3. [Sunsetting tf.contrib](https://github.com/tensorflow/community/blob/master/rfcs/20180907-contrib-sunset.md)

#### Jupyter Notebook starten

```sh
cd notebooks
jupyter notebook
```

Der (Default) Browser öffnet automatisch die Seite http://localhost:8888

## Erstes Beispiel mit [Keras](https://keras.io/)

Was ist supervised und unsupervised learning?

[01_XOR_Keras.ipynb](http://localhost:8888/notebooks/01_XOR_Keras.ipynb) öffnen

mit Run Schritt für Schritt ausführen 


keras 2.2.4
tensorflow 1.12.0


$$$

Ergebnis

[[0.04774003]
 [0.9481221 ]
 [0.9517887 ]
 [0.0586575 ]]


Imports

TensorFlow und Keras Session

sess=tf.Session()
K.set_session(sess)

Trainings- und Zieldaten
training_data = np.array([[0,0], [0,1], [1,0], [1,1]])
target_data   = np.array([  [0],   [1],   [1],   [0]])

Keras Modell

model = Sequential()
model.add(Dense(4, input_dim=2, activation='sigmoid', name='input-layer'))
model.add(Dense(1, activation='sigmoid', name='output-layer'))
model.compile(loss='mean_squared_error', optimizer=SGD(lr=1), metrics=['accuracy'])

input-layer mit hidden-layer
Dense = kompakt
4 Hidden-Nodes (Knoten)
2 Eingangswerte für die Trainingsdaten
Sigmoid Aktivierung: https://de.wikipedia.org/wiki/Sigmoidfunktion
https://en.wikipedia.org/wiki/File:Logistic-curve.svg
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6


output-layer
1 Ausgabe-Node fur die Zieldaten
Sigmoid Aktivierung

loss-Funktion
mean_squared_error

Optimier-Funktion
Stochastic gradient descent: https://en.wikipedia.org/wiki/Stochastic_gradient_descent
https://martin-thoma.com/tf-xor-tutorial/

TensorBoard
https://keras.io/callbacks/#tensorboard


Training mit 2000 Epochs


TensorBoard
source bin/activate
tensorboard --logdir notebooks/logs


http://localhost:6006/

Graph
<Bild>


Modell verbessern

activation
loss
optimizer



#XOR Modell mit TensorFlow

<logo>
https://github.com/tensorflow/tensorflow


https://hackernoon.com/9-things-you-should-know-about-tensorflow-9cf0a05e4995


02_XOR_TF Notebook

%%original: https://aimatters.wordpress.com/2016/01/16/solving-xor-with-a-neural-network-in-tensorflow/

Tensorflow Operator

placeholder
Variable
sigmoid  !!!
reduce_mean
GradientDescentOptimizer


TensorBoard
<Bild>

#mehr Modelle
https://github.com/keras-team/keras/tree/master/examples
https://github.com/tensorflow/models/tree/master/research


#TensorFlow.js

https://www.tensorflow.org/js/
https://github.com/tensorflow/tfjs-converter

https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e

pip install tensorflowjs==0.8.0

https://pypi.org/project/tensorflowjs/0.8.0/

%% pip freeze | xargs pip uninstall -y


https://www.jsdelivr.com/package/npm/@tensorflow/tfjs



python -m SimpleHTTPServer 8000
http://localhost:8000/


https://github.com/tensorflow/tfjs-examples


#Save Modelle
https://www.tensorflow.org/api_docs/python/tf/graph_util/convert_variables_to_constants


# MNIST

![Fully connected 2 layer NN](https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png)

https://github.com/tensorflow/tfjs-examples/tree/master/mnist-transfer-cnn


03_MNIST_TF Notebook

https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/mlp_mnist.png

%% https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721

#uTensor
https://blog.hackster.io/simple-neural-network-on-mcus-a7cbd3dc108c

https://raw.githubusercontent.com/uTensor/uTensor/develop/docs/img/uTensorFlow.jpg


??? Cloud9
%% https://github.com/uTensor/cloud9-installer


cd python
source bin/activate


mbed
https://github.com/ARMmbed/mbed-os
https://os.mbed.com/docs/mbed-os/v5.12/introduction/index.html

GCC-arm cross-compiler:
brew install https://raw.githubusercontent.com/osx-cross/homebrew-arm/0a6179693c15d8573360c94cee8a60bdf142f7b4/arm-gcc-bin.rb


brew install mercurial git





pip install mbed-cli==1.9.1 utensor_cgen

pip install -U protobuf
%% https://github.com/tensorflow/models/issues/3995


cd ../code

mbed new MNIST

cd MNIST

mbed add https://github.com/uTensor/uTensor

utensor-cli convert ../../python/notebooks/models/deep_mlp.pb --output-nodes=y_pred


%%https://gist.githubusercontent.com/neil-tan/4a41505f5c06f079e36b159137ad1bdd/raw/84cd6ff5078d3b34797ffa4003aaae49ab895c26/main.cpp

%%wget https://gist.github.com/neil-tan/0e032be578181ec0e3d9a47e1e24d011/raw/888d098683318d030b3c4f6f4b375a64e7ad0017/input_data.h


https://os.mbed.com/platforms/ST-Nucleo-F411RE/


mbed compile -m nucleo_f411re -t GCC_ARM --profile=uTensor/build_profile/release.json


mbed-os/platform/mbed_rtc_time.h
delete #if !defined(__GNUC__) || defined(__CC_ARM) || defined(__clang__)
und #endif

%% https://github.com/ARMmbed/mbed-os/issues/6988

cp ./BUILD/NUCLEO_F411RE/GCC_ARM-RELEASE/MNIST.bin /Volumes/

screen /dev/cu.usbmodem14603 115200

Reset Board

control AK



#mbed XOR

mbed new XOR

mbed add https://github.com/uTensor/uTensor

utensor-cli convert ../../python/notebooks/models/xor_tf.pb --output-nodes=layer3/Sigmoid

unsupported op type in uTensor: Sigmoid



04_XOR_TF_relu Notebook


utensor-cli convert ../../python/notebooks/models/xor_relu.pb --output-nodes=add_1


Softmax
https://www.tensorflow.org/api_docs/python/tf/nn/softmax
https://de.wikipedia.org/wiki/Softmax-Funktion


models/*
main.cpp


mbed compile -m nucleo_f411re -t GCC_ARM --profile=uTensor/build_profile/release.json

#weitere Beispiele

https://github.com/uTensor/ADL_demo
https://github.com/uTensor/utensor-cmsis-example
https://github.com/uTensor/utensor-mnist-demo
