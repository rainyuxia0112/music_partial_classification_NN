{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Lab:  Neural Networks for Music Classification\n",
    "\n",
    "Due June 5 at 11:59pm.  This lab may be done alone or in groups up to four.  Everyone should submit their own lab, but indicate on your submission if you worked with others.\n",
    "\n",
    "In addition to the concepts in the MNIST neural network demo (posted in the Demo section of CCLE), in this lab, you will learn to:\n",
    "\n",
    "* Load a file from a URL\n",
    "* Extract simple features from audio samples for machine learning tasks such as speech recognition and classification\n",
    "* Build a simple neural network for music classification using these features\n",
    "* Use a callback to store the loss and accuracy history in the training process\n",
    "* Optimize the learning rate of the neural network\n",
    "\n",
    "To illustrate the basic concepts, we will look at a relatively simple music classification problem.  Given a sample of music, we want to determine which instrument (e.g. trumpet, violin, piano) is playing.  This dataset was generously supplied by [Prof. Juan Bello](http://steinhardt.nyu.edu/faculty/Juan_Pablo_Bello) at NYU Stenihardt  and his former PhD student Eric Humphrey (now at Spotify).  They have a complete website dedicated to deep learning methods in music informatics:\n",
    "\n",
    "http://marl.smusic.nyu.edu/wordpress/projects/feature-learning-deep-architectures/deep-learning-python-tutorial/\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Loading Tensorflow\n",
    "\n",
    "Before starting this lab, you will need to install [Tensorflow](https://www.tensorflow.org/install/).  If you are using [Google colaboratory](https://colab.research.google.com), Tensorflow is already installed.  Run the following command to ensure Tensorflow is installed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Then, load the other packages."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Audio Feature Extraction with Librosa\n",
    "\n",
    "The key to audio classification is to extract the correct features. In addition to `keras`, we will need the `librosa` package.  The `librosa` package in python has a rich set of methods extracting the features of audio samples commonly used in machine learning tasks such as speech recognition and sound classification. \n",
    "\n",
    "Installation instructions and complete documentation for the package are given on the [librosa main page](https://librosa.github.io/librosa/).  On most systems, you should be able to simply use:\n",
    "\n",
    "    pip install -u librosa\n",
    "    \n",
    "After you have installed the package, try to import it."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "import librosa\n",
    "import librosa.display\n",
    "import librosa.feature"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In this lab, we will use a set of music samples from the website:\n",
    "\n",
    "http://theremin.music.uiowa.edu\n",
    "\n",
    "This website has a great set of samples for audio processing.  Look on the web for how to use the `requests.get` and `file.write` commands to load the file at the URL provided into your working directory.\n",
    "\n",
    "You can play the audio sample by copying the file to your local machine and playing it on any media player.  If you listen to it you will hear a soprano saxaphone (with vibrato) playing four notes (C, C#, D, Eb)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1418242"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import requests\n",
    "fn = \"SopSax.Vib.pp.C6Eb6.aiff\"\n",
    "url = \"http://theremin.music.uiowa.edu/sound files/MIS/Woodwinds/sopranosaxophone/\"+fn\n",
    "\n",
    "# TODO:  Load the file from url and save it in a file under the name fn\n",
    "r = requests.get(url, allow_redirects=True)\n",
    "open('fn.aiff', 'wb').write(r.content)     #file.write 可以load任何形式的文件，不一定是txt"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Next, use `librosa` command `librosa.load` to read the audio file with filename `fn` and get the samples `y` and sample rate `sr`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO\n",
    "y, sr = librosa.load('fn.aiff')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Extracting features from audio files is an entire subject on its own right.  A commonly used set of features are called the Mel Frequency Cepstral Coefficients (MFCCs).  These are derived from the so-called mel spectrogram which is something like a regular spectrogram, but the power and frequency are represented in log scale, which more naturally aligns with human perceptual processing.  You can run the code below to display the mel spectrogram from the audio sample.\n",
    "\n",
    "You can easily see the four notes played in the audio track.  You also see the 'harmonics' of each notes, which are other tones at integer multiples of the fundamental frequency of each note."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)\n",
    "librosa.display.specshow(librosa.amplitude_to_db(S),\n",
    "                         y_axis='mel', fmax=8000, x_axis='time')\n",
    "plt.colorbar(format='%+2.0f dB')\n",
    "plt.title('Mel spectrogram')\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Downloading the Data\n",
    "\n",
    "Using the MFCC features described above, Eric Humphrey and Juan Bellow have created a complete data set that can used for instrument classification.  Essentially, they collected a number of data files from the website above.  For each audio file, the segmented the track into notes and then extracted 120 MFCCs for each note.  The goal is to recognize the instrument from the 120 MFCCs.  The process of feature extraction is quite involved.  So, we will just use their processed data provided at:\n",
    "\n",
    "https://github.com/marl/dl4mir-tutorial/blob/master/README.md\n",
    "\n",
    "Note the password.  Load the four files into some directory, say  `instrument_dataset`.  Then, load them with the commands."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dir = 'instrument_dataset/'\n",
    "Xtr = np.load(data_dir+'uiowa_train_data.npy')\n",
    "ytr = np.load(data_dir+'uiowa_train_labels.npy')\n",
    "Xts = np.load(data_dir+'uiowa_test_data.npy')\n",
    "yts = np.load(data_dir+'uiowa_test_labels.npy')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Looking at the data files:\n",
    "* What are the number of training and test samples?\n",
    "* What is the number of features for each sample?\n",
    "* How many classes (i.e. instruments) are there per class?\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Before continuing, you must scale the training and test data, `Xtr` and `Xts`.  Compute the mean and std deviation of each feature in `Xtr` and create a new training data set, `Xtr_scale`, by subtracting the mean and dividing by the std deviation.  Also compute a scaled test data set, `Xts_scale` using the mean and std deviation learned from the training data set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO Scale the training and test matrices\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "Xtr_scale = scaler.fit_transform(Xtr)\n",
    "Xts_scale = scaler.transform(Xts)\n",
    "# other method\n",
    "# TODO Scale the training and test matrices\n",
    "train_mean = np.mean(Xtr, axis = 0)\n",
    "train_std = np.std(Xtr, axis = 0)\n",
    "Xtr_scale = (Xtr - train_mean)/train_std\n",
    "Xts_scale = (Xts - train_mean)/train_std"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.utils import to_categorical\n",
    "train_labels=to_categorical(ytr) \n",
    "test_labels=to_categorical(yts) "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "## Building a Neural Network Classifier\n",
    "\n",
    "Following the example in [MNIST neural network demo](./mnist_neural.ipynb), clear the keras session.  Then, create a neural network `model` with:\n",
    "* `nh=256` hidden units\n",
    "* `sigmoid` activation\n",
    "* select the input and output shapes correctly\n",
    "* print the model summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [],
   "source": [
    "from tensorflow.keras.models import Model, Sequential\n",
    "from tensorflow.keras.layers import Dense, Activation\n",
    "import tensorflow.keras.backend as K"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO clear session\n",
    "K.clear_session()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO: construct the model\n",
    "model = Sequential()\n",
    "model.add(Dense(256, activation = 'sigmoid', input_shape = (120,)))\n",
    "model.add(Dense(10, activation = 'softmax'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "_________________________________________________________________\n",
      "Layer (type)                 Output Shape              Param #   \n",
      "=================================================================\n",
      "dense (Dense)                (None, 256)               30976     \n",
      "_________________________________________________________________\n",
      "dense_1 (Dense)              (None, 10)                2570      \n",
      "=================================================================\n",
      "Total params: 33,546\n",
      "Trainable params: 33,546\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "# TODO:  Print the model summary\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Create an optimizer and compile the model.  Select the appropriate loss function and metrics.  For the optimizer, use the Adam optimizer with a learning rate of 0.001"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# TODO\n",
    "from tensorflow.keras.optimizers import Adam\n",
    "opt = Adam(lr = 0.001, decay=1e-6)\n",
    "model.compile(\n",
    "    loss='categorical_crossentropy',\n",
    "    optimizer= opt, \n",
    "    metrics = ['acc']\n",
    ") "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Fit the model for 10 epochs using the scaled data for both the training and validation.  Use the `validation_data` option to pass the test data.  Also, pass the callback class create above.  Use a batch size of 100.  Your final accuracy should be >99%."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 66247 samples, validate on 14904 samples\n",
      "Epoch 1/10\n",
      "66247/66247 [==============================] - 8s 124us/sample - loss: 0.3684 - acc: 0.8991 - val_loss: 0.1819 - val_acc: 0.9606\n",
      "Epoch 2/10\n",
      "66247/66247 [==============================] - 7s 103us/sample - loss: 0.1041 - acc: 0.9753 - val_loss: 0.0946 - val_acc: 0.9767\n",
      "Epoch 3/10\n",
      "66247/66247 [==============================] - 8s 116us/sample - loss: 0.0603 - acc: 0.9859 - val_loss: 0.0694 - val_acc: 0.9849\n",
      "Epoch 4/10\n",
      "66247/66247 [==============================] - 8s 128us/sample - loss: 0.0423 - acc: 0.9894 - val_loss: 0.0579 - val_acc: 0.9836\n",
      "Epoch 5/10\n",
      "66247/66247 [==============================] - 9s 129us/sample - loss: 0.0320 - acc: 0.9918 - val_loss: 0.0430 - val_acc: 0.9889\n",
      "Epoch 6/10\n",
      "66247/66247 [==============================] - 9s 130us/sample - loss: 0.0254 - acc: 0.9931 - val_loss: 0.0376 - val_acc: 0.9886\n",
      "Epoch 7/10\n",
      "66247/66247 [==============================] - 9s 138us/sample - loss: 0.0208 - acc: 0.9942 - val_loss: 0.0350 - val_acc: 0.9893\n",
      "Epoch 8/10\n",
      "66247/66247 [==============================] - 9s 129us/sample - loss: 0.0171 - acc: 0.9956 - val_loss: 0.0360 - val_acc: 0.9879\n",
      "Epoch 9/10\n",
      "66247/66247 [==============================] - 9s 131us/sample - loss: 0.0147 - acc: 0.9960 - val_loss: 0.0291 - val_acc: 0.9900\n",
      "Epoch 10/10\n",
      "66247/66247 [==============================] - 8s 128us/sample - loss: 0.0127 - acc: 0.9970 - val_loss: 0.0376 - val_acc: 0.9866\n"
     ]
    }
   ],
   "source": [
    "# TODO\n",
    "hist = model.fit(Xtr_scale, train_labels, epochs=10, batch_size=100, validation_data=(Xts_scale, test_labels))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot the validation accuracy saved in `hist.history` dictionary. This gives one accuracy value per epoch.  You should see that the validation accuracy saturates at a little higher than 99%.  After that it \"bounces around\" due to the noise in the stochastic gradient descent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# TODO\n",
    "import matplotlib.pyplot as plt\n",
    "plt.clf()\n",
    "epochs = range(1, 11)\n",
    "val_acc = hist.history['val_acc']\n",
    "plt.plot(epochs, val_acc, 'b', label = 'validation acc')\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('acc')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot the loss values saved in the `hist.history` dictionary.  You should see that the loss is steadily decreasing.  Use the `semilogy` plot."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "# TODO\n",
    "val_loss = hist.history['val_loss']\n",
    "plt.semilogy(epochs, val_loss)\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('val_loss')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Optimizing the Learning Rate\n",
    "\n",
    "One challenge in training neural networks is the selection of the learning rate.  Rerun the above code, trying four learning rates as shown in the vector `rates`.  For each learning rate:\n",
    "* clear the session\n",
    "* construct the network\n",
    "* select the optimizer.  Use the Adam optimizer with the appropriate learrning rate.\n",
    "* train the model for 20 epochs\n",
    "* save the accuracy and losses"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 66247 samples, validate on 14904 samples\n",
      "Epoch 1/20\n",
      "66247/66247 [==============================] - 10s 146us/sample - loss: 0.1048 - acc: 0.9678 - val_loss: 0.0460 - val_acc: 0.9854\n",
      "Epoch 2/20\n",
      "66247/66247 [==============================] - 8s 121us/sample - loss: 0.0293 - acc: 0.9904 - val_loss: 0.0269 - val_acc: 0.9903\n",
      "Epoch 3/20\n",
      "66247/66247 [==============================] - 7s 108us/sample - loss: 0.0213 - acc: 0.9931 - val_loss: 0.0461 - val_acc: 0.9815\n",
      "Epoch 4/20\n",
      "66247/66247 [==============================] - 7s 110us/sample - loss: 0.0153 - acc: 0.9949 - val_loss: 0.0492 - val_acc: 0.9826\n",
      "Epoch 5/20\n",
      "66247/66247 [==============================] - 8s 116us/sample - loss: 0.0168 - acc: 0.9947 - val_loss: 0.0982 - val_acc: 0.9732\n",
      "Epoch 6/20\n",
      "66247/66247 [==============================] - 9s 133us/sample - loss: 0.0149 - acc: 0.9951 - val_loss: 0.0271 - val_acc: 0.9903\n",
      "Epoch 7/20\n",
      "66247/66247 [==============================] - 9s 143us/sample - loss: 0.0115 - acc: 0.9962 - val_loss: 0.0359 - val_acc: 0.9895\n",
      "Epoch 8/20\n",
      "66247/66247 [==============================] - 9s 130us/sample - loss: 0.0145 - acc: 0.9956 - val_loss: 0.0273 - val_acc: 0.9914\n",
      "Epoch 9/20\n",
      "66247/66247 [==============================] - 9s 136us/sample - loss: 0.0117 - acc: 0.9965 - val_loss: 0.0597 - val_acc: 0.9843\n",
      "Epoch 10/20\n",
      "66247/66247 [==============================] - 9s 138us/sample - loss: 0.0119 - acc: 0.9962 - val_loss: 0.0383 - val_acc: 0.9894\n",
      "Epoch 11/20\n",
      "66247/66247 [==============================] - 8s 126us/sample - loss: 0.0109 - acc: 0.9964 - val_loss: 0.0386 - val_acc: 0.9889\n",
      "Epoch 12/20\n",
      "66247/66247 [==============================] - 8s 128us/sample - loss: 0.0098 - acc: 0.9969 - val_loss: 0.0414 - val_acc: 0.9883\n",
      "Epoch 13/20\n",
      "66247/66247 [==============================] - 7s 110us/sample - loss: 0.0102 - acc: 0.9969 - val_loss: 0.0652 - val_acc: 0.9809\n",
      "Epoch 14/20\n",
      "66247/66247 [==============================] - 8s 119us/sample - loss: 0.0102 - acc: 0.9969 - val_loss: 0.0450 - val_acc: 0.9903\n",
      "Epoch 15/20\n",
      "66247/66247 [==============================] - 8s 121us/sample - loss: 0.0088 - acc: 0.9975 - val_loss: 0.0347 - val_acc: 0.9903\n",
      "Epoch 16/20\n",
      "66247/66247 [==============================] - 8s 126us/sample - loss: 0.0102 - acc: 0.9970 - val_loss: 0.0356 - val_acc: 0.9913\n",
      "Epoch 17/20\n",
      "66247/66247 [==============================] - 8s 118us/sample - loss: 0.0075 - acc: 0.9977 - val_loss: 0.0431 - val_acc: 0.9899\n",
      "Epoch 18/20\n",
      "66247/66247 [==============================] - 8s 124us/sample - loss: 0.0089 - acc: 0.9974 - val_loss: 0.0830 - val_acc: 0.9826\n",
      "Epoch 19/20\n",
      "66247/66247 [==============================] - 8s 121us/sample - loss: 0.0081 - acc: 0.9975 - val_loss: 0.1103 - val_acc: 0.9781\n",
      "Epoch 20/20\n",
      "66247/66247 [==============================] - 8s 116us/sample - loss: 0.0074 - acc: 0.9978 - val_loss: 0.0512 - val_acc: 0.9868\n",
      "Train on 66247 samples, validate on 14904 samples\n",
      "Epoch 1/20\n",
      "66247/66247 [==============================] - 8s 119us/sample - loss: 0.3638 - acc: 0.9008 - val_loss: 0.1959 - val_acc: 0.9370\n",
      "Epoch 2/20\n",
      "66247/66247 [==============================] - 8s 124us/sample - loss: 0.1021 - acc: 0.9754 - val_loss: 0.0881 - val_acc: 0.9798\n",
      "Epoch 3/20\n",
      "66247/66247 [==============================] - 7s 110us/sample - loss: 0.0604 - acc: 0.9851 - val_loss: 0.0586 - val_acc: 0.9874\n",
      "Epoch 4/20\n",
      "66247/66247 [==============================] - 8s 117us/sample - loss: 0.0425 - acc: 0.9891 - val_loss: 0.0518 - val_acc: 0.9858\n",
      "Epoch 5/20\n",
      "66247/66247 [==============================] - 8s 116us/sample - loss: 0.0322 - acc: 0.9915 - val_loss: 0.0412 - val_acc: 0.9897\n",
      "Epoch 6/20\n",
      "66247/66247 [==============================] - 8s 113us/sample - loss: 0.0257 - acc: 0.9934 - val_loss: 0.0334 - val_acc: 0.9897\n",
      "Epoch 7/20\n",
      "66247/66247 [==============================] - 7s 112us/sample - loss: 0.0208 - acc: 0.9943 - val_loss: 0.0423 - val_acc: 0.9865\n",
      "Epoch 8/20\n",
      "66247/66247 [==============================] - 7s 109us/sample - loss: 0.0175 - acc: 0.9956 - val_loss: 0.0353 - val_acc: 0.9889\n",
      "Epoch 9/20\n",
      "66247/66247 [==============================] - 7s 104us/sample - loss: 0.0148 - acc: 0.9963 - val_loss: 0.0299 - val_acc: 0.9903\n",
      "Epoch 10/20\n",
      "66247/66247 [==============================] - 7s 112us/sample - loss: 0.0128 - acc: 0.9967 - val_loss: 0.0267 - val_acc: 0.9917\n",
      "Epoch 11/20\n",
      "66247/66247 [==============================] - 8s 118us/sample - loss: 0.0115 - acc: 0.9971 - val_loss: 0.0292 - val_acc: 0.9898\n",
      "Epoch 12/20\n",
      "66247/66247 [==============================] - 7s 111us/sample - loss: 0.0102 - acc: 0.9974 - val_loss: 0.0260 - val_acc: 0.9912\n",
      "Epoch 13/20\n",
      "66247/66247 [==============================] - 7s 108us/sample - loss: 0.0088 - acc: 0.9979 - val_loss: 0.0240 - val_acc: 0.9912\n",
      "Epoch 14/20\n",
      "66247/66247 [==============================] - 7s 112us/sample - loss: 0.0084 - acc: 0.9978 - val_loss: 0.0222 - val_acc: 0.9915\n",
      "Epoch 15/20\n",
      "66247/66247 [==============================] - 8s 115us/sample - loss: 0.0077 - acc: 0.9979 - val_loss: 0.0238 - val_acc: 0.9915\n",
      "Epoch 16/20\n",
      "66247/66247 [==============================] - 7s 109us/sample - loss: 0.0068 - acc: 0.9982 - val_loss: 0.0212 - val_acc: 0.9925\n",
      "Epoch 17/20\n",
      "66247/66247 [==============================] - 7s 110us/sample - loss: 0.0062 - acc: 0.9983 - val_loss: 0.0281 - val_acc: 0.9897\n",
      "Epoch 18/20\n",
      "66247/66247 [==============================] - 8s 115us/sample - loss: 0.0060 - acc: 0.9984 - val_loss: 0.0259 - val_acc: 0.9903\n",
      "Epoch 19/20\n",
      "66247/66247 [==============================] - 7s 105us/sample - loss: 0.0053 - acc: 0.9985 - val_loss: 0.0237 - val_acc: 0.9908\n",
      "Epoch 20/20\n",
      "66247/66247 [==============================] - 7s 112us/sample - loss: 0.0050 - acc: 0.9987 - val_loss: 0.0224 - val_acc: 0.9921\n",
      "Train on 66247 samples, validate on 14904 samples\n",
      "Epoch 1/20\n",
      "66247/66247 [==============================] - 8s 126us/sample - loss: 1.1122 - acc: 0.6600 - val_loss: 0.8596 - val_acc: 0.6758\n",
      "Epoch 2/20\n",
      "66247/66247 [==============================] - 7s 107us/sample - loss: 0.5531 - acc: 0.8531 - val_loss: 0.5803 - val_acc: 0.8186\n",
      "Epoch 3/20\n",
      "66247/66247 [==============================] - 7s 109us/sample - loss: 0.3833 - acc: 0.9126 - val_loss: 0.4324 - val_acc: 0.8808\n",
      "Epoch 4/20\n",
      "66247/66247 [==============================] - 8s 117us/sample - loss: 0.2961 - acc: 0.9333 - val_loss: 0.3404 - val_acc: 0.9120\n",
      "Epoch 5/20\n",
      "66247/66247 [==============================] - 8s 119us/sample - loss: 0.2407 - acc: 0.9461 - val_loss: 0.2933 - val_acc: 0.9174\n",
      "Epoch 6/20\n",
      "66247/66247 [==============================] - 8s 119us/sample - loss: 0.2012 - acc: 0.9541 - val_loss: 0.2378 - val_acc: 0.9361\n",
      "Epoch 7/20\n",
      "66247/66247 [==============================] - 8s 127us/sample - loss: 0.1713 - acc: 0.9605 - val_loss: 0.2060 - val_acc: 0.9436\n",
      "Epoch 8/20\n",
      "66247/66247 [==============================] - 10s 152us/sample - loss: 0.1478 - acc: 0.9652 - val_loss: 0.1786 - val_acc: 0.9496\n",
      "Epoch 9/20\n",
      "66247/66247 [==============================] - 9s 141us/sample - loss: 0.1289 - acc: 0.9691 - val_loss: 0.1586 - val_acc: 0.9545\n",
      "Epoch 10/20\n",
      "66247/66247 [==============================] - 10s 153us/sample - loss: 0.1135 - acc: 0.9731 - val_loss: 0.1361 - val_acc: 0.9625\n",
      "Epoch 11/20\n",
      "66247/66247 [==============================] - 8s 118us/sample - loss: 0.1009 - acc: 0.9765 - val_loss: 0.1199 - val_acc: 0.9668\n",
      "Epoch 12/20\n",
      "66247/66247 [==============================] - 8s 121us/sample - loss: 0.0904 - acc: 0.9786 - val_loss: 0.1069 - val_acc: 0.9728\n",
      "Epoch 13/20\n",
      "66247/66247 [==============================] - 8s 119us/sample - loss: 0.0818 - acc: 0.9811 - val_loss: 0.0953 - val_acc: 0.9764\n",
      "Epoch 14/20\n",
      "66247/66247 [==============================] - 8s 117us/sample - loss: 0.0744 - acc: 0.9828 - val_loss: 0.0933 - val_acc: 0.9759\n",
      "Epoch 15/20\n",
      "66247/66247 [==============================] - 8s 121us/sample - loss: 0.0683 - acc: 0.9842 - val_loss: 0.0864 - val_acc: 0.9781\n",
      "Epoch 16/20\n",
      "66247/66247 [==============================] - 7s 111us/sample - loss: 0.0629 - acc: 0.9853 - val_loss: 0.0754 - val_acc: 0.9825\n",
      "Epoch 17/20\n",
      "66247/66247 [==============================] - 8s 115us/sample - loss: 0.0584 - acc: 0.9860 - val_loss: 0.0722 - val_acc: 0.9830\n",
      "Epoch 18/20\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "66247/66247 [==============================] - 7s 110us/sample - loss: 0.0545 - acc: 0.9868 - val_loss: 0.0660 - val_acc: 0.9846\n",
      "Epoch 19/20\n",
      "66247/66247 [==============================] - 8s 123us/sample - loss: 0.0509 - acc: 0.9877 - val_loss: 0.0610 - val_acc: 0.9860\n",
      "Epoch 20/20\n",
      "66247/66247 [==============================] - 7s 112us/sample - loss: 0.0479 - acc: 0.9883 - val_loss: 0.0619 - val_acc: 0.9850\n"
     ]
    }
   ],
   "source": [
    "rates = [0.01,0.001,0.0001]\n",
    "batch_size = 100\n",
    "loss_hist = []\n",
    "accurancy_hist = []\n",
    "# TODO\n",
    "for lr in rates:\n",
    "    K.clear_session()\n",
    "    model = Sequential()\n",
    "    model.add(Dense(256, activation = 'sigmoid', input_shape = (120,)))\n",
    "    model.add(Dense(10, activation = 'softmax'))\n",
    "    opt = Adam(lr = lr, decay=1e-6)\n",
    "    model.compile(\n",
    "        loss='categorical_crossentropy',\n",
    "        optimizer= opt, \n",
    "        metrics = ['acc']\n",
    "    ) \n",
    "    hist = model.fit(Xtr_scale, train_labels, epochs=20, batch_size=100, validation_data=(Xts_scale, test_labels))\n",
    "    loss = hist.history['loss']\n",
    "    loss_hist.append(loss)\n",
    "    acc = hist.history['acc'] \n",
    "    accurancy_hist.append(acc)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Plot the loss funciton vs. the epoch number for all three learning rates on one graph.  You should see that the lower learning rates are more stable, but converge slower."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "label = [0.01,0.001,0.0001]\n",
    "epochs = range(1, 21)\n",
    "for i in range(3):\n",
    "    plt.semilogy(epochs, loss_hist[i])\n",
    "plt.legend(label, loc=0, ncol=1) # ncol指的是legend的列数\n",
    "plt.xlabel('epochs')\n",
    "plt.ylabel('loss')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}
