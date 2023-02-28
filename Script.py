import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
# matplotlib inline
TRAIN_DIR = "C:/Users/pauls/OneDrive/Escritorio/archivosIA/Identificador/entrenamiento"
TEST_DIR = "C:/Users/pauls/OneDrive/Escritorio/archivosIA/Identificador/prueba"
IMG_SIZE = 100
LR = 1e-3
MODEL_NAME = 'carro-vs-moto-convnet'

# función para etiquetar archivos de imágenes

def label_img(img):
    word_label = img.split('.')[:1]
    if word_label == ['carro']:
        print('carro')
        return [1, 0, 0, 0]
    elif word_label == ['moto']:
        print('moto')
        return [0, 1, 0, 0]
    elif word_label == ['bicicleta']:
        print('bicicleta')
        return [0, 0, 1, 0]
    elif word_label == ['camion']:
        print('camion')
        return [0, 0, 0 ,1]


def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('imagen procesada en train', img, ' label: ', label)
        path = os.path.join(TRAIN_DIR, img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            training_data.append([np.array(img), np.array(label)])
            print('train image loaded ')
        else:
            print("train image not loaded")

        # img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        # training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def create_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        print('imagen procesada en test', img, ' label: ', label)
        path = os.path.join(TEST_DIR, img)

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            testing_data.append([np.array(img), np.array(label)])
            print('test image loaded')
        else:
            print('test image not loaded')

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

# If dataset is not created:
train_data = create_train_data()
test_data = create_test_data()

# If you have already created the dataset:
# train_data = np.load('train_data.npy')
# test_data = np.load('test_data.npy')

"""A continuación, crearemos nuestras matrices de datos.  
Por lo que hago esto para separar mis características y etiquetas:
"""
train = train_data[:-20]
test = train_data[-20:]
X_train = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_train = [i[1] for i in train]
X_test = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y_test = [i[1] for i in test]

#tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)
model.fit({'input': X_train}, {'targets': y_train}, n_epoch=50, 
          validation_set=({'input': X_test}, {'targets': y_test}), 
          snapshot_step=5, show_metric=True, run_id=MODEL_NAME)

fig=plt.figure(figsize=(16, 12))
for num, data in enumerate(test_data[:16]):
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(4, 4, num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: 
        str_label='Carro: Tiene que pagar'
    elif np.argmax(model_out) == 1:
        str_label='Moto: No tiene que pagar'
    elif np.argmax(model_out) == 2:
        str_label='Bicicleta: No tiene que pagar'
    else:
        str_label='Camión: Tiene que pagar'
        
    y.imshow(orig, cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()



