import keras
import numpy as np
import tensorflow.python.keras as K
from keras.api.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, Input, Lambda
from keras.api.models import Sequential, Model


def build_base_network(input_shape):
    seq = Sequential()
    nb_filter = [6, 12]
    kernel_size = 3

    # Convolutional layer 1
    seq.add(Conv2D(nb_filter[0], (kernel_size, kernel_size), input_shape=input_shape,
                   padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(0.25))

    # Convolutional layer 2
    seq.add(Conv2D(nb_filter[1], (kernel_size, kernel_size), padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(0.25))

    # Flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


def calculate_euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    euclidean_distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
    # print(euclidean_distance)
    return euclidean_distance


def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


class NeuralModel:
    def __init__(self, h5_file=None, model_json=None):
        if not h5_file or not model_json:
            input_dim = (400, 400, 3)
            input_a = Input(shape=input_dim)
            input_b = Input(shape=input_dim)

            base_network = build_base_network(input_dim)
            feat_vecs_a = base_network(input_a)
            feat_vecs_b = base_network(input_b)
            distance = Lambda(calculate_euclidean_distance,
                              output_shape=euclidean_dist_output_shape)([feat_vecs_a, feat_vecs_b])
            self.model = Model(inputs=[input_a, input_b], outputs=distance)
        else:
            self.model = keras.models.model_from_json(model_json)
            self.model.load_weights(h5_file)

    def predict(self, img1: np.ndarray, img2: np.ndarray):
        """
            create predictions ndarray from two ndarrays with shape (n, 400,400, 3)
            where n is a number of samples (frames)
            :param img1: np.ndarray: first ndarray of frames
            :param img2: np.ndarray: second ndarray of frames
            :return: database connection
        """
        img_1_f = np.array(img1.tolist(), dtype=np.float32)
        img_2_f = np.array(img2.tolist(), dtype=np.float32)
        predictions = self.model.predict([img_1_f, img_2_f])
        return predictions


"""
    1   2       1   2      .......
    [1] [1]     [1] [2]    .......
    [2] [1]     [2] [2]    .......
    [3] [1]     [3] [2]    .......
    .   .       .   .      .......
    .   .       .   .
"""
