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


def euclidean_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


class NeuralModel:
    def __init__(self, h5_file=None, model_json=None):
        if not h5_file or not model_json:
            input_dim = (200, 200, 3)
            input_a = Input(shape=input_dim)
            input_b = Input(shape=input_dim)

            base_network = build_base_network(input_dim)
            feat_vecs_a = base_network(input_a)
            feat_vecs_b = base_network(input_b)
            distance = Lambda(self.calculate_euclidean_distance,
                              output_shape=euclidean_dist_output_shape)([feat_vecs_a, feat_vecs_b])
            self.model = Model(inputs=[input_a, input_b], outputs=distance)
        else:
            self.model = keras.models.model_from_json(model_json)
            self.model.load_weights(h5_file)
        self.embeddings = None

    def calculate_euclidean_distance(self, vectors):
        x, y = vectors
        self.embeddings = x
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        euclidean_distance = K.sqrt(K.maximum(sum_square, K.epsilon()))
        return euclidean_distance

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

    def get_image_embeddings(self, img: np.ndarray):
        img2_stub = np.zeros((200, 200, 3))
        self.model.predict(img, img2_stub)
        return self.embeddings


def make_similarity_with_model(neural_model: NeuralModel, frames_vec1: np.ndarray, frames_vec2: np.ndarray):
    """
    Calculate similarity predictions between frames of two videos using a neural model.

    :param neural_model: NeuralModel: The neural network model used for prediction.
    :param frames_vec1: np.ndarray: First ndarray of frames with shape (n, 400, 400, 3).
    :param frames_vec2: np.ndarray: Second ndarray of frames with shape (m, 400, 400, 3).
    :return: np.ndarray: Matrix of similarity predictions with shape (n, m).
    """
    n = len(frames_vec1)
    m = len(frames_vec2)

    similarity_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        batch_frames_vec1 = np.repeat(frames_vec1[i:i + 1], m, axis=0)
        batch_frames_vec2 = frames_vec2

        img_1_f = np.array(batch_frames_vec1.tolist(), dtype=np.float32)
        img_2_f = np.array(batch_frames_vec2.tolist(), dtype=np.float32)

        predictions = neural_model.predict(img_1_f, img_2_f)

        similarity_matrix[i, :] = predictions

    return similarity_matrix


"""
    To reduce time for making similarity table for every element of first video with every element of second video 
    there is an algorithm making an equal sets like first of the first video with first of the second video, second of 
    first video with first of second video etc. Sets are given to neural model 
    and processing in GPU memory many-at-once. 
    1   2       1   2      .......
    [1] [1]     [1] [2]    .......
    [2] [1]     [2] [2]    .......
    [3] [1]     [3] [2]    .......
    .   .       .   .      .......
    .   .       .   .
"""
