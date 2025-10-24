import numpy as np
import tensorflow as tf
import os
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import F1Score, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, \
    FalseNegatives
from tensorflow.keras.layers import Dense, Bidirectional, SimpleRNN, GRU, LSTM, Embedding, TimeDistributed
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow import one_hot

import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


class SklearnMetricsCallback(Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.X_val, self.Y_val = validation_data
        self.threshold = 0.5
        self.merged_Y_val = [item for sublist in self.Y_val for item in sublist]

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)

        y_pred_class = (np.array([item for sublist in y_pred for item in sublist]) > self.threshold).astype(int)

        # Calculate f1 for 1 class
        precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)

        logs['first_class_val_precision'] = precision
        logs['first_class_val_recall'] = recall
        logs['first_class_val_f1'] = f1

        # precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        # recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        # f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        #
        # logs['zero_class_val_precision'] = precision
        # logs['zero_class_val_recall'] = recall
        # logs['zero_class_val_f1'] = f1

        # Calculate weighted f1
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="weighted", zero_division=0)

        logs['weighted_val_f1'] = f1


class BidirectionalRNNClassifier:
    def __init__(self, random_padding_on_train=False, random_padding_on_val=False, epochs=None, sequence_length=None, neurons_num=None,
                 first_layer_activation=None, first_layer_arch=None, weight_multiplier=None):

        # Random padding for training set
        self.random_padding_on_train = random_padding_on_train

        # Random padding for testing set
        self.random_padding_on_val = random_padding_on_val

        # One-hot size
        self.byte_embedding_size = 256

        # TODO Batch size can affect the evaluation metrics
        # Batch size
        self.batch_size = 32

        # TODO Epochs number can affect the evaluation metrics
        # Epochs number
        if epochs is None:
            self.epochs = 30
        else:
            self.epochs = epochs

        # TODO Validation split percent can affect the evaluation metrics
        # Validation split percent
        self.validation_split = 0.2

        # TODO Input sequence length can affect the evaluation metrics
        # Input sequence length
        if sequence_length is None:
            self.sequence_length = 1000
        else:
            self.sequence_length = sequence_length

        # TODO Number of neurons in the recurrent layer can affect the evaluation metrics
        # Number of neurons in the recurrent layer
        if neurons_num is None:
            self.neurons_num = 16
        else:
            self.neurons_num = neurons_num

        # TODO Activation function in the recurrent layer can affect the evaluation metrics
        # Activation function in the recurrent layer
        if first_layer_activation is None:
            self.first_layer_activation = "relu"
        else:
            self.first_layer_activation = first_layer_activation

        # TODO Architecture of the recurrent layer can affect the evaluation metrics
        # Architecture of the recurrent layer
        if (first_layer_arch is None) or (first_layer_arch == "rnn"):
            self.first_layer_arch = SimpleRNN
        elif first_layer_arch == "gru":
            self.first_layer_arch = GRU
        elif first_layer_arch == "lstm":
            self.first_layer_arch = LSTM

        # TODO Loss function weight multiplier can affect the evaluation metrics
        # Loss function weight multiplier
        if weight_multiplier is None:
            self.weight_multiplier = 0.9
        else:
            self.weight_multiplier = weight_multiplier

        # Random seed
        self.random_seed = 42

        self.model = self.__build_model()

    def __build_model(self):
        # 1 bidirectional layer and 1 activation layer
        model = Sequential([
            Bidirectional(
                self.first_layer_arch(self.neurons_num,
                                      activation=self.first_layer_activation,
                                      return_sequences=True,
                                      input_shape=(None, self.byte_embedding_size)
                                      )
            ),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])

        # TODO Loss function optimizer can affect the evaluation metrics
        model.compile(
            optimizer='rmsprop',
            loss=BinaryFocalCrossentropy(alpha=self.weight_multiplier, gamma=2.0),
        )
        return model

    def __preprocess_mask_data(self, sequences, property_sequences, align_value, align_max_len):

        for sequence_num in range(len(property_sequences)):
            for byte_num in range(align_max_len + 1, len(property_sequences[sequence_num])):
                if property_sequences[sequence_num][byte_num] == 1:
                    for align_ctr in range(1, align_max_len + 1):
                        if sequences[sequence_num][byte_num - align_ctr] == align_value:
                            sequences[sequence_num][byte_num - align_ctr] = random.randint(0, 255)

        processed_seq = one_hot(
            pad_sequences(
                sequences,
                maxlen=self.sequence_length,
                padding='post',
                dtype='int32',
                value=0
            ),
            self.byte_embedding_size
        )

        processed_property_sequences = pad_sequences(
            property_sequences,
            maxlen=self.sequence_length,
            padding='post',
            dtype='int32',
            value=0
        )

        return processed_seq, processed_property_sequences

    def __preprocess_data(self, sequences, property_sequences=None):
        processed_seq = one_hot(
            pad_sequences(
                sequences,
                maxlen=self.sequence_length,
                padding='post',
                dtype='int32',
                value=0
            ),
            self.byte_embedding_size
        )

        if property_sequences is None:
            return processed_seq, None

        processed_property_sequences = pad_sequences(
            property_sequences,
            maxlen=self.sequence_length,
            padding='post',
            dtype='int32',
            value=0
        )

        return processed_seq, processed_property_sequences

    def set_seed(self):
        os.environ['PYTHONHASHSEED'] = str(self.random_seed)
        random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        np.random.seed(self.random_seed)

    def split_byte_sequences(self, sequences, property_sequences):
        # Splitting sequence to subsequences
        split_sequences = []
        split_property_sequences = []

        if len(sequences) != len(property_sequences):
            raise Exception("len(sequences) != len(property_sequences)")

        for i in range(len(sequences)):
            if len(sequences[i]) != len(property_sequences[i]):
                raise Exception(f"len(sequences[{i}]) != len(property_sequences[{i}])")

        func_start_count = 0
        func_start_repeat_at_seq = 0

        for i in range(len(sequences)):
            for j in range(0, len(sequences[i]), self.sequence_length):

                prop_seq_weight = sum(property_sequences[i][j:j + self.sequence_length])

                func_start_count += prop_seq_weight

                if prop_seq_weight > 1:
                    func_start_repeat_at_seq += prop_seq_weight

                split_sequences.append(list(sequences[i][j:j + self.sequence_length]))
                split_property_sequences.append(list(property_sequences[i][j:j + self.sequence_length]))

            tail_length = len(sequences[i]) % self.sequence_length

            if tail_length != 0:
                split_sequences.append(
                    list(sequences[i][len(sequences[i]) - tail_length:len(sequences[i])]
                         )
                )
                split_property_sequences.append(
                    list(property_sequences[i][len(property_sequences[i]) - tail_length:len(property_sequences[i])]
                         )
                )

        print(f"Total func start number after split: {func_start_count}")
        print(f"Total func start number in same sequences: {func_start_repeat_at_seq}")

        print(f"Total sequences: {len(split_sequences)}")
        print(f"Total property sequences: {len(split_property_sequences)}")

        return split_sequences, split_property_sequences

    def train(self, x_data, y_data):
        # Setting the seed for reproducibility !!!!
        self.set_seed()

        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data,
            test_size=self.validation_split,
            shuffle=True,
            random_state=self.random_seed
        )

        # ALIGN VALUE and ALIGN MAX LEN depend on microcontroller architecture !!!!
        if self.random_padding_on_train:
            x_train_preprocessed, y_train_preprocessed = self.__preprocess_mask_data(x_train, y_train, 0x0, 0x3)
        else:
            x_train_preprocessed, y_train_preprocessed = self.__preprocess_data(x_train, y_train)

        if self.random_padding_on_val:
            x_val_preprocessed, y_val_preprocessed = self.__preprocess_mask_data(x_val, y_val, 0x0, 0x3)
        else:
            x_val_preprocessed, y_val_preprocessed = self.__preprocess_data(x_val, y_val)


        callbacks = [
            # TODO Early Stopping callback can affect the evaluation metrics
            # EarlyStopping(monitor='val_loss', patience=5),
            SklearnMetricsCallback(
                validation_data=(x_val_preprocessed, y_val_preprocessed)
            ),
        ]

        history = self.model.fit(
            x_train_preprocessed, y_train_preprocessed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_val_preprocessed, y_val_preprocessed),
            callbacks=callbacks,
            # TODO Class weights can affect the evaluation metrics
            class_weight={0: 1, 1: 100}
        )
        return history

    def predict(self, byte_sequence):
        encoded = self.__preprocess_data([byte_sequence], None)

        predictions = self.model.predict(encoded)
        return predictions

    def summary(self):
        return self.model.summary()

    def save_weights(self, filepath, overwrite=True):
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The weight file was not found: {filepath}")
        self.model.load_weights(filepath)

    def save_model(self, filepath, overwrite=True):
        self.model.save(filepath, overwrite=overwrite)

    def load_model(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"The model file was not found: {filepath}")
        self.model.load_model(filepath)
