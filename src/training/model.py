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
        # Получаем предсказания модели
        y_pred = self.model.predict(self.X_val, verbose=0)

        y_pred_class = (np.array([item for sublist in y_pred for item in sublist]) > 0.5).astype(int)

        # Вычисляем метрики для класса 1
        precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)

        print(
            f"\n[Sklearn Metrics for 1 class] Precision: {precision:.6f}, "
            f"Recall: {recall:.6f}, F1: {f1:.6f} (avg=binary)"
        )

        # Обновляем логи для класса 1
        logs['first_class_val_precision'] = precision
        logs['first_class_val_recall'] = recall
        logs['first_class_val_f1'] = f1

        # Вычисляем метрики для класса 0
        precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)

        print(
            f"\n[Sklearn Metrics for 0 class] Precision: {precision:.6f}, "
            f"Recall: {recall:.6f}, F1: {f1:.6f} (avg=binary)"
        )

        # Обновляем логи для класса 0
        logs['zero_class_val_precision'] = precision
        logs['zero_class_val_recall'] = recall
        logs['zero_class_val_f1'] = f1

        # Вычисляем взвешенный f1
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="weighted", zero_division=0)

        print(
            f"\n[Sklearn Weight Metrics] "
            f"F1: {f1:.6f} (avg=weighted)"
        )

        logs['weighted_val_f1'] = f1


# Основной класс
class BidirectionalRNNClassifier:
    def __init__(self, epochs=None, sequence_length=None, neurons_num=None,
                 first_layer_activation=None, first_layer_arch=None, weight_multiplier=None):

        # Размер One-hot вектора
        self.byte_embedding_size = 256

        # Размер батча
        self.batch_size = 32  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА

        # Эпохи
        if epochs is None:
            self.epochs = 30  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        else:
            self.epochs = epochs

        # Разделение данных на обучающие и валидирующие
        self.validation_split = 0.2  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА

        # Длина входной последовательности
        if sequence_length is None:
            self.sequence_length = 1000  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        else:
            self.sequence_length = sequence_length

        # Количество нейронов в первом слое
        if neurons_num is None:
            self.neurons_num = 16  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        else:
            self.neurons_num = neurons_num

        # Функция активации на первом слое
        if first_layer_activation is None:
            self.first_layer_activation = "relu"  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        else:
            self.first_layer_activation = first_layer_activation

        # Функция активации на первом слое
        if (first_layer_arch is None) or (first_layer_arch == "rnn"):
            self.first_layer_arch = SimpleRNN  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        elif first_layer_arch == "gru":
            self.first_layer_arch = GRU  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        elif first_layer_arch == "lstm":
            self.first_layer_arch = LSTM  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА

        # Множитель веса функции потерь
        if weight_multiplier is None:
            self.weight_multiplier = 400  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        else:
            self.weight_multiplier = weight_multiplier

        # Рандомный сид
        self.random_seed = 42

        self.model = self.__build_model()

    def __build_model(self):
        # Создание архитектуры модели
        # Двунаправленная RNN

        model = Sequential([
            # Embedding(
            #     input_dim=self.byte_embedding_size + 1,
            #     output_dim=self.neurons_num,
            #     input_length=self.sequence_length
            # ),
            Bidirectional(
                self.first_layer_arch(self.neurons_num,
                                      activation=self.first_layer_activation,
                                      return_sequences=True,
                                      input_shape=(None, self.byte_embedding_size)
                                      )
            ),
            TimeDistributed(Dense(1, activation='sigmoid'))
        ])

        model.compile(
            optimizer='rmsprop',  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
            loss='binary_crossentropy',
        )
        return model

    def __preprocess_data(self, sequences, property_sequences=None):
        processed_seq = one_hot(
            pad_sequences(
                sequences,
                maxlen=self.sequence_length,
                padding='post',  # 'post' — нули в конец, 'pre' — в начало
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
            padding='post',  # 'post' — нули в конец, 'pre' — в начало
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
        # Создание последовательностей байт из функций
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
        # Обучение модели
        # Обязательно устанавливаем seed для воспроизводимости!!!!
        self.set_seed()

        # Разбиение данных
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data,
            test_size=self.validation_split,
            shuffle=True,
            random_state=self.random_seed
        )

        # Преобразование данных
        x_train_preprocessed, y_train_preprocessed = self.__preprocess_data(x_train, y_train)
        x_val_preprocessed, y_val_preprocessed = self.__preprocess_data(x_val, y_val)

        # Добавляем EarlyStopping по умолчанию, если не переданы другие коллбэки
        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=5),  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
            SklearnMetricsCallback(  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
                validation_data=(x_val_preprocessed, y_val_preprocessed)
            ),
        ]

        history = self.model.fit(
            x_train_preprocessed, y_train_preprocessed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_val_preprocessed, y_val_preprocessed),
            callbacks=callbacks,
            class_weight={0: 1, 1: self.weight_multiplier}  # # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        )
        return history

    def predict(self, byte_sequence):
        # Предсказание свойства начала функции для каждого байта в последовательности

        # Предобработка данных (дополнение нулями)
        encoded = self.__preprocess_data([byte_sequence], None)

        # Предсказание
        predictions = self.model.predict(encoded)
        return predictions  # Возвращаем прогнозы для первой последовательности

    def summary(self):
        # Вывод информации о модели
        return self.model.summary()

    def save_weights(self, filepath, overwrite=True):
        # Сохранение весов
        self.model.save_weights(filepath, overwrite=overwrite)

    def load_weights(self, filepath):
        # Загрузка весов модели
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл весов не найден: {filepath}")
        self.model.load_weights(filepath)

    def save_model(self, filepath, overwrite=True):
        # Сохранение модели
        self.model.save(filepath, overwrite=overwrite)

    def load_model(self, filepath):
        # Загрузка модели
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл модели не найден: {filepath}")
        self.model.load_model(filepath)
