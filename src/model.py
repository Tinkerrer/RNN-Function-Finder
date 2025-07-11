import numpy as np
import tensorflow as tf
import os
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import F1Score, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, \
    FalseNegatives
from tensorflow.keras.layers import Dense, Bidirectional, SimpleRNN, GRU, LSTM
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import Callback

import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

RANDOM_SEED = 42


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

        # Вычисляем метрики
        precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=1)

        print(
            f"\n[Sklearn Metrics for 1 class] Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f} (avg=binary)"
        )

        precision = precision_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        recall = recall_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)
        f1 = f1_score(self.merged_Y_val, y_pred_class, average="binary", zero_division=0, pos_label=0)

        print(
            f"\n[Sklearn Metrics for 0 class] Precision: {precision:.4f}, "
            f"Recall: {recall:.4f}, F1: {f1:.4f} (avg=binary)"
        )

        f1 = f1_score(self.merged_Y_val, y_pred_class, average="weighted", zero_division=0)

        print(
            f"\n[Sklearn Weight Metrics] "
            f"F1: {f1:.4f} (avg=weighted)"
        )


class ShuffleCallback(Callback):
    def __init__(self, x_data, y_data):
        super().__init__()
        self.X = x_data
        self.Y = y_data

    def on_epoch_end(self, epoch, logs=None):
        # Перемешиваем данные
        permutation = np.random.permutation(len(self.X))
        self.X[:] = self.X[permutation]
        self.Y[:] = self.Y[permutation]


# Изменение скорости обучения
def lr_scheduler(epoch, lr):
    return 0.001 / math.sqrt(epoch + 1)


# Основной класс
class BidirectionalRNNClassifier:
    def __init__(self):
        # Размер One-hot вектора
        self.byte_embedding_size = 256
        # Размер батча
        self.batch_size = 32  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        # Эпохи
        self.epochs = 30  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        # Разделение данных на обучающие и валидирующие
        self.validation_split = 0.2  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        # Количество нейронов в первом слое или длина входной последовательности
        self.sequence_length = 24  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        # Функция активации на первом слое
        self.first_layer_activation = "sigmoid"

        self.model = self.__build_model()

    def __build_model(self):
        # Создание архитектуры модели
        # Двунаправленная RNN
        layer = SimpleRNN(self.sequence_length,
                                   activation=self.first_layer_activation,
                                   go_backwards=False,
                                   return_sequences=True,
                                   input_shape=(None, self.byte_embedding_size))

        # forward_lstm_2 = SimpleRNN(30,
        #                            activation='sigmoid',
        #                            go_backwards=False,
        #                            )
        #
        # backward_lstm_2 = SimpleRNN(30,
        #                             activation='sigmoid',
        #                             go_backwards=True,
        #                             )

        model = Sequential([
            Bidirectional(
                layer
            ),
            Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='rmsprop',  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
            loss='binary_crossentropy',
            #     Bullshit!
            #     Very very strange and SUS values. Maybe it works with 2D vectors as a multilabel indicator. IDK
            # metrics=[

            #     'binary_accuracy',
            #     Precision(),
            #     Recall(),
            #     F1Score(threshold=0.5, average="weighted"),
            #     TruePositives(thresholds=0.5),
            #     TrueNegatives(thresholds=0.5),
            #     FalsePositives(thresholds=0.5),
            #     FalseNegatives(thresholds=0.5),
            # ]
        )
        return model

    def __preprocess_data(self, sequences, property_sequences=None):
        # Преобразование последовательностей байт в one-hot encoded данные

        processed_seq = np.zeros((len(sequences), self.sequence_length, self.byte_embedding_size),
                                 dtype=np.float32)

        # Преобразуем каждый байт в one-hot вектор
        for i, seq in enumerate(sequences):

            for t, byte in enumerate(seq):
                processed_seq[i, t, byte] = 1.0

        if property_sequences is None:
            return processed_seq, None

        # Блаблабла
        padded_prop_seq = pad_sequences(
            property_sequences,
            padding='post',  # 'post' — нули в конец, 'pre' — в начало
            dtype='float32'
        )

        processed_property_sequences = np.array(padded_prop_seq)

        return processed_seq, processed_property_sequences

    # def split_byte_sequences(self, sequences, property_sequences):
    #     # Создание последовательностей байт из функций
    #     split_sequences = []
    #     split_property_sequences = []

    #     for seq in sequences:
    #         split_sequences.extend([seq[i:i + self.sequence_length] for i in range(0, len(seq), self.sequence_length)])

    #     for prop_seq in property_sequences:
    #         split_property_sequences.extend(
    #             [prop_seq[i:i + self.sequence_length] for i in range(0, len(prop_seq), self.sequence_length)])

    #     return split_sequences, split_property_sequences

    # Как будто бы все равно недостаточно из-за параллельности вычислений tensorflow((
    def set_seed(self, seed=RANDOM_SEED):
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)

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

                if prop_seq_weight == 0:
                    continue

                func_start_count += prop_seq_weight

                if prop_seq_weight > 1:
                    func_start_repeat_at_seq += prop_seq_weight

                split_sequences.append(sequences[i][j:j + self.sequence_length])
                split_property_sequences.append(property_sequences[i][j:j + self.sequence_length])

        print(func_start_count)
        print(func_start_repeat_at_seq)

        print(len(split_sequences))
        print(len(split_property_sequences))

        return split_sequences, split_property_sequences

    def train(self, x_data, y_data):
        # Обучение модели

        # Разбиение данных
        x_train, x_val, y_train, y_val = train_test_split(
            x_data, y_data,
            test_size=self.validation_split,
            shuffle=True,
            random_state=RANDOM_SEED
        )

        # Преобразование данных
        x_train_preprocessed, y_train_preprocessed = self.__preprocess_data(x_train, y_train)
        x_val_preprocessed, y_val_preprocessed = self.__preprocess_data(x_val, y_val)

        # Обязательно устанавливаем seed для воспроизводимости!!!!
        self.set_seed()

        # Добавляем EarlyStopping по умолчанию, если не переданы другие коллбэки
        callbacks = [
            # EarlyStopping(monitor='val_loss', patience=5),  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
            SklearnMetricsCallback(  # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
                validation_data=(x_val_preprocessed, y_val_preprocessed)
            ),
            ShuffleCallback(x_train_preprocessed, y_train_preprocessed)
            # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
            # LearningRateScheduler(lr_scheduler)
        ]
        # callbacks = []

        history = self.model.fit(
            x_train_preprocessed, y_train_preprocessed,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(x_val_preprocessed, y_val_preprocessed),
            callbacks=callbacks,
            class_weight={0: 1, 1: 250}  # # TODO МОЖЕТ ВЛИЯТЬ НА РЕЗУЛЬТАТ ОБУЧЕНИЯ, ТРЕБУЕТ ОТСМОТРА
        )
        return history

    def predict(self, byte_sequence):
        # Предсказание свойства начала функции для каждого байта в последовательности

        # Преобразование в one-hot
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
