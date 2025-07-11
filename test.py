import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import recall_score

# Создаем метрику Recall
m0 = keras.metrics.Recall()

# Обновляем состояние метрики с истинными и предсказанными значениями
m0.update_state(
    [0, 1, 1] + [0, 1, 1] + [0, 1, 1] + [0, 1, 1],  # y_true (фактические значения)
    [1, 1, 1] + [0, 1, 1] + [0, 1, 0] + [1, 0, 0]  # y_pred (предсказанные значения)
   # FP TP TP    TN TP TP    TN TP FN   FP FN FN

    # TP 5
    # FP 2
    # FN 3
    # TN 2
)
# Получаем результат
result = m0.result()
print(f"Recall: {result.numpy()}")

sklearn_recall = recall_score(
    [0, 1, 1] + [0, 1, 1] + [0, 1, 1] + [0, 1, 1],  # y_true (фактические значения)
    [1, 1, 1] + [0, 1, 1] + [0, 1, 0] + [1, 0, 0]  # y_pred (предсказанные значения)
)

print(f"Sklearn Recall: {sklearn_recall:.4f}")


import numpy as np


print(f"Seed")
np.random.seed(42)
print(np.random.permutation(10))
print(np.random.permutation(10))

print(f"Seed")
np.random.seed(42)
print(np.random.permutation(10))
print(np.random.permutation(10))