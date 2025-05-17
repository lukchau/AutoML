import pandas as pd
import requests
import tempfile
import os
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import pickle

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def validate_data(df: pd.DataFrame, target_column: str):
    """
    Выполняет базовые проверки качества данных.
    :param df: DataFrame с данными.
    :param target_column: Целевая переменная.
    :raises ValueError: Если данные не проходят валидацию.
    """
    if df.empty:
        raise ValueError("Датасет пуст.")

    if target_column not in df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' отсутствует в данных.")

    if df.shape[1] < 2:
        raise ValueError("Недостаточно признаков для обучения модели.")

    unique_target_values = df[target_column].nunique()

    if unique_target_values < 2:
        raise ValueError(f"Целевая переменная содержит только одно уникальное значение: {df[target_column].iloc[0]}")

    logging.info(f"Валидация успешна. Уникальных значений в таргете: {unique_target_values}")


def load_data(file_path):
    """
    Загружает данные из файла или URL.
    :param file_path: Путь к файлу или URL.
    :return: DataFrame с загруженными данными или None в случае ошибки.
    """
    try:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            # Если путь является URL
            response = requests.get(file_path)
            response.raise_for_status()

            # Создаем временный файл для хранения данных
            ext = ".csv" if ".csv" in file_path else ".xlsx"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(response.content)
            tmp.close()

            file_path = tmp.name

        # Загрузка данных из файла
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Неверный формат файла")

        logging.info("Данные были загружены успешно.")
        return df

    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        return None


def preprocess_data(df, target_column):
    """
    Предобрабатывает данные: удаляет NaN, кодирует категориальные признаки и нормализует числовые.
    :param df: DataFrame с исходными данными.
    :param target_column: Название целевого столбца.
    :return: DataFrame с предобработанными данными.
    """
    df = df.dropna(axis=0).copy()  # Удаляем строки с NaN и избегаем SettingWithCopyWarning

    # Отдельно сохраняем целевой столбец
    target = df[target_column]
    features = df.drop(columns=[target_column])

    # Кодируем категориальные признаки (только признаки, не target)
    for col in features.select_dtypes(include=['object']).columns:
        features.loc[:, col] = LabelEncoder().fit_transform(features[col])

    # Нормализуем числовые признаки
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    features[numeric_cols] = StandardScaler().fit_transform(features[numeric_cols])

    # Объединяем обратно признаки и целевую переменную
    df_processed = pd.concat([features, target.reset_index(drop=True)], axis=1)
    logging.info("Данные успешно обработаны.")
    return df_processed


def infer_target_column(df, task_type=None):
    """
    Определяет целевой столбец на основе типа задачи или предположений.
    :param df: DataFrame с данными.
    :param task_type: Тип задачи ('classification' или 'regression').
    :return: Название целевого столбца или None, если не удалось определить.
    """
    if task_type == "classification":
        # Для классификации выбираем категориальный столбец или числовой с небольшим числом уникальных значений
        for col in df.select_dtypes(include=['object']).columns:
            return col
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if df[col].nunique() <= 20:
                return col
    elif task_type == "regression":
        # Для регрессии выбираем числовой столбец с большим числом уникальных значений
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            return col

    # Если task_type не задан, пробуем выбрать столбец, исходя из предположений
    for col in df.columns:
        if df[col].dtype == 'object':
            return col  # Возвращаем первый категориальный столбец для классификации
        elif df[col].dtype in ['int64', 'float64'] and df[col].nunique() > 20:
            return col  # Возвращаем первый числовой столбец для регрессии

    logging.warning("Не удалось вывести целевой столбец")
    return None  # Если не удалось выбрать целевой столбец


def infer_task_type(df, target_column):
    """
    Определяет тип задачи на основе целевого столбца.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    :return: Тип задачи ('classification', 'regression' или 'clustering').
    """
    if target_column is None or target_column not in df.columns:
        return "clustering"

    target = df[target_column]

    if target.dtype == 'object':
        return "classification"
    elif pd.api.types.is_integer_dtype(target) and target.nunique() <= 20:
        return "classification"
    elif pd.api.types.is_numeric_dtype(target):
        return "regression"
    else:
        raise ValueError("Не удалось определить тип задачи")


def select_model(task_type, n_estimators=100, n_clusters=3):
    """
    Выбирает модель в зависимости от типа задачи.
    :param task_type: Тип задачи ('classification', 'regression' или 'clustering').
    :param n_estimators: Количество деревьев для RandomForest.
    :param n_clusters: Количество кластеров для KMeans.
    :return: Модель для задачи.
    """
    if task_type == "classification":
        return RandomForestClassifier(n_estimators=n_estimators)
    elif task_type == "regression":
        return RandomForestRegressor(n_estimators=n_estimators)
    elif task_type == "clustering":
        return KMeans(n_clusters=n_clusters)
    else:
        raise ValueError("Неподдерживаемая задача")


def train_and_evaluate(df, target_column, task_type, n_estimators=100, n_clusters=3):
    """
    Обучает модель и оценивает её качество.
    :param df: DataFrame с данными.
    :param target_column: Название целевого столбца.
    :param task_type: Тип задачи ('classification' или 'regression').
    :param n_estimators: Количество деревьев для RandomForest.
    :param n_clusters: Количество кластеров для KMeans.
    :return: Обученная модель и метрика качества.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if len(X) < 2:
        raise ValueError("Недостаточно данных для разделения на train/test.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = select_model(task_type, n_estimators, n_clusters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metric = accuracy_score(y_test, y_pred)
    elif task_type == "regression":
        metric = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    else:
        metric = None

    logging.info(f"Модель обучена и оценена. Метрики: {metric}")
    return model, metric


def save_model(model, filename="model", format="joblib"):
    """
    Сохраняет обученную модель в файл в выбранном формате.
    :param model: Обученная модель.
    :param filename: Имя файла для сохранения модели.
    :param format: Формат сохранения модели ('pkl', 'joblib').
    """
    if format == "pkl":
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(model, f)
    elif format == "joblib":
        joblib.dump(model, f"{filename}.joblib")
    else:
        raise ValueError("Неподдерживаемый формат сохранения модели")

    logging.info(f"Модель сохранена как {filename}.{format}")
