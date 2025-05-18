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
logger = logging.getLogger(__name__)


def is_id_like(col_name):
    """Проверяет, является ли столбец ID-подобным"""
    id_keywords = ['id', 'name', 'ticket', 'cabin', 'key', 'identifier']
    return any(keyword in str(col_name).lower() for keyword in id_keywords)


def infer_target_column(df: pd.DataFrame, task_type=None):
    """
    Автоматически определяет целевой столбец на основе типа задачи или предположений.
    """

    # Список приоритетных колонок по типу задачи
    target_candidates = {
        'classification': ['survived', 'target', 'y', 'class', 'label'],
        'regression': ['fare', 'age', 'price', 'income', 'sales']
    }

    if task_type and task_type in target_candidates:
        for candidate in target_candidates[task_type]:
            if candidate in df.columns.str.lower():
                idx = df.columns.tolist().index(df.columns[df.columns.str.lower() == candidate][0])
                logger.info(f"Найден приоритетный таргет: {df.columns[idx]}")
                return df.columns[idx]

    # Проверка на наличие бинарных столбцов
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if not is_id_like(col) and df[col].nunique() == 2:
            logger.info(f"Выбран бинарный целевой столбец: {col}")
            return col

    # Для классификации: категориальные или числовые с малым числом уникальных значений
    if task_type == "classification":
        for col in df.select_dtypes(include=['object']).columns:
            if not is_id_like(col) and df[col].nunique() < len(df) * 0.5:
                logger.info(f"Выбран целевой столбец: {col}")
                return col
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if not is_id_like(col) and df[col].nunique() <= 20:
                logger.info(f"Выбран числовой целевой столбец: {col}")
                return col

    # Для регрессии: числовой с высокой вариативностью
    elif task_type == "regression":
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            if not is_id_like(col) and df[col].nunique() > 20:
                logger.info(f"Выбран числовой целевой столбец (регрессия): {col}")
                return col

    # Если task_type не задан
    else:
        for col in df.columns:
            if not is_id_like(col):
                if df[col].dtype == 'object' and df[col].nunique() < len(df) * 0.5:
                    logger.info(f"Автоматически выбран целевой столбец: {col}")
                    return col
                elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 20:
                    logger.info(f"Автоматически выбран числовой целевой столбец: {col}")
                    return col

    logger.warning("Не удалось вывести целевой столбец")
    return None


def load_data(file_path):
    """
    Загружает данные из файла или URL.
    :param file_path: Путь к файлу или URL.
    :return: DataFrame с данными или None в случае ошибки.
    """
    try:
        if file_path.startswith("http://") or file_path.startswith("https://"):
            response = requests.get(file_path)
            response.raise_for_status()

            ext = ".csv" if ".csv" in file_path else ".xlsx"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
            tmp.write(response.content)
            tmp.close()
            file_path = tmp.name

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Неверный формат файла")

        logger.info("Данные были загружены успешно.")
        return df

    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        return None


def preprocess_data(df: pd.DataFrame, target_column: str):
    """
    Предобрабатывает данные: удаляет NaN, кодирует категориальные признаки, нормализует числовые.
    """
    if df[target_column].isna().any():
        df = df.dropna(subset=[target_column]).copy()
    features = df.drop(columns=[target_column])

    # Кодируем категориальные признаки
    for col in features.select_dtypes(include=['object']).columns:
        features.loc[:, col] = LabelEncoder().fit_transform(features[col])

    # Нормализуем числовые признаки
    numeric_cols = features.select_dtypes(include=['int64', 'float64']).columns
    features[numeric_cols] = StandardScaler().fit_transform(features[numeric_cols])

    # Объединяем обратно
    df_processed = pd.concat([features, df[[target_column]].reset_index(drop=True)], axis=1)
    logger.info("Данные успешно обработаны.")
    return df_processed


def infer_task_type(df: pd.DataFrame, target_column: str):
    """
    Определяет тип задачи автоматически.
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


def select_model(task_type: str, n_estimators: int = 100, n_clusters: int = 3):
    """
    Выбирает модель в зависимости от типа задачи.
    """
    if task_type == "classification":
        return RandomForestClassifier(n_estimators=n_estimators)
    elif task_type == "regression":
        return RandomForestRegressor(n_estimators=n_estimators)
    elif task_type == "clustering":
        return KMeans(n_clusters=n_clusters)
    else:
        raise ValueError("Неподдерживаемая задача")


def validate_data(df: pd.DataFrame, target_column: str):
    """
    Проверяет данные перед обучением.
    """
    if df.empty:
        raise ValueError("Датасет пуст.")

    if target_column not in df.columns:
        raise ValueError(f"Целевой столбец '{target_column}' отсутствует в данных.")

    if df.shape[1] < 2:
        raise ValueError("Недостаточно признаков для обучения модели.")

    unique_values = df[target_column].nunique()
    if unique_values < 2:
        raise ValueError(f"Целевая переменная содержит только одно уникальное значение: {df[target_column].iloc[0]}")

    logger.info(f"Валидация успешна. Уникальных значений в таргете: {unique_values}")


def train_and_evaluate(df: pd.DataFrame, target_column: str, task_type: str, n_estimators: int = 100, n_clusters: int = 3):
    """
    Обучает модель и оценивает её качество.
    """
    df = df.dropna(subset=[target_column]).copy()
    X = df.drop(columns=[target_column])
    y = df[target_column]

    if len(X) < 2:
        raise ValueError("Недостаточно данных для разделения на train/test.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if y_train.isna().any():
        raise ValueError("Целевая переменная содержит NaN значения.")

    model = select_model(task_type, n_estimators, n_clusters)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if task_type == "classification":
        metric = accuracy_score(y_test, y_pred)
    elif task_type == "regression":
        metric = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    else:
        metric = None

    logger.info(f"Модель обучена и оценена. Метрики: {metric}")
    return model, metric


def save_model(model, filename="model", format="joblib"):
    """
    Сохраняет обученную модель в файл в выбранном формате.
    """
    if format == "pkl":
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(model, f)
    elif format == "joblib":
        joblib.dump(model, f"{filename}.joblib")
    else:
        raise ValueError("Неподдерживаемый формат сохранения модели")

    logger.info(f"Модель сохранена как {filename}.{format}")
