from fastapi import APIRouter, Cookie, HTTPException, Request
from fastapi.responses import FileResponse
import logging
import os
import pandas as pd
import sys

# Добавляем корень проекта в путь импорта
sys.path.append(os.path.abspath(".."))

from AutoML.testScript import (
    load_data,
    preprocess_data,
    infer_target_column,
    infer_task_type,
    train_and_evaluate,
    validate_data,
    save_model
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["Model"])

# Глобальные переменные для хранения состояния модели
trained_model = None
metrics = None
model_format = "joblib"
TEMP_DATA_PATH = "data.csv"  # Файл загружается заранее


@router.post("/train")
async def train_model(
    request: Request,
    token: str = Cookie(None, alias="users_access_token"),
):
    """Обучает модель на основе загруженного датасета.
    """
    # Проверка токена
    if not token:
        raise HTTPException(status_code=401, detail="Пользователь не авторизован")

    try:
        from schemas.user import UserJWT
        from api.helpers import decode_access_token

        user_data = decode_access_token(token)
        user = UserJWT(**user_data)
        logger.info(f"Пользователь {user.id} запрашивает обучение модели.")
    except Exception as e:
        logger.error(f"Ошибка декодирования токена: {e}")
        raise HTTPException(status_code=401, detail="Неверный токен")

    # Проверяем наличие файла с данными
    if not os.path.exists(TEMP_DATA_PATH):
        raise HTTPException(status_code=400, detail="Файл с данными не найден. Загрузите данные сначала.")

    # Загружаем данные
    df = pd.read_csv(TEMP_DATA_PATH)
    target_column = infer_target_column(df)

    if not target_column:
        raise HTTPException(status_code=400, detail="Не удалось определить целевой столбец")

    task_type = infer_task_type(df, target_column)
    logger.info(f"Определён тип задачи: {task_type}")

    # Валидацию данных
    try:
        validate_data(df, target_column)
    except ValueError as ve:
        logger.warning(f"Ошибка валидации данных: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # Обрабатываем данные
    df_processed = preprocess_data(df, target_column)

    # Обучаем модель
    global trained_model, metrics, model_format
    try:
        trained_model, metrics = train_and_evaluate(df_processed, target_column, task_type)
        save_model(trained_model, format=model_format)
        logger.info("Модель успешно обучена и сохранена.")
        return {"success": True, "message": "Модель обучена", "metric": metrics}
    except Exception as e:
        logger.error(f"Ошибка обучения модели: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обучения модели")


@router.get("/download")
async def download_model():
    """
    Скачивание обученной модели.
    """
    global trained_model
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Модель ещё не обучена")

    model_file = f"model.{model_format}"
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="Файл модели не найден")

    return FileResponse(model_file, filename=model_file, media_type='application/octet-stream')
