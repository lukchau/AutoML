import pandas as pd
import sys
import os

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

def run_test():
    data_source = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    print("Загрузка данных")
    df = load_data(data_source)

    if df is None:
        print("Ошибка: данные не загружены.")
        return

    print("Данные успешно загружены. Первые строки:")
    print(df.head())

    print("\nОпределение целевого столбца")
    target_column = infer_target_column(df)
    if target_column is None:
        print("Не удалось определить целевой столбец")
        return
    print(f"Целевой столбец: {target_column}")

    print("\nОпределение типа задачи")
    task_type = infer_task_type(df, target_column)
    print(f"Тип задачи: {task_type}")

    print("\nПредобработка данных")
    try:
        df_processed = preprocess_data(df, target_column)
        print("Данные после предобработки:")
        print(df_processed.head())
    except Exception as e:
        print(f"Ошибка предобработки: {e}")
        return

    print("\nВалидация данных")
    try:
        validate_data(df_processed, target_column)
        print("Валидация данных пройдена")
    except ValueError as ve:
        print(f"Ошибка валидации: {ve}")
        return

    if task_type in ['classification', 'regression']:
        print("\nОбучение модели")
        try:
            model, metric = train_and_evaluate(df_processed, target_column, task_type)
            if task_type == 'classification':
                print(f"Метрика точности (accuracy): {metric:.2f}")
            elif task_type == 'regression':
                print(f"RMSE (корень из MSE): {metric:.2f}")

            # Сохранение модели
            save_model(model, filename="trained_model", format="joblib")
            print("Модель успешно сохранена.")
        except Exception as e:
            print(f"Ошибка обучения: {e}")
    else:
        print("\nКластеризация завершена.")

if __name__ == "__main__":
    run_test()
