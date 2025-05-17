import pandas as pd
from AutoML.testScript import (
    load_data,
    preprocess_data,
    infer_target_column,
    infer_task_type,
    train_and_evaluate,
    validate_data  
)

def run_test():
    data_source = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv "
    print("Загрузка данных")
    df = load_data(data_source)

    if df is None:
        print("Ошибка: данные не загружены.")
        return

    print("Данные успешно загружены. Первые строки:")
    print(df.head())

    print("\nОпределение целевого столбца")
    target_column = infer_target_column(df)
    print(f"Целевой столбец: {target_column}")

    print("\nОпределение типа задачи")
    task_type = infer_task_type(df, target_column)
    print(f"Тип задачи: {task_type}")

    print("\nПредобработка данных")
    df_processed = preprocess_data(df, target_column)
    print("Данные после предобработки:")
    print(df_processed.head())

    try:
        validate_data(df_processed, target_column)
        print("Валидация данных пройдена")
    except ValueError as e:
        print(f"Ошибка валидации: {e}")
        return

    if task_type in ['classification', 'regression']:
        print("\nОбучение модели")
        model, metric = train_and_evaluate(df_processed, target_column, task_type)

        if task_type == 'classification':
            print(f"Метрика точности (accuracy): {metric:.2f}")
        elif task_type == 'regression':
            print(f"RMSE (корень из MSE): {metric:.2f}")
    else:
        print("\nКластеризация завершена.")
