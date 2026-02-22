"""
Скрипт для обучения и сохранения пайплайна XGBoost (препроцессор + модель).
Запустите один раз перед использованием Streamlit-приложения.
"""
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

NUM_COLS = [
    'X', 'Y', 'Z', 'Ore_Grade (%)', 'Tonnage', 'Ore_Value (USD/tonne)',
    'Mining_Cost (USD)', 'Processing_Cost (USD)', 'Waste_Flag'
]
CAT_COLS = ['Rock_Type']
TARGET = 'Profit (USD)'


def main():
    df = pd.read_csv('mining_block_model.csv')
    X_num = df[NUM_COLS].fillna(0)
    X_cat = df[CAT_COLS]
    y = df[TARGET]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), NUM_COLS),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), CAT_COLS)
    ])
    X_full = preprocessor.fit_transform(pd.concat([X_num, X_cat], axis=1))

    X_train, _, y_train, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    pipeline = {
        'preprocessor': preprocessor,
        'model': model,
        'num_cols': NUM_COLS,
        'cat_cols': CAT_COLS,
    }
    joblib.dump(pipeline, 'xgboost_pipeline.joblib')
    print('Модель сохранена в xgboost_pipeline.joblib')


if __name__ == '__main__':
    main()
