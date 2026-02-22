"""
Streamlit-сервис для предсказания Profit (USD) с помощью модели XGBoost.
Дизайн в зелёных тонах.
"""
import streamlit as st
import pandas as pd
import joblib
import os

# Настройка страницы
st.set_page_config(
    page_title='Прогноз прибыли блока',
    page_icon='⛏️',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Зелёная тема: кастомный CSS
st.markdown("""
<style>
    /* Основной фон и акценты */
    .stApp {
        background: linear-gradient(180deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
    }
    /* Заголовки */
    h1, h2, h3 {
        color: #1b5e20 !important;
        font-weight: 600;
    }
    /* Кнопка */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(46, 125, 50, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #388e3c 0%, #2e7d32 100%) !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.5);
    }
    /* Блок с результатом */
    .result-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #2e7d32;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.2);
    }
    .result-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1b5e20;
    }
    /* Сайдбар */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #c8e6c9 0%, #a5d6a7 100%);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #1b5e20;
    }
    /* Числовые поля */
    .stNumberInput label {
        color: #2e7d32 !important;
    }
    /* Селектбокс */
    .stSelectbox label {
        color: #2e7d32 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Загрузка сохранённого пайплайна (препроцессор + XGBoost)."""
    path = 'xgboost_pipeline.joblib'
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def main():
    st.title('⛏️ Прогноз прибыли горного блока')
    st.markdown('Предсказание **Profit (USD)** по признакам блоковой модели с помощью XGBoost.')

    pipeline = load_pipeline()
    if pipeline is None:
        st.error(
            'Модель не найдена. Сначала выполните: **python train_and_save_model.py** '
            'в папке с данными mining_block_model.csv.'
        )
        return

    preprocessor = pipeline['preprocessor']
    model = pipeline['model']
    num_cols = pipeline['num_cols']
    cat_cols = pipeline['cat_cols']

    st.sidebar.header('Параметры блока')
    st.sidebar.markdown('Заполните признаки блока и нажмите **Предсказать**.')

    # Форма ввода
    with st.form('prediction_form'):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader('Координаты и тип породы')
            x = st.number_input('X', min_value=0, max_value=500, value=100, step=1)
            y = st.number_input('Y', min_value=0, max_value=500, value=200, step=1)
            z = st.number_input('Z', min_value=0, max_value=100, value=50, step=1)
            rock_type = st.selectbox('Rock_Type', ['Magnetite', 'Hematite', 'Waste'])

        with c2:
            st.subheader('Руда и тоннаж')
            ore_grade = st.number_input(
                'Ore_Grade (%)', min_value=0.0, max_value=100.0, value=55.0, step=1.0, format='%.1f'
            )
            tonnage = st.number_input('Tonnage', min_value=500, max_value=5000, value=2000, step=100)
            ore_value = st.number_input(
                'Ore_Value (USD/tonne)', min_value=0.0, max_value=5.0, value=1.9, step=0.1, format='%.2f'
            )

        with c3:
            st.subheader('Затраты и флаг')
            mining_cost = st.number_input(
                'Mining_Cost (USD)', min_value=0.1, max_value=1.0, value=0.35, step=0.01, format='%.2f'
            )
            processing_cost = st.number_input(
                'Processing_Cost (USD)', min_value=0.1, max_value=0.5, value=0.22, step=0.01, format='%.2f'
            )
            waste_flag = st.selectbox('Waste_Flag', [0, 1], format_func=lambda x: 'Нет (руда)' if x == 0 else 'Да (отходы)')

        submitted = st.form_submit_button('Предсказать Profit')

    if submitted:
        # Порядок столбцов как при обучении: num_cols + cat_cols
        row = pd.DataFrame([{
            'X': x, 'Y': y, 'Z': z,
            'Ore_Grade (%)': ore_grade,
            'Tonnage': tonnage,
            'Ore_Value (USD/tonne)': ore_value,
            'Mining_Cost (USD)': mining_cost,
            'Processing_Cost (USD)': processing_cost,
            'Waste_Flag': waste_flag,
            'Rock_Type': rock_type,
        }])
        row = row[num_cols + cat_cols]
        X = preprocessor.transform(row)
        profit_pred = model.predict(X)[0]

        st.markdown('---')
        st.markdown(
            f'<div class="result-box">'
            f'<div>Предсказанная прибыль</div>'
            f'<div class="result-value">{profit_pred:,.2f} USD</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        if profit_pred > 0:
            st.success(f'Блок экономически целесообразен к выемке (прибыль > 0).')
        else:
            st.warning('Блок убыточен; рекомендуется не включать в контур добычи.')

    st.sidebar.markdown('---')
    st.sidebar.markdown('*Модель: XGBoost, обучена на mining_block_model.csv*')


if __name__ == '__main__':
    main()
