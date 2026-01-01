import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Прогноз цены автомобиля")
st.write("EDA, предсказания и интерпретация линейной модели")
st.write("Используется ElasticNet, alpha=1, l1_ratio=0.9")

model = joblib.load('model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

st.header("Exploratory Data Analysis")

uploaded_file = st.file_uploader("Загрузите CSV-файл для EDA", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    num_cols = df.select_dtypes(include='number').columns

    col = st.selectbox("Выберите числовой признак", num_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=30, ax=ax)
    st.pyplot(fig)

st.header("Прогноз по CSV")

uploaded_pred = st.file_uploader("Загрузите CSV для предсказания", type="csv", key="pred")

if uploaded_pred is not None:
    df_pred = pd.read_csv(uploaded_pred)

    X = preprocessor.transform(df_pred)
    preds = model.predict(X)

    df_pred['predicted_price'] = preds
    st.write(df_pred)


st.header("Веса модели")

feature_names = preprocessor.get_feature_names_out()
coefs = model.coef_

coef_df = pd.DataFrame({
    'feature': feature_names,
    'coef': coefs
}).sort_values(by='coef', key=np.abs, ascending=False)

fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(
    x='coef',
    y='feature',
    data=coef_df.head(15),
    ax=ax
)

st.pyplot(fig)

