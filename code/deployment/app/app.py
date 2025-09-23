import streamlit as st
import requests

st.title("Предсказание финансирования режиссёра")

avg_budget = st.number_input("Средний бюджет фильмов (avg_budget)", min_value=0, value=1000000)
avg_revenue = st.number_input("Средний доход фильмов (avg_revenue)", min_value=0, value=5000000)
avg_vote = st.number_input("Средний рейтинг фильмов (avg_vote)", min_value=0.0, max_value=10.0, value=7.5)
num_movies = st.number_input("Количество фильмов (num_movies)", min_value=0, value=5)
avg_popularity = st.number_input("Средняя популярность (avg_popularity)", min_value=0.0, value=10.0)
avg_vote_count = st.number_input("Среднее количество голосов (avg_vote_count)", min_value=0, value=2000)
age = st.number_input("Возраст режиссёра (age)", min_value=0, value=55)

if st.button("Сделать предсказание"):
    data = {
        "avg_budget": avg_budget,
        "avg_revenue": avg_revenue,
        "avg_vote": avg_vote,
        "num_movies": num_movies,
        "avg_popularity": avg_popularity,
        "avg_vote_count": avg_vote_count,
        "age": age
    }

    try:
        response = requests.post("http://api:8000/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Класс: {result['predicted_class']}\nОписание: {result['description']}")
        else:
            st.error(f"Ошибка: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Не удалось отправить запрос: {e}")
