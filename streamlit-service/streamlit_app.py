import streamlit as st
import requests
from requests.exceptions import ConnectionError

ip_api = "ml-api"
port_api = "5000"

# Заголовок приложения
st.title("Insurance Selling Prediction")

# Ввод данных
st.write("Enter the client details:")

# Выпадающее меню для выбора класса билета
is_male = st.selectbox("Gender", ["Male","Female"])

# Текстовое поле для ввода возраста с проверкой на число
age = st.text_input("Age", value=10)
if not age.isdigit():
    st.error("Please enter a valid number for Age.")

driving_license = st.selectbox("Driving license keeping", [True,False])

previously_insured = st.selectbox("Previously Insured", [True,False])

vehicle_age = st.selectbox("Vehicle age", ["< 1 Year","1-2 Year","> 2 Years"])

vehicle_damage = st.selectbox("Vehicle Damage", [True,False])

annual_premium = st.text_input("Annual Premium", value=100)
if not annual_premium.isdigit():
    st.error("Please enter a valid number for Annual Premium.")

vintage = st.text_input("Vintage", value=100)
if not vintage.isdigit():
    st.error("Please enter a valid number for Vintage.")

# Кнопка для отправки запроса
if st.button("Predict"):
    # Проверка, что все поля заполнены
    if age.isdigit() and annual_premium.isdigit() and vintage.isdigit():
        # Подготовка данных для отправки
        data = {
            "Is_Male": int( 1 if is_male == "Male" else 0),  # Нормализованные данные
            "Age": int(age),
            "Driving_License": int( 1 if driving_license == True else 0),
            "Previously_Insured": int( 1 if previously_insured == True else 0),
            "Vehicle_Age": int(0 if vehicle_age == "< 1 Year" else (1 if vehicle_age == "1-2 Year" else 2)),
            "Vehicle_Damage": int( 1 if vehicle_damage == True else 0),
            "Annual_Premium": float(annual_premium),
            "Vintage": int(vintage)
        }

        try:
            # Отправка запроса к Flask API
            response = requests.post(f"http://{ip_api}:{port_api}/predict_model", json=data)

            # Проверка статуса ответа
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"Prediction: {prediction}")
            else:
                st.error(f"Request failed with status code {response.status_code}")
        except ConnectionError as e:
            st.error(f"Failed to connect to the server")
    else:
        st.error("Please fill in all fields with valid numbers.")