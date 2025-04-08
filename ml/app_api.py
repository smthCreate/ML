from fastapi import FastAPI, Request, HTTPException
import pickle
import pandas as pd
from pydantic import BaseModel
import uvicorn

app = FastAPI()

# Загрузка модели из файла pickle
with open('ml/clf_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Счетчик запросов
request_count = 0
# Модель для валидации входных данных
class PredictionInput(BaseModel):
    Is_Male: bool
    Age: int
    Driving_License: bool
    Previously_Insured: bool
    Vehicle_Age: int
    Vehicle_Damage: bool
    Annual_Premium: float
    Vintage: int

@app.get("/stats")
def stats():
    return {"request_count": request_count}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict_model")
def predict_model(input_data: PredictionInput):
    global request_count
    request_count += 1

    # Создание DataFrame из данных
    new_data = pd.DataFrame({
        'Is_Male': [input_data.Is_Male],
        'Age': [input_data.Age],
        'Driving_License': [input_data.Driving_License],
        'Previously_Insured': [input_data.Previously_Insured],
        'Vehicle_Age': [input_data.Vehicle_Age],
        'Vehicle_Damage': [input_data.Vehicle_Damage],
        'Annual_Premium': [input_data.Annual_Premium],
        'Vintage': [input_data.Vintage],

    })

    # Предсказание
    predictions = model.predict(new_data)

    # Преобразование результата в человеко-читаемый формат
    result = "Response negative" if predictions[0] == 1 else "Response positive"

    return {"prediction": result}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)