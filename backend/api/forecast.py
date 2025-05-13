from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from backend.models.lstm_model import LSTMRunoffModel
from backend.utils.data_utils import preprocess_input_data

router = APIRouter()

# 请求模型参数
class ForecastParams(BaseModel):
    history_days: int = 15
    forecast_days: int = 7

# 加载模型（只加载一次）
model_path = "backend/models/best_lstm_model.pth"
input_size = 4
hidden_size1 = 80
hidden_size2 = 240
dropout = 0.1

model = LSTMRunoffModel(input_size, hidden_size1, hidden_size2, dropout)
try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    raise RuntimeError(f"模型加载失败: {e}")

@router.post("/file")
async def forecast_from_excel(file: UploadFile = File(...), params: ForecastParams = ForecastParams()):
    """
    接收上传的 Excel 文件，输出未来若干天的径流预测
    """
    try:
        df = pd.read_excel(file.file)
        required_columns = ['date', 'evaporation_from_bare_soil_sum', 'total_precipitation_sum', 'temperature_2m_max', 'wind_speed_10m']
        if not all(col in df.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Excel 缺少必要列：{required_columns}")

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # 预处理数据（可扩展逻辑至 data_utils）
        features = df[required_columns[1:]].values.astype(np.float32)
        dates = df['date'].values
        last_history = features[-params.history_days:]
        last_date = df['date'].iloc[-1]

        predictions = []
        prediction_dates = []

        for i in range(params.forecast_days):
            input_data = np.expand_dims(last_history, axis=0)
            X_tensor = torch.tensor(input_data, dtype=torch.float32)
            with torch.no_grad():
                output = model(X_tensor)
                prediction = torch.clamp(output, min=0).numpy()[0, -1]
                predictions.append(float(prediction))

            last_history = np.vstack([last_history[1:], last_history[-1]])  # 重复最后一行数据
            prediction_dates.append(last_date + timedelta(days=i + 1))

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in prediction_dates],
            "predicted_runoff": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败：{e}")
