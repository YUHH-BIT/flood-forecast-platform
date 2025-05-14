# streamlit_app/direct_forecast.py
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from datetime import timedelta
from io import StringIO

# 参数配置
DATA_COLUMNS = ['evaporation_from_bare_soil_sum',
                'total_precipitation_sum',
                'temperature_2m_max',
                'wind_speed_10m']

HISTORY_DAYS = 15
FORECAST_DAYS = 7
INPUT_SIZE = len(DATA_COLUMNS)

# 加载模型参数
with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

# 定义模型
class LSTMRunoffModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.fc = nn.Linear(hidden_size2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        return out.squeeze(-1)

# 加载模型
@st.cache_resource
def load_model():
    model = LSTMRunoffModel(INPUT_SIZE, best_params['hidden_size1'], best_params['hidden_size2'], best_params['dropout'])
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location="cpu"))
    model.eval()
    return model

# 标准化
def normalize_input(data):
    return (data - data.mean()) / (data.std() + 1e-8)

# Streamlit 页面
def run_direct_forecast():
    st.title("📈 多步径流预测（滑动窗口）")
    st.write(f"基于最近 {HISTORY_DAYS} 天气象数据，预测未来 {FORECAST_DAYS} 天径流值。")

    # 手动输入 or 文件上传
    manual_input = st.checkbox("手动输入数据")
    df = None

    if manual_input:
        text = st.text_area("输入格式：date,evaporation_from_bare_soil_sum,total_precipitation_sum,temperature_2m_max,wind_speed_10m")
        if text:
            try:
                df = pd.read_csv(StringIO(text)) if ',' in text else pd.read_csv(StringIO(text), sep="\t")
                st.success("✅ 数据读取成功")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ 数据读取失败：{e}")
                return
    else:
        uploaded = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])
        if uploaded:
            try:
                df = pd.read_csv(uploaded) if uploaded.name.endswith("csv") else pd.read_excel(uploaded)
                st.success("✅ 文件读取成功")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"❌ 文件读取失败：{e}")
                return
        else:
            st.warning("请上传数据文件或使用手动输入模式")
            return

    # 检查数据完整性
    if not set(['date'] + DATA_COLUMNS).issubset(df.columns):
        st.error(f"❌ 数据缺失必要列，请确保包含：date + {DATA_COLUMNS}")
        return

    df = df.dropna()
    df['date'] = pd.to_datetime(df['date'])
    features = df[DATA_COLUMNS].values
    dates = df['date'].values

    if len(features) < HISTORY_DAYS:
        st.error(f"❌ 数据长度不足 {HISTORY_DAYS} 天")
        return

    model = load_model()
    last_history = features[-HISTORY_DAYS:]
    last_date = pd.to_datetime(dates[-1])
    predictions, pred_dates = [], []

    for i in range(FORECAST_DAYS):
        input_tensor = torch.tensor(np.expand_dims(last_history, axis=0), dtype=torch.float32)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = output.numpy()[0, -1]
            predictions.append(prediction)
        new_input = last_history[-1]  # 简化处理：用最后一行输入复制
        last_history = np.vstack([last_history[1:], new_input])
        pred_dates.append(last_date + timedelta(days=i+1))

    # 展示结果
    result_df = pd.DataFrame({
        'date': pred_dates,
        'predicted_runoff': predictions
    })
    st.success("✅ 预测完成")
    st.dataframe(result_df)

    # 下载
    st.download_button("📥 下载预测结果", data=result_df.to_csv(index=False).encode('utf-8'), file_name="direct_forecast.csv")

# 运行页面
if __name__ == "__main__":
    run_direct_forecast()
