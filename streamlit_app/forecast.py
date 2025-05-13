# streamlit_app/forecast.py

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import os

# 模型路径
MODEL_PATH = "models/best_lstm_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM 模型结构（需和训练时一致）
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# 载入模型
@st.cache_resource
def load_model():
    model = LSTMModel(input_size=4, hidden_size=64)  # 依据训练时配置修改
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 数据标准化（如有必要可换成 MinMaxScaler 等）
def normalize_input(data):
    return (data - data.mean()) / (data.std() + 1e-8)

# 预测函数
def make_forecast(model, input_tensor):
    input_tensor = input_tensor.to(DEVICE)
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.cpu().numpy()

# Streamlit 主界面
def run_forecast_module():
    st.title("🌧️ 洪水预报模块")
    st.write("上传最新气象数据（CSV），进行未来月径流预测。")

    uploaded_file = st.file_uploader("📤 上传 CSV 文件（需包含: evap, precip, temp, wind 列）", type=["csv"])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("✅ 数据预览：", df.head())

            # 检查列
            expected_cols = {"evap", "precip", "temp", "wind"}
            if not expected_cols.issubset(df.columns):
                st.error(f"❌ 缺少所需列：{expected_cols - set(df.columns)}")
                return

            model = load_model()

            # 提取并预处理输入特征
            features = df[["evap", "precip", "temp", "wind"]].values.astype(np.float32)
            features = normalize_input(features)
            features_tensor = torch.tensor(features).unsqueeze(0)  # (1, seq_len, 4)

            # 执行预测
            prediction = make_forecast(model, features_tensor)
            st.success(f"🌊 预测结果：未来月径流量为 **{prediction[0][0]:.2f} m³/s**")

        except Exception as e:
            st.error(f"❌ 处理文件时出错：{e}")

