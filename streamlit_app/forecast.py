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

# LSTM 模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out = self.fc(out[:, -1, :])
        return out


# 载入模型
@st.cache_resource
def load_model(input_size, hidden_size, num_layers):
    model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 数据标准化
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
    st.write("上传最新气象数据（Excel 或 CSV），进行未来月径流预测。")

    # 用户输入模型参数
    st.sidebar.header("模型参数配置")
    input_size = st.sidebar.number_input("输入特征数 (input_size)", min_value=1, value=4)
    input_seq_len = st.sidebar.number_input("输入时间步长 (input_seq_len)", min_value=1, value=12)  # 默认 12 个月
    output_seq_len = st.sidebar.number_input("输出时间步长 (output_seq_len)", min_value=1, value=1)  # 默认 1 个月

    # 提供数据模板下载
    if st.sidebar.button("📥 下载数据模板"):
        st.sidebar.write("数据模板：")
        st.sidebar.write("date,evaporation_from_bare_soil_sum,total_precipitation_sum,temperature_2m_max,wind_speed_10m")
        st.sidebar.write("2025-01-01,1.2,3.4,5.6,7.8")
        st.sidebar.write("...")

    # 支持手动输入数据
    manual_input = st.checkbox("手动输入数据")
    if manual_input:
        st.write("请手动输入数据（以逗号或制表符分隔）：")
        raw_data = st.text_area("输入格式：date,evaporation_from_bare_soil_sum,total_precipitation_sum,temperature_2m_max,wind_speed_10m\n例如：2025-01-01,1.2,3.4,5.6,7.8")
        try:
            from io import StringIO
            if ',' in raw_data:
                df = pd.read_csv(StringIO(raw_data))
            else:
                df = pd.read_csv(StringIO(raw_data), sep="\t")
            st.write("✅ 数据预览：", df.head())
        except Exception as e:
            st.error(f"❌ 数据格式有误：{e}")
            return
    else:
        uploaded_file = st.file_uploader("📤 上传 Excel 或 CSV 文件（需包含: date, evaporation_from_bare_soil_sum, total_precipitation_sum, temperature_2m_max, wind_speed_10m 列）")
        if uploaded_file:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    df = pd.read_excel(uploaded_file)
                st.write("✅ 数据预览：", df.head())
            except Exception as e:
                st.error(f"❌ 文件读取失败：{e}")
                return
        else:
            st.warning("请上传数据文件或切换到手动输入模式。")
            return

    # 数据检查和处理
    try:
        # 修改后的必需列
        required_columns = ['date', 'evaporation_from_bare_soil_sum', 'total_precipitation_sum', 'temperature_2m_max', 'wind_speed_10m']
        if not set(required_columns).issubset(df.columns):
            missing_cols = set(required_columns) - set(df.columns)
            st.error(f"❌ 缺少所需列：{missing_cols}")
            return
        
        feature_columns = ['evaporation_from_bare_soil_sum', 'total_precipitation_sum', 'temperature_2m_max', 'wind_speed_10m']
        features = normalize_input(df[feature_columns])

        features_tensor = torch.tensor(features[-input_seq_len:].values, dtype=torch.float32).unsqueeze(0)

        # 动态加载模型
        model = load_model(input_size, hidden_size=80, num_layers=1)  # 固定 hidden_size 和 num_layers

        # 执行预测
        prediction = make_forecast(model, features_tensor)
        st.success(f"🌊 预测结果：未来 {output_seq_len} 径流量为 **{prediction[0][0]:.2f} m³/s**")

    except Exception as e:
        st.error(f"❌ 处理数据时出错：{e}")

# 运行主模块
if __name__ == "__main__":
    run_forecast_module()
