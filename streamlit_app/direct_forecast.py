import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import timedelta
import json
import io

# ========== Streamlit 页面配置 ==========
st.set_page_config(page_title="多步径流预测", layout="wide")
st.title("📈 基于 LSTM 的多步径流预测系统")

# ========== 模型定义 ==========
class LSTMRunoffModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, dropout=0.1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=hidden_size1, hidden_size=hidden_size2, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = self.fc(out)
        return out.squeeze(-1)

# ========== 参数设置 ==========
with st.sidebar:
    st.header("🔧 参数设置")
    history_days = st.number_input("输入历史天数", min_value=1, max_value=60, value=15)
    forecast_days = st.number_input("预测未来天数", min_value=1, max_value=30, value=7)

    uploaded_model = st.file_uploader("上传模型权重 (.pth)", type=["pth"])
    uploaded_params = st.file_uploader("上传模型参数 (.json)", type=["json"])

# ========== 数据输入 ==========
st.subheader("📤 输入气象数据")
input_method = st.radio("选择输入方式", ["上传Excel", "手动输入"])

if input_method == "上传Excel":
    excel_file = st.file_uploader("上传包含气象特征和日期的Excel文件", type=["xlsx"])
    if excel_file:
        df = pd.read_excel(excel_file)
else:
    default_data = pd.DataFrame({
        'date': pd.date_range(end=pd.Timestamp.today(), periods=history_days),
        'evaporation_from_bare_soil_sum': [0.1] * history_days,
        'total_precipitation_sum': [5.0] * history_days,
        'temperature_2m_max': [22.0] * history_days,
        'wind_speed_10m': [2.5] * history_days
    })
    df = st.data_editor(default_data, num_rows="dynamic")

# ========== 预测并展示结果 ==========
if st.button("🚀 开始预测"):
    if uploaded_model and uploaded_params and df is not None:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            params = json.load(uploaded_params)

            model = LSTMRunoffModel(
                input_size=4,
                hidden_size1=params['hidden_size1'],
                hidden_size2=params['hidden_size2'],
                dropout=params.get('dropout', 0.1)
            ).to(device)

            buffer = io.BytesIO(uploaded_model.read())
            buffer.seek(0)
            model.load_state_dict(torch.load(buffer, map_location=device))
            model.eval()

            # 预处理
            data = df.copy()
            data['date'] = pd.to_datetime(data['date'])
            data = data.sort_values('date')

            features = data[['evaporation_from_bare_soil_sum',
                             'total_precipitation_sum',
                             'temperature_2m_max',
                             'wind_speed_10m']].values
            dates = data['date'].values

            if len(features) < history_days:
                st.error(f"❌ 数据不足，至少需要 {history_days} 天历史数据")
            else:
                last_history = features[-history_days:]
                last_date = pd.to_datetime(dates[-1])

                predictions, prediction_dates = [], []

                for i in range(forecast_days):
                    input_data = np.expand_dims(last_history, axis=0)
                    X_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

                    with torch.no_grad():
                        output = model(X_tensor)
                        output = torch.clamp(output, min=0)
                        prediction = output.cpu().numpy()[0, -1]
                        predictions.append(prediction)

                    new_input = last_history[-1].copy()
                    last_history = np.vstack([last_history[1:], new_input])
                    prediction_dates.append(last_date + timedelta(days=i+1))

                result_df = pd.DataFrame({
                    'date': prediction_dates,
                    'predicted_runoff': predictions
                })

                st.success("✅ 预测完成！以下为结果：")
                st.dataframe(result_df)

                # 下载功能
                towrite = io.BytesIO()
                result_df.to_excel(towrite, index=False, engine='openpyxl')
                towrite.seek(0)
                st.download_button("📥 下载预测结果", data=towrite,
                                   file_name=f"runoff_prediction_{history_days}_{forecast_days}.xlsx")
        except Exception as e:
            st.error(f"❌ 预测过程中出错：{e}")
    else:
        st.warning("⚠️ 请上传模型权重和参数文件，并确保输入数据完整。")
