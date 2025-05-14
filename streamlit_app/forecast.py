import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import timedelta
from io import BytesIO

def show_predict_page():
    st.title("🌧️ LSTM 径流预测系统")

# 定义模型结构
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

# 页面设置
st.set_page_config(page_title="LSTM径流预测系统", layout="wide")
st.title("🌧️ LSTM 径流预测系统")
st.write("欢迎使用径流预测工具，您可以上传数据并生成预测结果。")

# 模型加载函数
def load_model():
    model = LSTMRunoffModel(input_size=4, hidden_size1=hidden_size1, hidden_size2=hidden_size2, dropout=dropout)
    try:
        # 更新后的模型路径
        model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location=torch.device("cpu")))
        model.eval()
        return model
    except FileNotFoundError:
        st.error("❌ 模型权重文件未找到，请确保文件 'models/best_lstm_model.pth' 存在于当前目录中。")
        return None
    except Exception as e:
        st.error(f"❌ 加载模型时出错：{e}")
        return None

# 参数选择
with st.sidebar:
    st.header("⚙️ 模型参数设置")
    history_days = st.slider("输入历史天数", 7, 60, 15)
    forecast_days = st.slider("预测未来天数", 1, 30, 7)
    hidden_size1 = st.slider("LSTM 层1隐藏单元数", 32, 512, 80, step=16)
    hidden_size2 = st.slider("LSTM 层2隐藏单元数", 32, 512, 240, step=16)
    dropout = st.slider("Dropout 概率", 0.0, 0.5, 0.1, step=0.05)

# 输入方式选择
data_input_method = st.radio("选择输入数据方式", ("上传 Excel 文件", "表格方式输入数据"))

# 上传方式
if data_input_method == "上传 Excel 文件":
    st.subheader("📁 上传包含天气数据的 Excel 文件")
    uploaded_file = st.file_uploader("选择 Excel 文件", type=['xlsx'])

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            st.write("✅ 数据预览：", df.head())

            # 数据验证
            required_columns = ['date', 'evaporation_from_bare_soil_sum', 'total_precipitation_sum', 'temperature_2m_max', 'wind_speed_10m']
            if not all(column in df.columns for column in required_columns):
                st.error(f"❌ 数据缺少必要列，请确保包含以下列：{', '.join(required_columns)}")
            else:
                if st.button("🚀 开始预测"):
                    # 预测逻辑
                    try:
                        features = df[required_columns[1:]].values
                        dates = pd.to_datetime(df['date'].values)
                        last_history = features[-history_days:]
                        last_date = dates[-1]

                        model = load_model()
                        if model is None:
                            st.stop()

                        predictions = []
                        prediction_dates = []

                        for i in range(forecast_days):
                            input_data = np.expand_dims(last_history, axis=0)
                            X_tensor = torch.tensor(input_data, dtype=torch.float32)
                            with torch.no_grad():
                                output = model(X_tensor)
                                output = torch.clamp(output, min=0)
                                prediction = output.numpy()[0, -1]
                                predictions.append(prediction)

                            new_input = features[-1].copy()
                            last_history = np.vstack([last_history[1:], new_input])
                            prediction_dates.append(last_date + timedelta(days=i + 1))

                        result_df = pd.DataFrame({
                            'date': prediction_dates,
                            'predicted_runoff': predictions
                        })

                        st.success("✅ 预测完成！结果如下：")
                        st.dataframe(result_df)

                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            result_df.to_excel(writer, index=False, sheet_name='Prediction')
                        st.download_button("📥 下载预测结果",
                                           data=output.getvalue(),
                                           file_name="runoff_prediction.xlsx",
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
                    except Exception as e:
                        st.error(f"❌ 预测时出错：{e}")
        except Exception as e:
            st.error(f"❌ 文件处理时出错：{e}")

# 表格方式输入
elif data_input_method == "表格方式输入数据":
    st.subheader(f"📝 批量输入历史天气数据（共 {history_days} 天）")

    # 创建空白表格
    empty_data = pd.DataFrame({
        "date": [""] * history_days,
        "evaporation_from_bare_soil_sum": [None] * history_days,
        "total_precipitation_sum": [None] * history_days,
        "temperature_2m_max": [None] * history_days,
        "wind_speed_10m": [None] * history_days
    })

    st.write("请完整填写每一行的数据（日期格式如 2024-01-01）：")
    edited_data = st.data_editor(
        empty_data,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True
    )

    if st.button("🚀 开始预测"):
        try:
            # 检查空值
            if edited_data.isnull().any().any() or (edited_data["date"] == "").any():
                raise ValueError("所有字段（包括日期和气象数据）都必须填写，不能有空值")

            # 检查日期格式
            edited_data["date"] = pd.to_datetime(edited_data["date"], errors="raise")

            features = edited_data[
                ['evaporation_from_bare_soil_sum',
                 'total_precipitation_sum',
                 'temperature_2m_max',
                 'wind_speed_10m']
            ].astype(np.float32).values

            last_date = edited_data["date"].max()

            model = load_model()
            if model is None:
                st.stop()

            predictions = []
            prediction_dates = []

            for i in range(forecast_days):
                input_data = np.expand_dims(features[-history_days:], axis=0)
                X_tensor = torch.tensor(input_data, dtype=torch.float32)
                with torch.no_grad():
                    output = model(X_tensor)
                    output = torch.clamp(output, min=0)
                    prediction = output.numpy()[0, -1]
                    predictions.append(prediction)

                new_input = features[-1].copy()
                features = np.vstack([features[1:], new_input])
                prediction_dates.append(last_date + timedelta(days=i + 1))

            result_df = pd.DataFrame({
                'date': prediction_dates,
                'predicted_runoff': predictions
            })

            st.success("✅ 预测完成！结果如下：")
            st.dataframe(result_df)

            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                result_df.to_excel(writer, index=False, sheet_name='Prediction')
            st.download_button("📥 下载预测结果",
                               data=output.getvalue(),
                               file_name="runoff_prediction.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"❌ 预测时出错：{e}")
