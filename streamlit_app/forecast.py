import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from datetime import timedelta
from io import StringIO, BytesIO
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

# 参数配置（动态可调）
DATA_COLUMNS = ['evaporation_from_bare_soil_sum',
                'total_precipitation_sum',
                'temperature_2m_max',
                'wind_speed_10m']

# 加载模型参数
with open("models/best_params.json", "r") as f:
    best_params = json.load(f)

# 定义模型（支持动态输入维度）
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
def load_model(input_size):
    model = LSTMRunoffModel(input_size, best_params['hidden_size1'], best_params['hidden_size2'], best_params['dropout'])
    model.load_state_dict(torch.load("models/best_lstm_model.pth", map_location="cpu"))
    model.eval()
    return model

# 标准化
def normalize_input(data):
    return (data - data.mean()) / (data.std() + 1e-8)

# 生成动态Excel模板
def create_excel_template(history_days):
    wb = openpyxl.Workbook()
    ws_data = wb.active
    ws_data.title = "数据输入"
    ws_guide = wb.create_sheet(title="填写指南")
    
    # 表头
    headers = ['date'] + DATA_COLUMNS
    ws_data.append(headers)
    
    # 列宽设置
    for col_idx, header in enumerate(headers, 1):
        col_letter = get_column_letter(col_idx)
        if header == 'date':
            ws_data.column_dimensions[col_letter].width = 15
        else:
            ws_data.column_dimensions[col_letter].width = 22
    
    # 生成动态行数的示例数据
    today = pd.Timestamp.today()
    for i in range(history_days):
        date_cell = ws_data[f'A{i+2}']
        date_cell.value = (today + timedelta(days=i)).strftime('%Y-%m-%d')
        date_cell.number_format = 'yyyy-mm-dd'
        
        # 数值列数据验证
        for col_idx in range(2, len(headers)+1):
            col_letter = get_column_letter(col_idx)
            cell = ws_data[f'{col_letter}{i+2}']
            dv = DataValidation(type="decimal", operator="greaterThan", formula1="-1000")
            dv.error = '请输入有效的数值！'
            dv.errorTitle = '输入错误'
            ws_data.add_data_validation(dv)
            dv.add(cell)
    
    # 提示信息（动态行数）
    ws_data[f'A{history_days+3}'] = f"⚠️ 注意：请填写完整{history_days}天的连续数据，不可留空"
    ws_data[f'A{history_days+3}'].font = openpyxl.styles.Font(color="FF0000", bold=True)
    
    # 填写指南（固定内容）
    ws_guide['A1'] = "数据填写指南"
    ws_guide['A3'] = "字段说明："
    for idx, field in enumerate(DATA_COLUMNS, 4):
        ws_guide[f'A{idx}'] = field
        ws_guide[f'B{idx}'] = f"{field} (单位: 请参考模型训练数据)"
    ws_guide['A7'] = "填写要求："
    ws_guide['B7'] = "1. 日期需按升序连续排列"
    ws_guide['B8'] = "2. 数值列不可留空，必须为数字"
    
    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)
    return buffer

# Streamlit 主界面
def run_forecast_module():
    st.title("🌧️ 洪水预报模块")
    
    # 动态参数调节
    st.sidebar.header("参数设置")
    history_days = st.sidebar.slider(
        "历史数据天数",
        min_value=7, max_value=30, value=15, step=1,
        help="用于预测的历史数据天数（需≥7天）"
    )
    forecast_days = st.sidebar.slider(
        "预测天数",
        min_value=1, max_value=14, value=7, step=1,
        help="未来预测的天数（≤14天）"
    )
    
    # 模板下载（动态行数）
    st.header("📁 数据输入")
    excel_buffer = create_excel_template(history_days)
    st.download_button(
        "📊 下载Excel模板",
        data=excel_buffer,
        file_name=f"data_template_{history_days}d.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.download_button(
        "📄 下载CSV模板",
        data=f"date,{','.join(DATA_COLUMNS)}\n" + "\n".join([f"YYYY-MM-DD,," for _ in range(history_days)]),
        file_name=f"data_template_{history_days}d.csv",
        mime="text/csv"
    )
    
    # 数据上传
    st.subheader("上传数据文件")
    uploaded_file = st.file_uploader(
        "上传CSV/Excel文件",
        type=["csv", "xlsx"],
        help=f"需包含{history_days}天连续数据"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"文件读取失败: {e}")
            return
        
        # 数据校验
        if not set(['date'] + DATA_COLUMNS).issubset(df.columns):
            st.error(f"缺少必要列！需包含: date, {', '.join(DATA_COLUMNS)}")
            return
        if len(df) < history_days:
            st.error(f"数据不足！需提供至少{history_days}天数据")
            return
        
        # 预处理
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').tail(history_days)  # 取最新N天数据
        features = normalize_input(df[DATA_COLUMNS].values)
        last_date = df['date'].iloc[-1]
        
        # 模型预测
        model = load_model(input_size=len(DATA_COLUMNS))
        predictions = []
        current_data = features[np.newaxis, :, :]  # (1, history_days, input_size)
        
        for _ in range(forecast_days):
            with torch.no_grad():
                output = model(torch.from_numpy(current_data).float())
                pred = output.numpy()[-1]
                predictions.append(pred)
                
                # 滚动更新输入数据（使用最后一个时间步作为下一时刻输入）
                current_data = np.concatenate([current_data[:, 1:, :], pred.reshape(1, 1, -1)], axis=1)
        
        # 结果展示
        st.header("📈 预测结果")
        pred_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        result_df = pd.DataFrame({
            '日期': pred_dates,
            '预测径流量': predictions
        })
        st.dataframe(result_df.style.format({"预测径流量": "{:.2f}"}))
        st.line_chart(result_df.set_index('日期'))
        
        # 下载结果
        st.download_button(
            "📥 下载预测结果",
            data=result_df.to_csv(index=False).encode('utf-8'),
            file_name=f"forecast_{forecast_days}d.csv"
        )

if __name__ == "__main__":
    run_forecast_module()
