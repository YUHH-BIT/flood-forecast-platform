import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import timedelta

# 基于历史 {15} 天输入，预测未来 {7} 天输出的滑动窗口多步预测
# ===================== 可调参数 =====================
history_days = 15   # 输入历史天数（例如：10 天）
forecast_days = 7   # 预测未来天数（例如：3 天）
input_size = 4      # 输入特征数量
data_path = r'E:/data/daily_weather_data.xlsx'
output_path = f'E:/direct_forecast{history_days}_{forecast_days}.xlsx'
# ===================================================

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
        return out.squeeze(-1)  # [batch, seq_len]

# 加载最优参数
with open('best_params.json', 'r') as f:
    best_params = json.load(f)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRunoffModel(
    input_size=input_size,
    hidden_size1=best_params['hidden_size1'],
    hidden_size2=best_params['hidden_size2'],
    dropout=best_params['dropout']
).to(device)

# 加载模型权重
model.load_state_dict(torch.load('best_lstm_model.pth', map_location=device))
model.eval()

# 读取数据
data = pd.read_excel(data_path)
features = data[['evaporation_from_bare_soil_sum',
                 'total_precipitation_sum',
                 'temperature_2m_max',
                 'wind_speed_10m']].values
dates = pd.to_datetime(data['date'].values)

# 取最近的 history_days 天数据
last_history = features[-history_days:]
last_date = dates[-1]

# 初始化预测结果
predictions = []
prediction_dates = []

# 开始逐步预测
for i in range(forecast_days):
    input_data = np.expand_dims(last_history, axis=0)
    X_tensor = torch.tensor(input_data, dtype=torch.float32).to(device)

    with torch.no_grad():
        output = model(X_tensor)
        output = torch.clamp(output, min=0)
        prediction = output.cpu().numpy()[0, -1]
        predictions.append(prediction)

    # 更新输入序列（简单假设新预测不会影响输入特征结构）
    new_input = last_history[-1].copy()  # 使用上一日特征填充
    last_history = np.vstack([last_history[1:], new_input])
    prediction_dates.append(last_date + timedelta(days=i+1))

# 保存预测结果
pred_df = pd.DataFrame({
    'date': prediction_dates,
    'predicted_runoff': predictions
})
pred_df.to_excel(output_path, index=False)
