flood-forecast-platform/
├── streamlit_app/
│   ├── app.py                  # Streamlit 主界面入口
│   ├── forecast.py             # 模型预测逻辑（可调用 LSTM）
│   ├── data_query.py           # SQLite 数据查询
│   ├── alert.py                # 阈值判断 + 通知
│   └── utils/
│       └── data_utils.py       # 数据预处理
├── data/
│   ├── flood_warning.db    # SQLite 数据库
├── notebooks/
├── README.md
