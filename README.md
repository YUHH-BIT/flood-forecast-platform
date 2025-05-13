# 项目结构说明（Streamlit 版本）
# flood-forecast-platform/
# ├── streamlit_app/
# │   ├── app.py                # ✅ Streamlit 主入口（使用 streamlit run 运行）
# │   ├── forecast.py           # 洪水预报模块
# │   ├── data_query.py         # 数据查询模块（使用 SQLite）
# │   ├── alert.py              # 预警模块（逻辑处理）
# │   └── utils/
# │       └── __init__.py       # 可放公用函数或工具类
# │
# ├── data/
# │   └── processed/
# │       └── flood_warning.db  # SQLite 数据库（用于数据查询）
# │
# ├── models/
# │   └── best_lstm_model.pth   # 训练好的 LSTM 模型权重
# │
# ├── notebooks/
# │   └── model_training.ipynb  # 模型训练记录
# │
# ├── README.md

# ✅ 使用方式：
# 1. 安装依赖：pip install -r requirements.txt
# 2. 运行应用：streamlit run streamlit_app/app.py
