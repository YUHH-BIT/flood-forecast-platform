# streamlit_app/app.py

import streamlit as st
from forecast import run_forecast_module
from data_query import run_query_module
from alert import run_alert_module

# 页面设置
st.set_page_config(page_title="洪水预报预警平台", layout="wide", page_icon="🌊")

# 侧边栏导航
st.sidebar.title("📊 洪水预报预警平台")
app_mode = st.sidebar.radio(
    "选择功能模块",
    ["主页", "洪水预报", "数据查询", "预警分析"]
)

# 主页面内容
if app_mode == "主页":
    st.title("🌊 洪水预报预警平台")
    st.markdown(
        """
        欢迎使用洪水预报预警平台！该系统集成了数据查询、洪水预报与预警功能，支持气象输入、模型预测、预警判断与结果导出等操作。

        **模块介绍：**
        - 📈 **洪水预报**：支持上传气象数据，使用 LSTM 模型进行未来流量预测。
        - 🗃️ **数据查询**：支持按日期查询历史观测数据，基于 SQLite 数据库。
        - 🚨 **预警分析**：基于预测结果，结合阈值设定给出洪水预警提示。
        """
    )

elif app_mode == "洪水预报":
    run_forecast_module()

elif app_mode == "数据查询":
    run_query_module()

elif app_mode == "预警分析":
    run_alert_module()

