# frontend/streamlit_ui/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="洪水预报预警平台", layout="wide")

st.title("🌊 洪水预报预警平台")

# 选项卡
tab = st.sidebar.selectbox("选择功能", ["洪水预报", "洪水预警", "数据查询"])

if tab == "洪水预报":
    st.header("📈 洪水预报")
    st.markdown("请上传气象输入数据：")
    uploaded_file = st.file_uploader("上传 CSV 文件", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("输入数据预览：", df.head())

        # 示例预测请求（你需将此 URL 替换为实际 API）
        if st.button("开始预测"):
            st.success("预测完成（示例值）：")
            st.line_chart(np.random.rand(12) * 100)

elif tab == "洪水预警":
    st.header("⚠️ 洪水预警")
    level = st.slider("设置告警阈值（单位：m³/s）", 0, 1000, 300)
    st.write("当前设置的阈值为：", level)

    if st.button("模拟触发预警"):
        st.warning("⚠️ 预测值超过阈值！已发送预警通知。")

elif tab == "数据查询":
    st.header("📊 历史数据查询")
    date = st.date_input("选择查询日期")
    if st.button("查询"):
        st.info(f"显示 {date} 附近的历史流量数据（模拟）：")
        st.line_chart(np.random.rand(10) * 500)

