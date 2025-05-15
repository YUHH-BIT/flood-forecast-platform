# streamlit_app/data_query.py

import streamlit as st
import sqlite3
import pandas as pd
from datetime import datetime
from io import BytesIO
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

DB_PATH = "data/processed/flood_warning.db"

def run_query_module():
    st.title("🗃️ 数据查询模块")
    st.write("请选择数据表并输入查询日期")

    # 建立数据库连接并获取表名
    try:
        conn = sqlite3.connect(DB_PATH)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        table_names = tables['name'].tolist()
    except Exception as e:
        st.error(f"❌ 数据库连接失败：{e}")
        return

    # 表选择与日期输入
    selected_table = st.selectbox("选择数据表", table_names)
    query_date = st.text_input("输入日期（格式：YYYY 或 YYYY-MM 或 YYYY-MM-DD）", "")

    if st.button("🔍 查询"):
        if not query_date:
            st.warning("⚠️ 请输入日期后再查询。")
            return

        try:
            # 根据日期格式构建 SQL 查询
            if len(query_date) == 4:
                start = f"{query_date}-01-01"
                end = f"{int(query_date)+1}-01-01"
                sql = f"SELECT * FROM {selected_table} WHERE date >= ? AND date < ?"
                params = (start, end)
            elif len(query_date) == 7:
                start = f"{query_date}-01"
                year, month = map(int, query_date.split("-"))
                if month == 12:
                    end = f"{year+1}-01-01"
                else:
                    end = f"{year}-{month+1:02d}-01"
                sql = f"SELECT * FROM {selected_table} WHERE date >= ? AND date < ?"
                params = (start, end)
            else:
                parsed = datetime.strptime(query_date, "%Y-%m-%d").date()
                sql = f"SELECT * FROM {selected_table} WHERE date = ?"
                params = (parsed,)
        except ValueError:
            st.error("❌ 日期格式错误，请输入合法日期（如 2023 或 2023-07 或 2023-07-15）")
            return

        try:
            df = pd.read_sql(sql, conn, params=params)
            if df.empty:
                st.info("🔍 没有查到对应日期的数据。")
            else:
                st.success(f"✅ 查询到 {len(df)} 条数据：")
                st.dataframe(df)

                # 保存为 Excel
                output = BytesIO()
                wb = Workbook()
                ws = wb.active
                ws.title = "查询结果"
                for row in dataframe_to_rows(df, index=False, header=True):
                    ws.append(row)
                wb.save(output)
                output.seek(0)

                st.download_button(
                    "📥 下载结果",
                    data=output,
                    file_name="query_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.error(f"❌ 查询失败：{e}")
        finally:
            conn.close()
