from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List
import sqlite3
import pandas as pd
from datetime import datetime

router = APIRouter()

# 数据库路径（你可以根据实际路径修改）
DB_PATH = "data/flood_warning.db"

def get_table_names():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception as e:
        raise RuntimeError(f"获取表名失败: {e}")

@router.get("/tables")
def list_tables():
    try:
        tables = get_table_names()
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/query")
def query_data(
    table_name: str = Query(..., description="要查询的表名"),
    start_date: Optional[str] = Query(None, description="起始日期（格式 YYYY-MM-DD）"),
    end_date: Optional[str] = Query(None, description="结束日期（格式 YYYY-MM-DD）")
):
    tables = get_table_names()
    if table_name not in tables:
        raise HTTPException(status_code=400, detail="无效的表名")

    try:
        conn = sqlite3.connect(DB_PATH)
        query = f"SELECT * FROM {table_name}"
        conditions = []

        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        df = pd.read_sql_query(query, conn)
        conn.close()

        if df.empty:
            return {"message": "没有找到符合条件的数据。"}

        return {
            "columns": df.columns.tolist(),
            "data": df.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")

