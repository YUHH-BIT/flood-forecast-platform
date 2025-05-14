import pandas as pd
import sqlite3
import os
from datetime import datetime


def excel_to_sqlite(excel_file_path,
                    sqlite_db_path='flood_warning.db'):
    """
    将Excel文件中的数据导入到SQLite数据库

    参数:
    excel_file_path (str): Excel文件路径
    sqlite_db_path (str): SQLite数据库文件路径，默认为'flood_warning.db'
    """
    # 检查Excel文件是否存在
    if not os.path.exists(excel_file_path):
        raise FileNotFoundError(
            f"Excel文件不存在: {excel_file_path}")

    # 连接到SQLite数据库
    conn = sqlite3.connect(sqlite_db_path)

    try:
        # 读取Excel文件
        excel_file = pd.ExcelFile(excel_file_path)

        # 获取所有表名（sheet名）
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            # 读取sheet数据
            df = excel_file.parse(sheet_name)

            # 跳过空sheet
            if df.empty:
                print(f"跳过空sheet: {sheet_name}")
                continue

            # 替换表名中的非法字符
            table_name = sheet_name.replace(" ",
                                            "_").replace(
                "-", "_").replace("/", "_")

            # 确保表名符合SQLite命名规范
            if not table_name.isidentifier():
                table_name = f"table_{sheet_name.replace(' ', '_')}"

            # 将数据写入SQLite数据库
            print(
                f"正在导入表: {table_name} ({len(df)} 行)")

            # 使用chunksize分批导入，避免内存问题
            df.to_sql(
                name=table_name,
                con=conn,
                if_exists='replace',  # 如果表已存在，则替换
                index=False,  # 不导入索引列
                chunksize=1000  # 每批导入1000行
            )

            # 添加索引以提高查询性能
            if 'date' in df.columns:
                conn.execute(
                    f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name} (date)")

            print(f"成功导入表: {table_name}")

        print(f"所有数据已成功导入到 {sqlite_db_path}")

    except Exception as e:
        print(f"导入过程中发生错误: {e}")
        raise
    finally:
        # 关闭数据库连接
        conn.close()


if __name__ == "__main__":
    # 配置文件路径
    EXCEL_FILE_PATH = 'flood_warning.xlsx'  # 替换为您的Excel文件路径
    SQLITE_DB_PATH = 'flood_warning.db'  # 替换为您想要的SQLite数据库路径

    # 记录开始时间
    start_time = datetime.now()

    try:
        # 执行导入
        excel_to_sqlite(EXCEL_FILE_PATH, SQLITE_DB_PATH)

        # 计算导入时间
        elapsed_time = datetime.now() - start_time
        print(
            f"导入完成，耗时: {elapsed_time.total_seconds():.2f} 秒")

    except Exception as e:
        print(f"导入失败: {e}")
