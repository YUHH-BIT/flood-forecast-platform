from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import forecast, alert, data_query

app = FastAPI(
    title="🌊 洪水预报与预警平台 API",
    description="提供洪水预测、预警推送和历史数据查询服务。",
    version="1.0.0"
)

# CORS 中间件配置（前后端联调时使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议指定具体前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由模块
app.include_router(forecast.router, prefix="/api/forecast", tags=["洪水预报"])
app.include_router(alert.router, prefix="/api/alert", tags=["洪水预警"])
app.include_router(data_query.router, prefix="/api/query", tags=["数据查询"])

# 根路由
@app.get("/")
def read_root():
    return {"message": "欢迎访问洪水预报与预警平台 API 🚀"}

