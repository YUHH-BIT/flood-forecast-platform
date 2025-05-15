flood-forecast-platform/
├── streamlit_app/        # Streamlit 应用核心代码
│   ├── app.py            # 主界面路由与交互逻辑
│   ├── forecast.py       # 洪水预报核心算法
│   ├── data_query.py     # 数据库查询与数据处理
│   ├── alert.py          # 预警规则与通知逻辑
│   └── utils/            # 公用工具函数
│       └── __init__.py    # 初始化文件（可导入常用库或定义全局变量）
├── data/                 # 数据存储与管理
│   ├── processed/        # 处理后的数据（数据库）
│   │   └── flood_warning.db  # SQLite 数据库（存储清洗后的数据）
│   └── raw/              # 原始数据集
│       └── flood_warning.xlsx  # 原始输入数据（如历史水位、降雨数据）
├── models/               # 模型相关文件
│   ├── best_lstm_model.pth  # 训练好的 LSTM 模型权重
│   └── best_params.json  # 模型超参数与训练元数据
├── notebooks/            # 模型训练与数据分析记录
│   └── model_training.ipynb  # 包含数据预处理、模型训练、评估的完整流程
└── README.md             # 项目说明文档（含环境配置、运行指南、功能介绍）
