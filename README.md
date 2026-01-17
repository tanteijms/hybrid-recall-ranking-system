# HyRec - 智能内容推荐系统

一个完整的推荐系统实现，涵盖召回、粗排、精排全流程。

## 项目概述

本项目实现了工业级推荐系统的核心技术，包括：

- **多路召回**：热门召回、协同过滤(ItemCF)、内容召回、向量召回(双塔模型)
- **特征工程**：用户特征、物品特征、交叉特征
- **粗排阶段**：基于GBDT/XGBoost的轻量级模型
- **精排阶段**：DeepFM模型进行CTR预估
- **混合检索**：BM25 + 向量检索融合

## 技术栈

- **机器学习框架**：TensorFlow, XGBoost, scikit-learn
- **向量检索**：Faiss
- **数据处理**：pandas, numpy
- **评估指标**：Recall@K, NDCG@K, AUC

## 项目结构

```
hybrid-recall-ranking-system/
├── data/              # 数据目录
├── models/            # 模型文件
├── src/               # 源代码
│   ├── recall/        # 召回模块
│   ├── feature/       # 特征工程
│   ├── ranking/       # 排序模块
│   └── evaluation/    # 评估模块
├── notebooks/         # Jupyter notebooks
├── configs/           # 配置文件
└── doc/               # 文档

```

## 快速开始

### 环境要求

- Python 3.12
- TensorFlow 2.x
- 其他依赖见 `requirements.txt`

### 安装

```bash
pip install -r requirements.txt
```

### 运行

```bash
# 数据预处理
python src/data/preprocess.py

# 训练模型
python src/train.py

# 评估
python src/evaluate.py
```

## 开发计划

详见 [doc/计划书.md](doc/计划书.md)

## License

MIT License
