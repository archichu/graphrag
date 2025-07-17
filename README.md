# GraphRAG 项目

一个完整的GraphRAG（图检索增强生成）实现，结合了知识图谱构建和检索增强生成技术。

## 🌟 特性

- **多数据集支持**: SQuAD, HotpotQA, Natural Questions等开源数据集
- **智能图构建**: 自动实体提取、关系构建和社区检测
- **高效检索**: 基于图结构的智能检索机制
- **可视化分析**: 图结构可视化和统计分析
- **灵活配置**: 支持本地和云端LLM模型
- **批量处理**: 支持批量查询和评估

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/archichu/graphrag-project.git
cd graphrag-project

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 安装spaCy模型
python -m spacy download en_core_web_sm

