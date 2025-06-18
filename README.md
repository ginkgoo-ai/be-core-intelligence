# 签证自动填表工作流系统

基于AI的英国签证申请12步工作流自动化系统，集成文档解析、智能表单分析和自动填写功能。

## 🚀 核心功能

### 工作流管理
- **12步签证申请流程**: 完整覆盖英国签证申请的所有步骤
- **状态管理**: 实时跟踪每个步骤的执行状态
- **流程控制**: 支持暂停、恢复、跳转等操作
- **自动保存**: 防止数据丢失的自动保存机制

### AI智能分析 (LangGraph集成)
- **表单解析**: 使用LangGraph自动分析HTML表单结构
- **字段识别**: 智能识别表单字段类型和要求
- **问题生成**: AI自动为表单字段生成相关问题
- **智能回答**: 基于用户档案AI推理生成答案
- **动作生成**: 生成可执行的表单填写动作

### 文档处理
- **多格式支持**: 支持护照、身份证、银行对账单等文档
- **信息提取**: 自动提取关键信息构建用户档案
- **数据验证**: 验证提取数据的准确性和完整性

## 📋 12步工作流程

1. **申请人设置** (Applicant & Application Setup)
2. **个人详情** (Personal Details)
3. **联系地址** (Contact & Address)
4. **家庭详情** (Family Details)
5. **旅行历史** (Travel History)
6. **移民历史** (Immigration History)
7. **就业担保** (Employment & Sponsor)
8. **英语能力** (English Language)
9. **财务要求** (Financial Requirements)
10. **安全品格** (Security & Character)
11. **附加信息** (Additional Information)
12. **申请声明** (Application & Declaration)

## 🏗️ 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Layer     │    │  Service Layer  │    │  Data Layer     │
│  (FastAPI)      │◄──►│  (Business)     │◄──►│  (PostgreSQL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Pydantic      │    │   LangGraph     │    │   Database      │
│   Models        │    │   AI Workflow   │    │   Entities      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### 核心组件

- **数据模型层**: Pydantic模型定义API接口
- **数据库实体层**: SQLAlchemy ORM模型
- **仓储层**: 数据访问抽象
- **业务服务层**: 工作流和步骤管理逻辑
- **API路由层**: FastAPI端点定义
- **LangGraph处理器**: AI驱动的表单分析引擎

### LangGraph AI工作流节点
- **HTML解析节点**: 解析表单结构和字段
- **字段检测节点**: 检测和分析表单字段
- **问题生成节点**: 为表单字段生成相关问题
- **档案检索节点**: 检索用户档案数据
- **AI回答节点**: 基于档案AI推理生成答案
- **操作生成节点**: 将答案转换为表单操作
- **结果保存节点**: 保存分析结果到数据库

## 🛠️ 技术栈

- **Web框架**: FastAPI
- **数据库**: SQLAlchemy + PostgreSQL
- **AI框架**: LangChain + LangGraph
- **文档解析**: BeautifulSoup + PyPDF2
- **数据验证**: Pydantic
- **异步处理**: asyncio

## 📦 安装和运行

### 环境要求
- Python 3.8+
- PostgreSQL 12+
- pip

### 数据库配置

#### 1. 安装PostgreSQL
确保PostgreSQL服务正在运行，默认配置：
- 主机: 127.0.0.1
- 端口: 5433
- 数据库: postgres
- 用户: postgres
- 密码: postgres

#### 2. 快速配置环境变量
```bash
# 运行环境配置脚本
python scripts/setup_env.py
```

#### 3. 手动配置环境变量
创建 `.env` 文件：
```bash
# Database Configuration
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5433
POSTGRES_DB=postgres
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# Constructed Database URL
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5433/postgres

# AI Model Configuration
MODEL_BASE_URL=https://api.openai.com/v1
MODEL_API_KEY=your_openai_api_key_here
MODEL_WORKFLOW_NAME=gpt-3.5-turbo

# Debug Settings
SQL_DEBUG=false
LOG_LEVEL=INFO

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
```

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd be-core-intelligence
```

2. **安装依赖**
```bash
pip install -r requirements.txt
```

3. **测试数据库连接**
```bash
python scripts/test_database.py
```

4. **初始化数据库**
```bash
python scripts/init_database.py
```

5. **启动服务**
```bash
python -m src.main
```

服务将在 `http://localhost:8000` 启动

### API文档
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🔧 API使用示例

### 创建工作流
```bash
curl -X POST "http://localhost:8000/workflows/" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
    "workflow_definition_id": null,
    "initial_data": {},
    "form_url": "https://example.com/visa-form"
  }'
```

### 获取工作流状态
```bash
curl -X GET "http://localhost:8000/workflows/{workflow_id}/"
```

### 提交步骤数据
```bash
curl -X PATCH "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/" \
  -H "Content-Type: application/json" \
  -d '{
    "step_key": "personal_details",
    "given_name": "John",
    "family_name": "Doe",
    "date_of_birth": "1990-01-01"
  }'
```

### 处理表单 (LangGraph AI分析)
```bash
curl -X POST "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/process-form" \
  -H "Content-Type: application/json" \
  -d '{
    "form_html": "<html><form>...</form></html>",
    "profile_data": {
      "personalDetails": {
        "givenName": "张",
        "familyName": "三",
        "dateOfBirth": "1990-05-15"
      },
      "contactInformation": {
        "emails": [{"emailAddress": "zhangsan@example.com"}]
      }
    }
  }'
```

### 获取AI分析数据
```bash
# 获取完整分析数据
curl "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/analysis"

# 获取AI问题
curl "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/questions"

# 获取AI答案
curl "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/answers"

# 获取表单操作
curl "http://localhost:8000/workflows/{workflow_id}/steps/personal_details/actions"
```

## 📊 数据库设计

### 核心表结构
- **workflow_instances**: 工作流实例
- **step_instances**: 步骤实例 (包含AI分析结果)
- **workflow_definitions**: 工作流定义

### AI分析数据存储
所有LangGraph AI分析结果存储在 `step_instances.data` 字段中：
```json
{
  "html_analysis": {
    "form_fields": [...],
    "field_types": {...}
  },
  "ai_processing": {
    "questions": [...],
    "answers": [...]
  },
  "form_actions": [...]
}
```

## 🔄 工作流状态

### 工作流状态
- `PENDING`: 待开始
- `IN_PROGRESS`: 进行中
- `PAUSED`: 已暂停
- `COMPLETED`: 已完成
- `FAILED`: 失败

### 步骤状态
- `PENDING`: 待开始
- `ACTIVE`: 活跃中
- `COMPLETED_SUCCESS`: 成功完成
- `COMPLETED_ERROR`: 完成但有错误
- `SKIPPED`: 已跳过

## 🧪 测试

```bash
# 测试数据库连接
python scripts/test_database.py

# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_workflow_service.py

# 生成覆盖率报告
pytest --cov=src
```

## 🚀 快速开始演示

```bash
# 运行完整工作流演示
python examples/workflow_demo.py
```

## 🔧 故障排除

### 数据库连接问题
1. 确保PostgreSQL服务正在运行
2. 检查端口5433是否被占用
3. 验证用户名和密码是否正确
4. 运行数据库测试脚本: `python scripts/test_database.py`

### AI模型配置问题
1. 确保OpenAI API密钥已正确配置
2. 检查网络连接是否正常
3. 验证模型名称是否正确

### 依赖安装问题
```bash
# 升级pip
pip install --upgrade pip

# 重新安装依赖
pip install -r requirements.txt --force-reinstall
```

## 📝 开发指南

### 代码规范
- 使用 Black 进行代码格式化
- 使用 Flake8 进行代码检查
- 遵循 PEP 8 编码规范

### 提交规范
- feat: 新功能
- fix: 修复bug
- docs: 文档更新
- style: 代码格式调整
- refactor: 代码重构
- test: 测试相关

## 🤝 贡献

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 支持

如有问题或建议，请：
- 创建 Issue
- 发送邮件至 support@example.com
- 查看文档 `/docs`

---

**注意**: 本系统仅用于辅助签证申请，最终提交前请仔细核对所有信息的准确性。

## 核心特性

- **完整的12步签证申请工作流**：涵盖从申请人设置到最终声明的全过程
- **AI驱动的HTML表单分析**：使用LangGraph工作流自动解析和理解表单结构
- **智能表单自动填写**：基于用户档案数据和AI推理生成精确的表单操作
- **工作流状态管理**：支持暂停、恢复、错误处理等完整的状态管理
- **数据持久化存储**：所有分析结果、AI问答和操作记录都存储在数据库中
- **RESTful API设计**：提供完整的API接口支持前端集成
- **模块化架构**：清晰的分层设计，易于扩展和维护

## LangGraph AI工作流

### 工作流节点定义

系统使用LangGraph定义了以下AI处理节点：

1. **HTML分析节点** (`analyze_html`)
   - 解析HTML表单结构
   - 识别表单字段和属性
   - 提取字段约束和验证规则

2. **问题生成节点** (`generate_questions`)
   - 基于表单字段生成相关问题
   - 考虑字段类型和约束条件
   - 生成用户友好的问题描述

3. **答案生成节点** (`generate_answers`)
   - 基于用户档案数据回答问题
   - 使用AI推理生成合适的答案
   - 提供置信度和推理过程

4. **操作生成节点** (`generate_actions`)
   - 将答案转换为具体的表单操作
   - 生成CSS选择器和操作类型
   - 确保操作的正确性和顺序

### 数据存储结构

所有AI处理结果都存储在`StepInstance.data`字段中，结构如下：

```json
{
  "html_analysis": {
    "fields": [...],
    "field_count": 10,
    "form_structure": {...},
    "analysis_timestamp": "2024-01-01T12:00:00Z"
  },
  "ai_processing": {
    "questions": [
      {
        "field_name": "firstName",
        "question": "What is your first name?",
        "field_type": "text",
        "required": true,
        "constraints": {...}
      }
    ],
    "answers": [
      {
        "field_name": "firstName",
        "answer": "张",
        "confidence": 95,
        "reasoning": "Based on user profile personal details",
        "source": "user_profile.personalDetails.givenName"
      }
    ],
    "question_count": 10,
    "answer_count": 10,
    "processing_timestamp": "2024-01-01T12:00:01Z"
  },
  "form_actions": [
    {
      "selector": "#firstName",
      "action_type": "input",
      "value": "张",
      "order": 1,
      "field_name": "firstName"
    }
  ]
}
```

## 运行演示

项目包含一个完整的演示脚本，展示整个AI工作流：

```bash
# 启动API服务器
python -m uvicorn src.main:app --reload

# 在另一个终端运行演示
python examples/workflow_demo.py
```

演示脚本将：
1. 创建用户和工作流
2. 处理示例HTML表单
3. 展示AI分析结果
4. 显示生成的问题、答案和操作
5. 保存结果到`demo_result.json`