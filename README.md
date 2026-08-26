# Writing Coach Agent

Writing Coach Agent 是一个面向英语议论文的本地写作教练。它按照 **Planner → Tool Router → Executor → Reflector** 的 Agent 循环工作：先规划需要调用的工具，通过 TF-IDF 与 all-MiniLM-L6-v2 双路检索读取原文证据，生成结构化评分与修订建议；工具失败时，Reflector 基于 Memory 决定重试或重新规划。

项目同时提供：

- Gradio 可视化界面；
- FastAPI 诊断接口；
- Hugging Face 本地模型后端；
- Working / Episodic Memory；
- TF-IDF 与 all-MiniLM-L6-v2 双检索及 RRF 融合；
- 工具重试、重规划和最大调用预算；
- Agent Trace 与 JSON 检查点，便于观察和恢复执行过程。

## Agent 架构

```text
Web UI / HTTP API
        │
        ▼
WritingCoachAgent.run(prompt, essay)       对外统一 interface
        │
        ├── Planner                         选择工具并形成计划
        ├── Tool Router / Executor
        │   ├── inspect_text                文本统计与分句
        │   ├── load_rubric                 读取评分量表
        │   ├── retrieve_evidence           TF-IDF + MiniLM 双检索
        │   └── locate_evidence             规则证据工具
        ├── Memory
        │   ├── working                     当前计划、工具结果和失败上下文
        │   └── episodic                    追加式决策与执行记录
        ├── Evaluator                       评分并生成修订建议
        ├── Reflector                       报告反思与工具失败分类
        ├── Re-planner                      携带 Memory 重新生成可执行计划
        └── Checkpoint Store                保存计划、工具结果、Trace 和报告
                │
                ▼
        JSONBackend seam
        └── HuggingFaceJSONBackend          本地开源模型
```

核心设计是让调用方只需要了解 `WritingCoachAgent.run()`。模型、工具集合和检查点目录都通过构造参数注入，因此可以独立替换和测试。

## 目录结构

```text
Writing_Coach_Agent/
├── pyproject.toml                 # Python 构建、安装与依赖元数据
├── requirements.txt
├── scripts/
│   └── run_app.py                 # 本地开发启动脚本
├── data/
│   ├── rubric.jsonl               # 语言与论证评分量表
│   ├── essays.jsonl               # 示例作文数据
│   └── keystrokes.jsonl           # 示例写作过程数据
├── tests/
│   └── test_agent.py              # 工具、完整循环与 HTML 安全测试
└── src/
    └── writing_coach_agent/        # 可安装的 Python 包
        ├── agent.py                # Agent 编排与执行守卫
        ├── main.py                 # ASGI 应用入口
        ├── models.py               # AgentRun 领域模型
        ├── memory.py               # Working / Episodic Memory
        ├── retrieval.py            # TF-IDF、MiniLM 与 RRF 双检索
        ├── reflection.py           # 工具失败决策与重规划触发
        ├── contracts.py            # 计划、报告、反思的输出契约
        ├── prompts.py              # 集中管理模型提示词
        ├── tools.py                # 确定性写作分析工具
        ├── checkpoints.py          # JSON 检查点适配器
        ├── rendering.py            # 原文证据安全高亮
        ├── config.py               # 环境变量与项目路径
        ├── backends/
        │   ├── base.py             # JSONBackend interface
        │   └── huggingface.py      # 本地模型适配器
        └── web/
            ├── api.py              # FastAPI 工厂与依赖装配
            ├── ui.py               # Gradio 页面
            └── presenters.py       # 报告到 UI 的格式转换
```

## 环境要求

- Python 3.10–3.12（推荐 3.11）
- 首次使用本地模型时需要网络下载模型，之后可从本地缓存加载
- 默认模型：`Qwen/Qwen2.5-0.5B-Instruct`

## 安装

项目采用标准 `src layout`。在现有 Conda 环境中执行可编辑安装：

```powershell
conda activate writing_coach_agent
python -m pip install --upgrade pip
python -m pip install -e .
```

若仅需按冻结前的依赖清单安装，也可以执行：

```powershell
python -m pip install -r requirements.txt
```

## 快速启动

### 本地模型模式

```powershell
python scripts/run_app.py
```

浏览器访问 `http://127.0.0.1:7860`。首次诊断会下载并加载 Qwen 与 all-MiniLM-L6-v2。模型下载或推理失败时直接返回错误，不会切换规则模型后端；工具执行失败可触发有预算限制的重规划。

![Writing Coach Agent 界面](pic/wcs.png)



### 使用 Uvicorn

```powershell
uvicorn writing_coach_agent.main:app --host 0.0.0.0 --port 7860
```

## HTTP API

健康检查：

```http
GET /health
```

作文诊断：

```http
POST /api/diagnose
Content-Type: application/json

{
  "prompt": "Should students receive cash rewards for good grades?",
  "essay": "Students should receive rewards because ..."
}
```

响应包含两个维度的分数、原文句子证据、优先修订项、修订计划、后端状态和本次 `run_id`。

## Python 调用

```python
from writing_coach_agent import HuggingFaceJSONBackend, WritingCoachAgent
from writing_coach_agent.config import PROJECT_ROOT

coach = WritingCoachAgent(
    backend=HuggingFaceJSONBackend("Qwen/Qwen2.5-0.5B-Instruct"),
    rubric_path=PROJECT_ROOT / "data" / "rubric.jsonl",
    checkpoint_dir=PROJECT_ROOT / "outputs" / "my_runs",
)

run = coach.run(
    "Should students read every day?",
    "Students should read every day because books build knowledge.",
)
print(run.report)
print(run.trace)
```

## 配置项

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `WRITING_COACH_MODEL` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face 模型 ID |
| `WRITING_COACH_RUBRIC_PATH` | `data/rubric.jsonl` | 自定义评分量表路径 |
| `WRITING_COACH_CHECKPOINT_DIR` | `outputs/product_runs` | 运行检查点目录 |
| `WRITING_COACH_HOST` | `0.0.0.0` | 服务监听地址 |
| `WRITING_COACH_PORT` | `7860` | 服务端口 |

## 扩展方式

### 替换模型后端

实现 `JSONBackend` 的两个要求即可：提供 `name` 属性，以及 `generate_json(system, user, schema)` 方法。随后将实例注入 `WritingCoachAgent`。

### 添加工具

工具接收 `(essay, rubric_path)`，返回可 JSON 序列化的字典。将工具字典通过 `WritingCoachAgent(..., tools=...)` 注入。Executor 会强制执行 `load_rubric` 和证据工具；生产配置优先使用 `retrieve_evidence`，其失败后重规划才可选择 `locate_evidence`。

### Memory 与重规划

每次运行都有独立的 `AgentMemory`：`working` 保存当前计划、工具结果和失败上下文，`episodic` 保存追加式执行历史。工具错误先由 `ExecutionReflector` 分类；临时错误在预算内重试，其他错误把失败工具和 Memory 回传 Planner 生成新计划。`max_tool_retries`、`max_replans` 与 `max_tool_calls` 分别限制重试、重规划和总工具调用次数。

### 双检索

`DualRetriever` 同时执行 `TfidfRetriever` 和 `MiniLMRetriever`，再使用 Reciprocal Rank Fusion 合并两个排序。输出保留每个候选的两路原始相似度和融合分数，便于 Trace 与证据溯源。两路都会真实执行，不是异常时的自动 fallback。

### 调整提示词或输出结构

- 在 `prompts.py` 中调整角色指令；
- 在 `contracts.py` 中维护结构化输出契约；
- 在 `agent.py` 的 `_validate_report()` 中维护必须由代码强制保证的业务不变量。

## 测试

测试注入仅供测试使用的脚本化后端，不会下载模型，也不会改变生产环境“失败即报错”的行为：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖：文本工具与证据编号、双路排序融合、Working/Episodic Memory、工具失败后的重规划、Planner/Executor/Reflector 完整循环、检查点保存，以及学生输入的 HTML 转义。

## 运行产物

每次执行会在 `outputs/product_runs/<run_id>.json` 保存：

- 输入题目与作文；
- Agent 计划；
- 工具结果；
- Working / Episodic Memory；
- 完整可观测 Trace；
- 最终结构化报告。

这些文件可能包含学生原文。部署到真实教学环境前，应设置访问权限、保留周期与脱敏策略，不要将真实学生数据提交到 Git。
