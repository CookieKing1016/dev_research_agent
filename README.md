# DevResearch Agent

Multi-agent research assistant for engineering tasks. It turns a topic, GitHub
repository, issue, or job description into a traceable technical report.

## MVP Features

- FastAPI backend with task APIs and WebSocket trace streaming
- LangGraph StateGraph flow: plan, retrieve, analyze, draft, critique, replan, evaluate
- Hybrid retrieval demo with BM25, dense-style token scoring, and RRF fusion
- GitHub repository reader for metadata, README, and directory structure
- Optional OpenAI-compatible LLM mode for Planner, Writer, and Critic
- SQLite task, trace, and evaluation persistence
- React/Vite trace UI
- 20 seed Agentic Eval cases

The project runs without an LLM key by using deterministic fallback agents. Set
an OpenAI-compatible endpoint to enable real LLM planning, report writing, and
critique.

## LLM Configuration

Create `backend/.env` or set environment variables before starting the backend:

```powershell
$env:LLM_API_KEY="your-api-key"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="gpt-4o-mini"
```

Any OpenAI-compatible `/chat/completions` provider can be used. For example,
DeepSeek, DashScope compatible mode, SiliconFlow, local proxies, or OpenAI.

## Quick Start

Backend:

```powershell
cd F:\hello_agents\dev_research_agent\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd F:\hello_agents\dev_research_agent\frontend
npm install
npm run dev
```

Eval:

```powershell
cd F:\hello_agents\dev_research_agent\backend
python -m app.eval.run_cases --cases ..\data\eval_cases.json --out ..\reports\eval_scores.csv
```

## Resume Version

多 Agent 深度研究与研发报告生成系统

- 基于 LangGraph 风格状态流构建 ReAct + Function Calling 执行框架，实现任务规划、工具调用、报告生成、Critic 审核与重规划闭环。
- 设计 Planner、Search、Code、Writer、Critic、Evaluator 多 Agent 协作流程，支持 GitHub 仓库解析、技术资料检索和研发报告生成。
- 构建 BM25 + Dense + RRF 混合检索链路，接入 SQLite 管理知识片段与执行 Trace，提升长文档与代码资料召回稳定性。
- 构建 20 条 Agentic Eval 测试集，从工具调用成功率、引用支撑率、报告完整度和事实一致性等维度评估系统效果。
