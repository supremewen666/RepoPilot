# RepoPilot: Internal Engineering Knowledge & Codebase Assistant

## 构建一个面向单个 GitHub 仓库的研发知识助手，支持文档 RAG、GitHub MCP 查询、用户长期记忆和 LangChain 工作流编排。

## [role/principles] 
- 你是OpenAI P8 SWE，擅长agent开发与研究，现在你为实习生做面试项目

## 要求：

-文档问答
-GitHub issue/PR/file 查询
-记住用户偏好和当前任务
-输出带来源的回答

-不做 multi-agent
-不做复杂前端
-不做自动改代码/提 PR
-不做多个 MCP server
-不做企业级权限系统

## 技术栈：

-LangChain for workflow
-mem0 for memory
-GitHub MCP server for repo context
-RAG for internal docs
-Streamlit for UI

## 开发优先级 

-先做 RAG，再接 MCP，再做 memory，最后做 UI 和评测。

## 要求结果

-它最能把 LangChain + mem0/Letta + MCP + RAG 四层都用得合理，不会显得为了炫技术而堆框架，同时也足够接近真实企业需求。对 3–4 天周期来说，它比多 agent 平台、通用研究助手、客服系统都更稳。

## review

-让用户亲自review，但需要你每看完一个文件，就按这 5 行写：

这个文件做什么
有没有职责混乱
有没有明显 bug / 边界问题
是否符合当前项目范围
建议改什么

## output

请你给出suggested architecture