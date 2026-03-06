# HybridRAG 前端

混合检索增强生成系统的 Web 前端，基于原生 HTML / CSS / JavaScript 构建，无框架依赖，支持暗色/亮色主题切换、流式对话、多知识库管理。

---

## 目录结构

```
frontend/
├── index.html              # 单页应用入口（SPA）
├── README.md               # 本文档
├── css/
│   ├── base.css            # CSS 变量系统、全局重置、主题、无障碍、打印样式
│   ├── layout.css          # 应用布局、侧边栏、导航、页面容器、移动端适配
│   ├── chat.css            # 对话页面样式（消息气泡、推理过程、输入框）
│   ├── knowledge.css       # 知识库管理页面样式（卡片、表格、上传、骨架屏）
│   └── components.css      # 通用组件（按钮、Modal 弹窗、Toast 通知、Spinner）
└── js/
    ├── app.js              # 核心逻辑（路由、状态管理、主题、API 工具、通用函数）
    ├── chat.js             # 对话页面逻辑（流式输出、Markdown 渲染、推理过程）
    └── knowledge.js        # 知识库管理逻辑（模型状态、文档管理、上传、分块配置）
```

---

## 页面模块

### 1. 对话页面（Chat）

主要功能：与 RAG 系统进行智能问答对话。

| 功能 | 说明 |
|------|------|
| **流式输出** | 通过 SSE（Server-Sent Events）实现逐 token 流式渲染，支持中途停止生成 |
| **Markdown 渲染** | 使用 `marked.js` 渲染 Markdown，`highlight.js` 代码高亮，`DOMPurify` XSS 消毒 |
| **推理过程展示** | 支持问题分解（decomposition）和子问题推理（sub_answer）的折叠展示 |
| **联网搜索** | Pill 按钮切换联网搜索开关，状态通过隐藏 checkbox 保持 |
| **非流式降级** | 流式接口不可用时自动降级到普通 HTTP 请求 |
| **会话管理** | 侧边栏显示历史会话列表，支持加载、删除、新建对话 |
| **欢迎页** | 无对话时显示居中欢迎页，输入框居中；对话中输入框移至底部 |

**相关文件**: `js/chat.js`、`css/chat.css`

---

### 2. 知识库管理页面（Knowledge）

主要功能：管理向量知识库、上传文档、配置分块参数。

#### 2.1 仪表盘概览

| 功能 | 说明 |
|------|------|
| **模型资源卡片** | 显示 Embedding / Reranker / LLM 模型的在线状态、类型、维度等信息 |
| **知识库统计** | 展示知识库数量、总记录数、已上传文件数、Milvus 连接状态 |
| **骨架屏加载** | 数据加载中显示 shimmer 动画骨架屏，提升感知性能 |
| **刷新按钮** | 带防重复点击锁和旋转动画的刷新按钮 |

#### 2.2 数据导入（Tab: 数据导入）

| 功能 | 说明 |
|------|------|
| **拖拽上传** | 支持拖拽文件到上传区域，或点击选择文件 |
| **文件校验** | 格式校验（PDF / Word / TXT / Markdown）、单文件 ≤50MB、总大小 ≤200MB |
| **目标知识库选择** | 弹窗选择已有知识库或输入新名称创建 |
| **上传进度** | 实时进度条 + 文件标签列表 |
| **分块配置** | 可调节 Chunk Size（100-2000）和 Chunk Overlap（0-500），提供 3 种预设方案 |

#### 2.3 知识库列表（Tab: 知识库列表）

| 功能 | 说明 |
|------|------|
| **卡片网格** | 响应式网格布局展示知识库卡片，悬停动画效果 |
| **搜索过滤** | 带 200ms 防抖的实时搜索过滤 |
| **删除知识库** | 悬停显示删除按钮，二次确认后删除 |

#### 2.4 文档管理（Tab: 文档管理）

| 功能 | 说明 |
|------|------|
| **文档表格** | 展示文件名、大小、所属知识库、上传时间、状态 |
| **知识库筛选** | 下拉选择按知识库过滤文档 |
| **全选/批量删除** | 支持全选、部分选中（indeterminate 状态）、批量删除 |
| **单个删除** | 每行操作按钮支持单个文件删除 |

**相关文件**: `js/knowledge.js`、`css/knowledge.css`

---

## 核心架构

### 状态管理

使用全局 `AppState` 对象管理应用状态：

```javascript
const AppState = {
    currentPage: 'chat',        // 当前页面
    currentThreadId: null,       // 当前会话 ID
    conversations: [],           // 会话列表
    isStreaming: false,           // 是否正在流式输出
    abortController: null,       // 流式请求中断控制器
    theme: 'dark',               // 当前主题
    sidebarCollapsed: false,     // 侧边栏是否收缩
};
```

### API 通信

统一的 `apiFetch` 工具函数，提供：
- **自动重试**: 可配置重试次数和间隔
- **错误处理**: 可选 Toast 错误提示
- **Loading 状态**: 可选 loading 容器

API 路径集中定义在 `API` 常量对象中：

```javascript
const API = {
    conversations: '/api/conversations',
    chatStream:    '/api/chat/stream',
    chat:          '/api/chat',
    systemModels:  '/api/system/models',
    collections:   '/api/knowledge/collections',
    documents:     '/api/knowledge/documents',
    chunkConfig:   '/api/knowledge/chunk-config',
    upload:        '/api/knowledge/upload',
};
```

### 模块封装

- `app.js` — 全局作用域，通过 `Object.assign(window, {...})` 导出共享变量
- `chat.js` — IIFE 封装，仅导出 `renderConversationHistory`、`resetChatView`
- `knowledge.js` — IIFE 封装，仅导出 `onKnowledgePageEnter`

---

## CSS 架构

### 设计系统

基于 CSS 变量的完整设计系统，定义在 `base.css` 中：

| 类别 | 示例变量 |
|------|---------|
| **间距** | `--space-xs` (4px) ~ `--space-4xl` (48px) |
| **字体** | `--font-sm` (12px) ~ `--font-3xl` (32px)、`--font-mono` |
| **圆角** | `--radius-xs` (4px) ~ `--radius-pill` (9999px) |
| **颜色** | `--bg-primary`、`--text-primary`、`--accent`、`--danger` 等 |
| **阴影** | `--shadow`、`--shadow-sm` |
| **过渡** | `--transition` |

### 主题系统

| 模式 | 触发条件 |
|------|---------|
| **暗色主题** | `data-theme="dark"`（默认） |
| **亮色主题** | `data-theme="light"` |
| **自动跟随** | 无 `data-theme` 属性时，由 `prefers-color-scheme` 媒体查询决定 |

用户手动切换后，主题偏好保存在 `localStorage('theme')` 中。

### 响应式适配

- **桌面端**: 侧边栏固定显示，支持收缩/展开
- **移动端** (`≤768px`): 侧边栏改为抽屉式，点击遮罩关闭
- **打印**: `@media print` 隐藏侧边栏、输入框等非内容元素

### 无障碍

- `@media (prefers-reduced-motion: reduce)` 全局禁用动画和过渡

---

## 安全措施

| 措施 | 说明 |
|------|------|
| **CSP** | `Content-Security-Policy` meta 标签限制脚本/样式/连接来源 |
| **XSS 防护** | 用户输入使用 `textContent` / `escapeHtml`，Markdown 输出经 `DOMPurify` 消毒 |
| **DOM API** | 动态内容全部使用 `createElement` + `textContent`，避免 `innerHTML` 注入 |
| **上传校验** | 前端校验文件格式、单文件大小（≤50MB）、总大小（≤200MB） |

---

## 外部依赖

| 库 | 版本 | 用途 | 加载方式 |
|----|------|------|---------|
| [marked.js](https://github.com/markedjs/marked) | latest | Markdown 解析 | CDN `defer` |
| [highlight.js](https://github.com/highlightjs/highlight.js) | 11.x | 代码语法高亮 | CDN `defer` |
| [DOMPurify](https://github.com/cure53/DOMPurify) | 3.0.6 | HTML 消毒防 XSS | CDN `defer` |

所有 CDN 脚本均使用 `defer` 属性异步加载，不阻塞页面渲染。

---

## 开发说明

### 本地运行

前端通过 FastAPI 静态文件服务提供，默认端口 `8000`：

```bash
# 启动后端服务（同时提供前端静态文件）
python main.py
```

访问 `http://localhost:8000` 即可。

如果通过 IDE 预览（非 8000 端口），前端会自动将 API 请求代理到 `http://localhost:8000`。

### 编码规范

- **JavaScript**: 使用 `const` / `let`，禁止 `var`；IIFE 封装模块作用域
- **CSS**: 所有颜色、间距、字体使用 CSS 变量；禁止硬编码魔法值
- **HTML**: 语义化标签；动态内容使用 DOM API 创建，禁止 `innerHTML` 拼接用户数据
- **命名**: CSS 使用 BEM 风格（`block-element`）；JS 使用 camelCase

### 性能优化

- `base.css` 使用 `fetchpriority="high"` 优先加载
- `marked.setOptions` 只初始化一次（`_ensureMarkedConfig`）
- 流式渲染使用 120ms 节流，避免频繁 DOM 操作
- Toast 通知限制最多同时显示 5 条
- 知识库搜索使用 200ms 防抖
- 骨架屏 shimmer 动画提升感知加载速度
