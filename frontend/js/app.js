/**
 * HybridRAG 前端 - 主逻辑（路由、状态管理、通用工具）
 */

// 自动检测 API 地址：如果不是通过 FastAPI 服务访问（如 IDE 预览），则指向 localhost:8000
const API_BASE = window.location.port === '8000' ? '' : 'http://localhost:8000';

// ─── API 路径常量 ───
const API = {
    conversations: `${API_BASE}/api/conversations`,
    chatStream: `${API_BASE}/api/chat/stream`,
    chat: `${API_BASE}/api/chat`,
    systemModels: `${API_BASE}/api/system/models`,
    collections: `${API_BASE}/api/knowledge/collections`,
    documents: `${API_BASE}/api/knowledge/documents`,
    chunkConfig: `${API_BASE}/api/knowledge/chunk-config`,
    upload: `${API_BASE}/api/knowledge/upload`,
};

// ─── 全局状态 ───
const AppState = {
    currentPage: 'chat',
    currentThreadId: null,
    conversations: [],
    isStreaming: false,
    abortController: null,
    theme: localStorage.getItem('theme') || 'dark',
    sidebarCollapsed: localStorage.getItem('sidebarCollapsed') === 'true',
};

// ─── 初始化 ───
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initNavigation();
    initSidebar();
    initSidebarCollapse();
    loadConversations();
});

// ─── 主题管理 ───
function initTheme() {
    applyTheme(AppState.theme);
    document.getElementById('theme-toggle').addEventListener('click', () => {
        const newTheme = AppState.theme === 'dark' ? 'light' : 'dark';
        AppState.theme = newTheme;
        localStorage.setItem('theme', newTheme);
        applyTheme(newTheme);
    });
}

function applyTheme(theme) {
    // 始终显式设置 data-theme，避免 prefers-color-scheme 媒体查询干扰手动切换
    document.documentElement.setAttribute('data-theme', theme);
}

// ─── 页面导航 ───
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            const page = item.dataset.page;
            switchPage(page);
        });
    });
}

function switchPage(pageName) {
    AppState.currentPage = pageName;

    // 更新导航高亮
    document.querySelectorAll('.nav-item').forEach(item => {
        item.classList.toggle('active', item.dataset.page === pageName);
    });

    // 切换页面显示
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    const targetPage = document.getElementById(`page-${pageName}`);
    if (targetPage) {
        targetPage.classList.add('active');
    }

    // 页面切换回调
    if (pageName === 'knowledge' && typeof onKnowledgePageEnter === 'function') {
        onKnowledgePageEnter();
    }

    // 移动端关闭侧边栏
    closeSidebar();
}

// ─── 侧边栏（移动端） ───
function initSidebar() {
    const overlay = document.getElementById('sidebar-overlay');
    overlay.addEventListener('click', closeSidebar);
}

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.toggle('open');

    if (isMobile()) {
        const isOpen = sidebar.classList.contains('open');
        document.getElementById('btn-sidebar-expand').classList.toggle('hidden', isOpen);
    }
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    sidebar.classList.remove('open');

    if (isMobile()) {
        document.getElementById('btn-sidebar-expand').classList.remove('hidden');
    }
}

// ─── 侧边栏收缩/展开 ───
function isMobile() {
    return window.matchMedia('(max-width: 768px)').matches;
}

function initSidebarCollapse() {
    const btnToggle = document.getElementById('btn-sidebar-toggle');
    const btnExpand = document.getElementById('btn-sidebar-expand');

    btnToggle.addEventListener('click', () => {
        if (isMobile()) {
            closeSidebar();
        } else {
            collapseSidebar();
        }
    });

    btnExpand.addEventListener('click', () => {
        if (isMobile()) {
            toggleSidebar();
        } else {
            expandSidebar();
        }
    });

    // 恢复上次状态（仅 PC 端）
    if (AppState.sidebarCollapsed && !isMobile()) {
        collapseSidebar(false);
    }
}

function collapseSidebar(animate = true) {
    const sidebar = document.getElementById('sidebar');
    const btnExpand = document.getElementById('btn-sidebar-expand');

    if (!animate) {
        sidebar.style.transition = 'none';
        requestAnimationFrame(() => {
            sidebar.classList.add('collapsed');
            btnExpand.classList.remove('hidden');
            requestAnimationFrame(() => {
                sidebar.style.transition = '';
            });
        });
    } else {
        sidebar.classList.add('collapsed');
        btnExpand.classList.remove('hidden');
    }

    AppState.sidebarCollapsed = true;
    localStorage.setItem('sidebarCollapsed', 'true');
}

function expandSidebar() {
    const sidebar = document.getElementById('sidebar');
    const btnExpand = document.getElementById('btn-sidebar-expand');

    sidebar.classList.remove('collapsed');
    btnExpand.classList.add('hidden');

    AppState.sidebarCollapsed = false;
    localStorage.setItem('sidebarCollapsed', 'false');
}

// ─── 会话列表管理 ───
async function loadConversations() {
    try {
        const data = await apiFetch(API.conversations, {}, { retries: 1 });
        if (Array.isArray(data)) {
            AppState.conversations = data;
        }
        renderConversationList();
    } catch (error) {
        // 服务未启动时静默失败，不影响前端使用
        console.warn('加载会话列表失败（服务可能未启动）:', error.message);
    }
}

function renderConversationList() {
    const container = document.getElementById('conversation-items');
    container.innerHTML = '';

    if (!AppState.conversations.length) {
        const emptyDiv = document.createElement('div');
        emptyDiv.style.cssText = 'padding: 12px; color: var(--text-muted); font-size: 12px;';
        emptyDiv.textContent = '暂无历史会话';
        container.appendChild(emptyDiv);
        return;
    }

    AppState.conversations.forEach(conv => {
        const item = document.createElement('div');
        item.className = 'conversation-item' + (conv.id === AppState.currentThreadId ? ' active' : '');
        item.dataset.threadId = conv.id;

        const title = document.createElement('span');
        title.className = 'conversation-item-title';
        title.textContent = conv.title || '新对话';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'conversation-item-delete';
        deleteBtn.title = '删除';
        deleteBtn.appendChild(createSvgIcon('delete'));

        item.appendChild(title);
        item.appendChild(deleteBtn);

        // 点击加载会话
        item.addEventListener('click', (event) => {
            if (event.target.closest('.conversation-item-delete')) return;
            loadConversation(conv.id);
        });

        // 删除会话
        deleteBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            await deleteConversation(conv.id);
        });

        container.appendChild(item);
    });
}

async function loadConversation(threadId) {
    try {
        const conv = await apiFetch(`${API.conversations}/${threadId}`, {}, { retries: 1 });

        AppState.currentThreadId = threadId;
        renderConversationList();

        // 切换到对话页面并渲染历史消息
        switchPage('chat');
        if (typeof renderConversationHistory === 'function') {
            renderConversationHistory(conv.messages || []);
        }
    } catch (error) {
        showToast('加载会话失败', 'error');
    }
}

async function deleteConversation(threadId) {
    try {
        await apiFetch(`${API.conversations}/${threadId}`, { method: 'DELETE' }, { retries: 1 });
        if (AppState.currentThreadId === threadId) {
            AppState.currentThreadId = null;
            if (typeof resetChatView === 'function') {
                resetChatView();
            }
        }
        await loadConversations();
        showToast('会话已删除', 'success');
    } catch (error) {
        showToast('删除失败', 'error');
    }
}

function addConversationToList(threadId, title) {
    const existing = AppState.conversations.find(c => c.id === threadId);
    if (!existing) {
        AppState.conversations.unshift({ id: threadId, title: title, message_count: 1 });
    }
    AppState.currentThreadId = threadId;
    renderConversationList();
}

// ─── Toast 通知 ───
function showToast(message, type = 'info', duration = 3000) {
    let container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    // 限制最多同时显示 5 条 toast，超出时移除最早的
    while (container.children.length >= 5) {
        container.removeChild(container.firstChild);
    }

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        toast.style.transition = '0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ─── 统一 API 请求工具函数 ───

/**
 * 带 loading/error/retry 的统一 API 请求
 * @param {string} url - 请求地址
 * @param {object} options - fetch 选项
 * @param {object} config - 额外配置
 * @param {number} config.retries - 重试次数（默认 2）
 * @param {number} config.retryDelay - 重试间隔 ms（默认 1000）
 * @param {HTMLElement} config.loadingContainer - 显示 loading 的容器元素
 * @param {string} config.loadingHtml - 自定义 loading HTML
 * @param {boolean} config.showErrorToast - 是否显示错误 toast（默认 false）
 * @returns {Promise<any>} 解析后的 JSON 数据
 */
async function apiFetch(url, options = {}, config = {}) {
    const {
        retries = 2,
        retryDelay = 1000,
        loadingContainer = null,
        loadingHtml = null,
        showErrorToast = false,
    } = config;

    // 显示 loading 状态
    if (loadingContainer && loadingHtml) {
        loadingContainer.innerHTML = loadingHtml;
    }

    let lastError = null;

    for (let attempt = 0; attempt <= retries; attempt++) {
        try {
            const response = await fetch(url, options);
            if (!response.ok) {
                throw new Error(`请求失败: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            lastError = error;
            if (attempt < retries) {
                await new Promise(resolve => setTimeout(resolve, retryDelay));
            }
        }
    }

    if (showErrorToast) {
        showToast(`请求失败: ${lastError.message}`, 'error');
    }
    throw lastError;
}

/**
 * 创建 SVG 图标元素（避免重复内联 SVG）
 */
function createSvgIcon(type) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', '14');
    svg.setAttribute('height', '14');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('fill', 'none');
    svg.setAttribute('stroke', 'currentColor');
    svg.setAttribute('stroke-width', '2');

    if (type === 'delete') {
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', '3 6 5 6 21 6');
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('d', 'M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2');
        svg.appendChild(polyline);
        svg.appendChild(path);
    } else if (type === 'file') {
        const path1 = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path1.setAttribute('d', 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z');
        const polyline = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
        polyline.setAttribute('points', '14 2 14 8 20 8');
        svg.appendChild(path1);
        svg.appendChild(polyline);
    }

    return svg;
}

// ─── 工具函数 ───
const _escapeMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
function escapeHtml(text) {
    if (!text) return '';
    return String(text).replace(/[&<>"']/g, m => _escapeMap[m]);
}

function generateThreadId() {
    return 'thread_' + Date.now() + '_' + Math.random().toString(36).substring(2, 8);
}

// marked 配置只初始化一次
let _markedConfigured = false;
function _ensureMarkedConfig() {
    if (_markedConfigured || typeof marked === 'undefined') return;
    marked.setOptions({
        highlight: function(code, lang) {
            if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                return hljs.highlight(code, { language: lang }).value;
            }
            return code;
        },
        breaks: true,
        gfm: true,
    });
    _markedConfigured = true;
}

function renderMarkdown(text) {
    if (typeof marked === 'undefined') return escapeHtml(text);
    _ensureMarkedConfig();
    const rawHtml = marked.parse(text);
    // DOMPurify 消毒：防止 Markdown 注入 XSS
    if (typeof DOMPurify !== 'undefined') {
        return DOMPurify.sanitize(rawHtml, {
            ADD_TAGS: ['iframe'],
            ADD_ATTR: ['target', 'class', 'id'],
        });
    }
    return rawHtml;
}

// ─── 导出共享变量/函数到全局（供 IIFE 封装的 chat.js / knowledge.js 访问） ───
Object.assign(window, {
    AppState,
    API,
    apiFetch,
    showToast,
    escapeHtml,
    renderMarkdown,
    createSvgIcon,
    generateThreadId,
    addConversationToList,
    renderConversationList,
    switchPage,
    loadConversations,
    closeSidebar,
});
