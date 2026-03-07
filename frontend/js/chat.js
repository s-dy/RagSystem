/**
 * HybridRAG 前端 - 对话页面逻辑
 * 支持：流式输出、联网搜索 pill 按钮、推理过程展示、Markdown 渲染
 */
(function () {
'use strict';

// ─── DOM 引用 ───
const chatInput = document.getElementById('chat-input');
const btnSend = document.getElementById('btn-send');
const btnStop = document.getElementById('btn-stop');
const btnNewChat = document.getElementById('btn-new-chat');
const chatMessages = document.getElementById('chat-messages');
const chatWelcome = document.getElementById('chat-welcome');
const toggleWebSearch = document.getElementById('toggle-web-search');
const pillWebSearch = document.getElementById('pill-web-search');

// ─── 初始化 ───
document.addEventListener('DOMContentLoaded', () => {
    initChatEvents();
});

function initChatEvents() {
    // 发送按钮
    btnSend.addEventListener('click', sendMessage);

    // 停止按钮
    btnStop.addEventListener('click', stopStreaming);

    // 新建对话
    btnNewChat.addEventListener('click', startNewChat);

    // 联网搜索 pill 按钮 — 点击切换高亮
    pillWebSearch.addEventListener('click', () => {
        const isActive = pillWebSearch.classList.toggle('active');
        toggleWebSearch.checked = isActive;
    });

    // 输入框事件
    chatInput.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    });

    // 自动调整输入框高度 + 控制发送按钮状态
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
        btnSend.disabled = !chatInput.value.trim();
    });
}

// ─── 发送消息 ───
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || AppState.isStreaming) return;

    // 如果没有当前会话，创建新的
    if (!AppState.currentThreadId) {
        AppState.currentThreadId = generateThreadId();
    }

    // 显示消息区域，隐藏欢迎页
    showChatMessages();

    // 渲染用户消息
    appendMessage('user', message);

    // 清空输入框
    chatInput.value = '';
    chatInput.style.height = 'auto';
    btnSend.disabled = true;

    // 添加到会话列表
    addConversationToList(AppState.currentThreadId, message.substring(0, 30));

    // 开始流式请求
    await streamChat(message);
}

// ─── 流式对话 ───
async function streamChat(message) {
    AppState.isStreaming = true;
    AppState.abortController = new AbortController();
    updateSendButtonState();

    // 创建助手消息占位
    const assistantMessageElement = appendMessage('assistant', '');
    const bodyElement = assistantMessageElement.querySelector('.message-body');

    // 构建两个区域：推理过程区域（顶部）+ 答案内容区域（底部）
    // 用 DOM API 创建推理区域和答案区域
    bodyElement.innerHTML = '';
    const reasoningArea = document.createElement('div');
    reasoningArea.className = 'reasoning-area';
    const answerArea = document.createElement('div');
    answerArea.className = 'answer-area';
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.appendChild(document.createElement('span'));
    typingIndicator.appendChild(document.createElement('span'));
    typingIndicator.appendChild(document.createElement('span'));
    answerArea.appendChild(typingIndicator);
    bodyElement.appendChild(reasoningArea);
    bodyElement.appendChild(answerArea);
    // reasoningArea 和 answerArea 已在上方通过 DOM API 创建

    let fullContent = '';
    let reasoningSteps = [];
    let hasReceivedToken = false;
    let receivedFinalAnswer = false;
    let renderTimer = null;
    let pendingRender = false;

    // 节流渲染：只更新答案区域，不影响推理过程区域
    function throttledRender() {
        if (renderTimer) return;
        pendingRender = true;
        renderTimer = setTimeout(() => {
            if (pendingRender) {
                answerArea.innerHTML = renderMarkdown(fullContent);
                scrollToBottom();
                pendingRender = false;
            }
            renderTimer = null;
        }, 120);
    }

    try {
        const response = await fetch(API.chatStream, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                thread_id: AppState.currentThreadId,
                user_id: 'default',
                enable_web_search: toggleWebSearch.checked,
            }),
            signal: AppState.abortController.signal,
        });

        if (!response.ok) {
            // 流式接口不可用，降级到非流式
            bodyElement.innerHTML = '';
            await fallbackChatWithElement(message, bodyElement);
            return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const jsonStr = line.substring(6).trim();
                if (!jsonStr) continue;

                let event;
                try {
                    event = JSON.parse(jsonStr);
                } catch {
                    continue;
                }

                switch (event.type) {
                    case 'token':
                        if (!hasReceivedToken) {
                            answerArea.innerHTML = '';
                            hasReceivedToken = true;
                        }
                        fullContent += event.content;
                        throttledRender();
                        break;

                    case 'retrieval_progress':
                        // 检索进度展示
                        updateRetrievalProgress(reasoningArea, event);
                        scrollToBottom();
                        break;

                    case 'decomposition':
                        if (event.sub_questions && event.sub_questions.length > 0) {
                            // 立即渲染子问题分解到推理过程区域
                            const decompositionHtml = renderDecomposition(event.sub_questions);
                            reasoningArea.insertAdjacentHTML('beforeend', decompositionHtml);
                            bindReasoningToggle(reasoningArea);
                            // 显示答案区域的等待状态
                            answerArea.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
                            scrollToBottom();
                        }
                        break;

                    case 'sub_answer':
                        reasoningSteps.push({
                            question: event.sub_question,
                            answer: event.answer,
                        });
                        // 立即更新推理过程折叠块
                        updateReasoningDisplay(reasoningArea, reasoningSteps);
                        scrollToBottom();
                        break;

                    case 'final_answer':
                        // 用 final_answer 的完整内容替换（清除可能混入的子问题 token）
                        fullContent = event.answer;
                        hasReceivedToken = true;
                        receivedFinalAnswer = true;
                        // 立即渲染最终答案
                        answerArea.innerHTML = renderMarkdown(fullContent);
                        scrollToBottom();
                        break;

                    case 'error':
                        answerArea.innerHTML = `<div style="color: var(--danger);">❌ ${escapeHtml(event.content)}</div>`;
                        break;

                    case 'heartbeat':
                        break;

                    case 'done':
                        break;
                }
            }
        }

        // 最终渲染：确保答案区域内容完整
        if (hasReceivedToken && fullContent) {
            answerArea.innerHTML = renderMarkdown(fullContent);
        }

        // 如果完全没有收到内容
        if (!hasReceivedToken && !fullContent) {
            answerArea.innerHTML = '<span style="color: var(--text-muted);">未收到回复内容</span>';
        }

    } catch (error) {
        if (error.name === 'AbortError') {
            if (!fullContent) {
                bodyElement.innerHTML = '<span style="color: var(--text-muted);">已停止生成</span>';
            }
        } else {
            // 网络错误等，降级到非流式
            bodyElement.innerHTML = '';
            try {
                await fallbackChatWithElement(message, bodyElement);
                return;
            } catch (fallbackError) {
                bodyElement.innerHTML = `<div style="color: var(--danger);">❌ 请求失败: ${escapeHtml(fallbackError.message)}</div>`;
                showToast('请求失败，请检查服务是否启动', 'error');
            }
        }
    } finally {
        // 确保最后一次渲染完成
        if (renderTimer) {
            clearTimeout(renderTimer);
            renderTimer = null;
        }
        if (fullContent && hasReceivedToken) {
            answerArea.innerHTML = renderMarkdown(fullContent);
        }
        AppState.isStreaming = false;
        AppState.abortController = null;
        updateSendButtonState();
        scrollToBottom();
    }
}

// ─── 非流式降级（复用已有的消息元素） ───
async function fallbackChatWithElement(message, bodyElement) {
    bodyElement.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';

    const response = await fetch(API.chat, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: message,
            thread_id: AppState.currentThreadId,
            user_id: 'default',
            enable_web_search: toggleWebSearch.checked,
        }),
    });

    if (!response.ok) {
        throw new Error(`请求失败: ${response.status}`);
    }

    const result = await response.json();
    const answer = result.answer || '未收到回复内容';
    bodyElement.innerHTML = renderMarkdown(answer);
    scrollToBottom();
}

// ─── 停止生成 ───
function stopStreaming() {
    if (AppState.abortController) {
        AppState.abortController.abort();
    }
}

// ─── 新建对话 ───
function startNewChat() {
    AppState.currentThreadId = null;
    resetChatView();
    renderConversationList();
    switchPage('chat');
}

// ─── 渲染消息 — OpenAI 风格 ───
function appendMessage(role, content) {
    const messageDiv = document.createElement('div');

    if (role === 'user') {
        messageDiv.className = 'message user-message';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = content;
        messageDiv.appendChild(bubble);
    } else {
        messageDiv.className = 'message assistant-message';
        const body = document.createElement('div');
        body.className = 'message-body';
        if (content) {
            body.innerHTML = renderMarkdown(content);
        }
        messageDiv.appendChild(body);
    }

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
    return messageDiv;
}

// ─── 渲染推理过程折叠块 ───
function renderReasoningBlock(steps) {
    const stepsHtml = steps.map((step, index) => `
        <div class="reasoning-step">
            <div class="reasoning-step-question">📌 子问题 ${index + 1}: ${escapeHtml(step.question)}</div>
            <div class="reasoning-step-answer">${renderMarkdown(step.answer)}</div>
        </div>
    `).join('');

    return `
        <div class="reasoning-block">
            <button class="reasoning-header">
                <span class="arrow">▶</span>
                🧩 推理过程（${steps.length} 个子问题）
            </button>
            <div class="reasoning-content">
                ${stepsHtml}
            </div>
        </div>
    `;
}

function renderDecomposition(subQuestions) {
    const questionsHtml = subQuestions.map((question, index) => {
        const questionText = typeof question === 'string' ? question : (question.sub_question || JSON.stringify(question));
        return `<li>${escapeHtml(questionText)}</li>`;
    }).join('');

    return `
        <div class="reasoning-block" style="margin-bottom: 12px;">
            <button class="reasoning-header expanded">
                <span class="arrow" style="transform: rotate(90deg);">▶</span>
                🔍 问题分解
            </button>
            <div class="reasoning-content expanded">
                <ol style="padding-left: 20px;">${questionsHtml}</ol>
            </div>
        </div>
    `;
}

function bindReasoningToggle(container) {
    container.querySelectorAll('.reasoning-header').forEach(header => {
        // 避免重复绑定
        if (header.dataset.bound) return;
        header.dataset.bound = 'true';
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            content.classList.toggle('expanded');
        });
    });
}

// 实时更新检索进度展示
function updateRetrievalProgress(reasoningArea, event) {
    let progressBlock = reasoningArea.querySelector('.retrieval-progress-block');

    if (!progressBlock) {
        progressBlock = document.createElement('div');
        progressBlock.className = 'reasoning-block retrieval-progress-block';
        progressBlock.innerHTML = `
            <button class="reasoning-header expanded" data-bound="true">
                <span class="arrow" style="transform: rotate(90deg);">▶</span>
                🔍 检索进度
            </button>
            <div class="reasoning-content expanded">
                <div class="retrieval-steps"></div>
            </div>
        `;
        const header = progressBlock.querySelector('.reasoning-header');
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            content.classList.toggle('expanded');
        });
        reasoningArea.appendChild(progressBlock);
    }

    const stepsContainer = progressBlock.querySelector('.retrieval-steps');
    const stage = event.stage;
    const message = event.message || '';

    // 根据阶段选择图标
    const stageIcons = {
        'start': '🚀',
        'internal_done': '📚',
        'dedup_done': '🔄',
        'done': '✅',
    };
    const icon = stageIcons[stage] || '📌';

    // 追加进度步骤
    const stepDiv = document.createElement('div');
    stepDiv.className = 'retrieval-step';
    stepDiv.style.cssText = 'padding: 4px 0; font-size: 13px; color: var(--text-secondary);';

    let stepContent = `${icon} ${escapeHtml(message)}`;
    if (stage === 'done' && event.avg_score) {
        stepContent += ` <span style="color: var(--text-muted); font-size: 12px;">(平均置信度: ${event.avg_score})</span>`;
    }
    if (stage === 'start' && event.collections && event.collections.length > 0) {
        stepContent += ` <span style="color: var(--text-muted); font-size: 12px;">[${event.collections.join(', ')}]</span>`;
    }
    stepDiv.innerHTML = stepContent;
    stepsContainer.appendChild(stepDiv);

    // 完成时自动折叠
    if (stage === 'done') {
        const header = progressBlock.querySelector('.reasoning-header');
        const content = progressBlock.querySelector('.reasoning-content');
        setTimeout(() => {
            header.classList.remove('expanded');
            content.classList.remove('expanded');
            header.querySelector('.arrow').style.transform = '';
        }, 1500);
    }
}

// 实时更新推理过程展示（每收到一个 sub_answer 就更新）
function updateReasoningDisplay(reasoningArea, steps) {
    // 查找或创建推理过程折叠块
    let reasoningBlock = reasoningArea.querySelector('.reasoning-steps-block');
    if (!reasoningBlock) {
        reasoningBlock = document.createElement('div');
        reasoningBlock.className = 'reasoning-block reasoning-steps-block';
        reasoningBlock.innerHTML = `
            <button class="reasoning-header expanded" data-bound="true">
                <span class="arrow" style="transform: rotate(90deg);">▶</span>
                🧩 推理过程（<span class="step-count">0</span> 个子问题）
            </button>
            <div class="reasoning-content expanded"></div>
        `;
        // 绑定折叠事件
        const header = reasoningBlock.querySelector('.reasoning-header');
        header.addEventListener('click', () => {
            header.classList.toggle('expanded');
            const content = header.nextElementSibling;
            content.classList.toggle('expanded');
        });
        reasoningArea.appendChild(reasoningBlock);
    }

    // 更新步骤计数
    reasoningBlock.querySelector('.step-count').textContent = steps.length;

    // 渲染最新的步骤（只追加新的，不重新渲染全部）
    const contentEl = reasoningBlock.querySelector('.reasoning-content');
    const existingCount = contentEl.querySelectorAll('.reasoning-step').length;

    for (let i = existingCount; i < steps.length; i++) {
        const step = steps[i];
        const stepDiv = document.createElement('div');
        stepDiv.className = 'reasoning-step';
        stepDiv.innerHTML = `
            <div class="reasoning-step-question">📌 子问题 ${i + 1}: ${escapeHtml(step.question)}</div>
            <div class="reasoning-step-answer">${renderMarkdown(step.answer)}</div>
        `;
        contentEl.appendChild(stepDiv);
    }
}

// ─── 渲染历史会话 ───
function renderConversationHistory(messages) {
    chatMessages.innerHTML = '';
    showChatMessages();

    messages.forEach(msg => {
        appendMessage(msg.role, msg.content);
    });
}

// ─── 视图控制 ───
function showChatMessages() {
    chatWelcome.style.display = 'none';
    chatMessages.classList.add('has-messages');
    document.getElementById('page-chat').classList.add('chatting');
}

function resetChatView() {
    chatMessages.innerHTML = '';
    chatMessages.classList.remove('has-messages');
    chatWelcome.style.display = '';
    chatInput.value = '';
    chatInput.style.height = 'auto';
    document.getElementById('page-chat').classList.remove('chatting');
}

function updateSendButtonState() {
    if (AppState.isStreaming) {
        btnSend.classList.add('hidden');
        btnStop.classList.remove('hidden');
    } else {
        btnSend.classList.remove('hidden');
        btnStop.classList.add('hidden');
    }
}

function scrollToBottom() {
    const container = chatMessages.parentElement;
    requestAnimationFrame(() => {
        container.scrollTop = container.scrollHeight;
    });
}

// ─── 导出需要跨文件调用的函数到全局 ───
window.renderConversationHistory = renderConversationHistory;
window.resetChatView = resetChatView;

})(); // IIFE 结束
