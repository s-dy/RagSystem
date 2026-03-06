/**
 * HybridRAG 前端 - 知识库管理页面逻辑
 */
(function () {
'use strict';

let knowledgeLoaded = false;
let knowledgeEventsInitialized = false;
let cachedCollections = [];
let cachedDocuments = [];

// ─── 页面进入回调 ───
function onKnowledgePageEnter() {
    if (!knowledgeLoaded) {
        loadDashboard();
        knowledgeLoaded = true;
    }
    if (!knowledgeEventsInitialized) {
        initKnowledgeEvents();
        knowledgeEventsInitialized = true;
    }
}

async function loadDashboard() {
    const results = await Promise.allSettled([
        loadModelStatus(),
        loadCollections(),
        loadChunkConfig(),
    ]);
    // 记录失败的任务，但不阻塞其他任务
    results.forEach((result, index) => {
        if (result.status === 'rejected') {
            const names = ['loadModelStatus', 'loadCollections', 'loadChunkConfig'];
            console.warn(`${names[index]} 加载失败:`, result.reason?.message || result.reason);
        }
    });
}

function initKnowledgeEvents() {
    // 刷新仪表盘（防重复点击）
    document.getElementById('btn-refresh-dashboard').addEventListener('click', () => {
        const btn = document.getElementById('btn-refresh-dashboard');
        if (btn.disabled) return;
        btn.disabled = true;
        btn.classList.add('spinning');
        loadDashboard().finally(() => {
            setTimeout(() => {
                btn.classList.remove('spinning');
                btn.disabled = false;
            }, 400);
        });
    });

    // 刷新知识库列表（防重复点击）
    document.getElementById('btn-refresh-collections').addEventListener('click', () => {
        const btn = document.getElementById('btn-refresh-collections');
        if (btn.disabled) return;
        btn.disabled = true;
        btn.classList.add('spinning');
        loadCollections().finally(() => {
            setTimeout(() => {
                btn.classList.remove('spinning');
                btn.disabled = false;
            }, 400);
        });
    });

    // 上传按钮（dropzone 内的"点击选择文件"）
    document.getElementById('btn-upload-doc').addEventListener('click', (event) => {
        event.preventDefault();
        openUploadDialog();
    });

    // 搜索过滤（200ms 防抖）
    let _searchTimer = null;
    document.getElementById('knowledge-search-input').addEventListener('input', (event) => {
        clearTimeout(_searchTimer);
        const keyword = event.target.value.trim();
        _searchTimer = setTimeout(() => filterCollections(keyword), 200);
    });

    // Tabs 切换
    initTabs();

    // 分块配置
    initChunkConfig();

    // 文档管理
    initDocumentManager();

    // 拖拽上传
    initDropzone();
}

// ─── Tabs 切换 ───
function initTabs() {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;

            document.querySelectorAll('.tab-btn').forEach(tabBtn => tabBtn.classList.remove('active'));
            document.querySelectorAll('.tab-panel').forEach(panel => panel.classList.remove('active'));

            btn.classList.add('active');
            const targetPanel = document.getElementById(targetTab);
            if (targetPanel) targetPanel.classList.add('active');

            if (targetTab === 'tab-documents') {
                loadDocuments();
            }
        });
    });
}

// ─── 模型状态 ───
function renderModelSkeleton() {
    const container = document.getElementById('model-cards');
    container.innerHTML = '';
    for (let i = 0; i < 3; i++) {
        const card = document.createElement('div');
        card.className = 'skeleton-model-card';
        card.innerHTML = `
            <div class="skeleton skeleton-avatar"></div>
            <div class="skeleton-info">
                <div class="skeleton skeleton-line-sm"></div>
                <div class="skeleton skeleton-line-md"></div>
                <div class="skeleton skeleton-line-xs"></div>
            </div>
            <div class="skeleton skeleton-dot"></div>
        `;
        container.appendChild(card);
    }
}

async function loadModelStatus() {
    renderModelSkeleton();
    try {
        const data = await apiFetch(API.systemModels, {}, { retries: 2, retryDelay: 1500 });
        rebuildModelCards(data);
    } catch (error) {
        rebuildModelCardsOffline();
    }
}

function rebuildModelCards(data) {
    const container = document.getElementById('model-cards');
    container.innerHTML = '';

    const models = [
        { type: 'llm', label: '大语言模型', abbr: 'LLM', iconClass: 'llm-icon', info: data.llm },
        { type: 'embed', label: '嵌入模型', abbr: 'EMB', iconClass: 'embed-icon', info: data.embedding },
        { type: 'reranker', label: '重排序模型', abbr: 'RRK', iconClass: 'reranker-icon', info: data.reranker },
    ];

    models.forEach(({ type, label, abbr, iconClass, info }) => {
        const card = document.createElement('div');
        card.className = 'model-card';

        const icon = document.createElement('div');
        icon.className = `model-card-icon ${iconClass}`;
        icon.textContent = abbr;

        const infoDiv = document.createElement('div');
        infoDiv.className = 'model-card-info';

        const labelEl = document.createElement('div');
        labelEl.className = 'model-card-label';
        labelEl.textContent = label;

        const nameEl = document.createElement('div');
        nameEl.className = 'model-card-name';
        nameEl.id = `model-${type}-name`;
        nameEl.textContent = info.name || '未配置';

        const providerEl = document.createElement('div');
        providerEl.className = 'model-card-provider';
        providerEl.id = `model-${type}-provider`;
        providerEl.textContent = info.provider || '';

        infoDiv.appendChild(labelEl);
        infoDiv.appendChild(nameEl);
        infoDiv.appendChild(providerEl);

        const statusDiv = document.createElement('div');
        statusDiv.className = 'model-card-status';
        statusDiv.id = `model-${type}-status`;
        const dot = document.createElement('span');
        dot.className = `status-dot ${info.status === 'online' ? 'online' : 'offline'}`;
        statusDiv.appendChild(dot);

        card.appendChild(icon);
        card.appendChild(infoDiv);
        card.appendChild(statusDiv);
        container.appendChild(card);
    });
}

function rebuildModelCardsOffline() {
    const container = document.getElementById('model-cards');
    container.innerHTML = '';

    const models = [
        { type: 'llm', label: '大语言模型', abbr: 'LLM', iconClass: 'llm-icon' },
        { type: 'embed', label: '嵌入模型', abbr: 'EMB', iconClass: 'embed-icon' },
        { type: 'reranker', label: '重排序模型', abbr: 'RRK', iconClass: 'reranker-icon' },
    ];

    models.forEach(({ type, label, abbr, iconClass }) => {
        const card = document.createElement('div');
        card.className = 'model-card';

        const icon = document.createElement('div');
        icon.className = `model-card-icon ${iconClass}`;
        icon.textContent = abbr;

        const infoDiv = document.createElement('div');
        infoDiv.className = 'model-card-info';

        const labelEl = document.createElement('div');
        labelEl.className = 'model-card-label';
        labelEl.textContent = label;

        const nameEl = document.createElement('div');
        nameEl.className = 'model-card-name';
        nameEl.textContent = '连接失败';

        infoDiv.appendChild(labelEl);
        infoDiv.appendChild(nameEl);

        const statusDiv = document.createElement('div');
        statusDiv.className = 'model-card-status';
        const dot = document.createElement('span');
        dot.className = 'status-dot offline';
        statusDiv.appendChild(dot);

        card.appendChild(icon);
        card.appendChild(infoDiv);
        card.appendChild(statusDiv);
        container.appendChild(card);
    });
}



// ─── 分块配置 ───
function initChunkConfig() {
    const sizeSlider = document.getElementById('chunk-size-slider');
    const overlapSlider = document.getElementById('chunk-overlap-slider');
    const sizeValue = document.getElementById('chunk-size-value');
    const overlapValue = document.getElementById('chunk-overlap-value');

    sizeSlider.addEventListener('input', () => {
        sizeValue.textContent = sizeSlider.value;
        clearActivePreset();
    });

    overlapSlider.addEventListener('input', () => {
        overlapValue.textContent = overlapSlider.value;
        clearActivePreset();
    });

    let saveTimeout = null;
    const debouncedSave = () => {
        clearTimeout(saveTimeout);
        saveTimeout = setTimeout(() => {
            saveChunkConfig(parseInt(sizeSlider.value), parseInt(overlapSlider.value));
        }, 800);
    };

    sizeSlider.addEventListener('change', debouncedSave);
    overlapSlider.addEventListener('change', debouncedSave);

    document.querySelectorAll('.chunk-preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const size = parseInt(btn.dataset.size);
            const overlap = parseInt(btn.dataset.overlap);

            sizeSlider.value = size;
            overlapSlider.value = overlap;
            sizeValue.textContent = size;
            overlapValue.textContent = overlap;

            document.querySelectorAll('.chunk-preset-btn').forEach(presetBtn => presetBtn.classList.remove('active'));
            btn.classList.add('active');

            saveChunkConfig(size, overlap);
        });
    });
}

function clearActivePreset() {
    document.querySelectorAll('.chunk-preset-btn').forEach(btn => btn.classList.remove('active'));
}

async function loadChunkConfig() {
    try {
        const config = await apiFetch(API.chunkConfig, {}, { retries: 1 });

        const sizeSlider = document.getElementById('chunk-size-slider');
        const overlapSlider = document.getElementById('chunk-overlap-slider');
        const sizeValue = document.getElementById('chunk-size-value');
        const overlapValue = document.getElementById('chunk-overlap-value');

        sizeSlider.value = config.chunk_size;
        overlapSlider.value = config.chunk_overlap;
        sizeValue.textContent = config.chunk_size;
        overlapValue.textContent = config.chunk_overlap;

        document.querySelectorAll('.chunk-preset-btn').forEach(btn => {
            const isMatch = parseInt(btn.dataset.size) === config.chunk_size
                         && parseInt(btn.dataset.overlap) === config.chunk_overlap;
            btn.classList.toggle('active', isMatch);
        });
    } catch (error) {
        // 使用默认值
    }
}

async function saveChunkConfig(chunkSize, chunkOverlap) {
    try {
        await apiFetch(API.chunkConfig, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap }),
        }, { retries: 1 });
    } catch (error) {
        showToast('保存分块配置失败', 'error');
    }
}

// ─── 文档管理 ───
function initDocumentManager() {
    // 刷新文档列表
    document.getElementById('btn-refresh-documents').addEventListener('click', () => {
        const btn = document.getElementById('btn-refresh-documents');
        btn.classList.add('spinning');
        loadDocuments().finally(() => {
            setTimeout(() => btn.classList.remove('spinning'), 400);
        });
    });

    // 知识库筛选
    document.getElementById('doc-collection-select').addEventListener('change', () => {
        loadDocuments();
    });

    // 全选（支持 indeterminate 状态）
    document.getElementById('doc-select-all').addEventListener('change', (event) => {
        const checkboxes = document.querySelectorAll('#doc-table-body input[type="checkbox"]');
        checkboxes.forEach(cb => { cb.checked = event.target.checked; });
        event.target.indeterminate = false;
        updateBatchActions();
    });

    // 批量删除
    document.getElementById('btn-batch-delete').addEventListener('click', async () => {
        const selected = getSelectedDocuments();
        if (selected.length === 0) return;

        if (!confirm(`确定要删除选中的 ${selected.length} 个文件吗？`)) return;

        let successCount = 0;
        for (const doc of selected) {
            try {
                await apiFetch(API.documents, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ collection_name: doc.collection, filename: doc.filename }),
                }, { retries: 1 });
                successCount++;
            } catch (error) {
                // 继续删除其他文件
            }
        }

        showToast(`成功删除 ${successCount} 个文件`, 'success');
        loadDocuments();
        loadCollections();
    });
}

function getSelectedDocuments() {
    const checkboxes = document.querySelectorAll('#doc-table-body input[type="checkbox"]:checked');
    return Array.from(checkboxes).map(cb => ({
        filename: cb.dataset.filename,
        collection: cb.dataset.collection,
    }));
}

function updateBatchActions() {
    const selected = getSelectedDocuments();
    const batchBar = document.getElementById('doc-batch-actions');
    const countEl = document.getElementById('doc-batch-count');
    const allCheckboxes = document.querySelectorAll('#doc-table-body input[type="checkbox"]');
    const selectAll = document.getElementById('doc-select-all');

    // 更新全选 checkbox 的 indeterminate 状态
    if (allCheckboxes.length > 0) {
        const checkedCount = selected.length;
        selectAll.checked = checkedCount === allCheckboxes.length;
        selectAll.indeterminate = checkedCount > 0 && checkedCount < allCheckboxes.length;
    }

    if (selected.length > 0) {
        batchBar.classList.remove('hidden');
        countEl.textContent = `已选 ${selected.length} 个文件`;
    } else {
        batchBar.classList.add('hidden');
    }
}

async function loadDocuments() {
    const collectionName = document.getElementById('doc-collection-select').value;
    const queryParam = collectionName ? `?collection_name=${encodeURIComponent(collectionName)}` : '';

    // 填充知识库下拉选项
    const select = document.getElementById('doc-collection-select');
    const currentValue = select.value;
    select.innerHTML = '<option value="">-- 全部知识库 --</option>';
    cachedCollections.forEach(coll => {
        const option = document.createElement('option');
        option.value = coll.name;
        option.textContent = coll.name;
        select.appendChild(option);
    });
    select.value = currentValue;

    try {
        const documents = await apiFetch(`${API.documents}${queryParam}`, {}, { retries: 2 });
        cachedDocuments = documents;

        renderDocumentTable(documents);
        updateOverviewFileCount(documents.length);
    } catch (error) {
        const tbody = document.getElementById('doc-table-body');
        tbody.innerHTML = '';

        const tr = document.createElement('tr');
        tr.className = 'doc-empty-row';
        const td = document.createElement('td');
        td.colSpan = 7;

        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.style.padding = '40px 20px';

        const iconEl = document.createElement('div');
        iconEl.className = 'empty-state-icon';
        iconEl.textContent = '⚠️';

        const textEl = document.createElement('div');
        textEl.className = 'empty-state-text';
        textEl.textContent = '加载文档列表失败';

        const hintEl = document.createElement('div');
        hintEl.className = 'empty-state-hint';
        hintEl.textContent = error.message;

        emptyState.appendChild(iconEl);
        emptyState.appendChild(textEl);
        emptyState.appendChild(hintEl);
        td.appendChild(emptyState);
        tr.appendChild(td);
        tbody.appendChild(tr);
    }
}

function renderDocumentTable(documents) {
    const tbody = document.getElementById('doc-table-body');
    tbody.innerHTML = '';

    if (documents.length === 0) {
        const emptyRow = document.createElement('tr');
        emptyRow.className = 'doc-empty-row';
        const emptyCell = document.createElement('td');
        emptyCell.colSpan = 7;

        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';
        emptyState.style.padding = '40px 20px';

        const iconEl = document.createElement('div');
        iconEl.className = 'empty-state-icon';
        iconEl.textContent = '📄';

        const textEl = document.createElement('div');
        textEl.className = 'empty-state-text';
        textEl.textContent = '暂无文档';

        const hintEl = document.createElement('div');
        hintEl.className = 'empty-state-hint';
        hintEl.textContent = '上传文件后将在此处显示';

        emptyState.appendChild(iconEl);
        emptyState.appendChild(textEl);
        emptyState.appendChild(hintEl);
        emptyCell.appendChild(emptyState);
        emptyRow.appendChild(emptyCell);
        tbody.appendChild(emptyRow);
        return;
    }

    const statusMap = { uploaded: '已上传', processing: '处理中', failed: '失败' };

    documents.forEach(doc => {
        const tr = document.createElement('tr');

        // 复选框
        const tdCheck = document.createElement('td');
        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.dataset.filename = doc.filename;
        checkbox.dataset.collection = doc.collection_name;
        checkbox.addEventListener('change', updateBatchActions);
        tdCheck.appendChild(checkbox);

        // 文件名
        const tdName = document.createElement('td');
        const strong = document.createElement('strong');
        strong.textContent = doc.filename;
        tdName.appendChild(strong);

        // 大小
        const tdSize = document.createElement('td');
        tdSize.textContent = formatFileSize(doc.size);

        // 所属知识库
        const tdCollection = document.createElement('td');
        tdCollection.textContent = doc.collection_name;

        // 上传时间
        const tdTime = document.createElement('td');
        tdTime.textContent = doc.upload_time;

        // 状态
        const tdStatus = document.createElement('td');
        const statusSpan = document.createElement('span');
        const statusClass = doc.status === 'uploaded' ? 'uploaded' : doc.status === 'processing' ? 'processing' : 'failed';
        statusSpan.className = `doc-status ${statusClass}`;
        statusSpan.textContent = statusMap[doc.status] || doc.status;
        tdStatus.appendChild(statusSpan);

        // 操作
        const tdAction = document.createElement('td');
        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-danger-sm';
        deleteBtn.title = '删除';
        deleteBtn.appendChild(createSvgIcon('delete'));
        deleteBtn.addEventListener('click', async () => {
            if (!confirm(`确定要删除文件「${doc.filename}」吗？`)) return;
            try {
                await apiFetch(API.documents, {
                    method: 'DELETE',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ collection_name: doc.collection_name, filename: doc.filename }),
                }, { retries: 1, showErrorToast: true });
                showToast(`已删除文件「${doc.filename}」`, 'success');
                loadDocuments();
            } catch (error) {
                showToast(`删除失败: ${error.message}`, 'error');
            }
        });
        tdAction.appendChild(deleteBtn);

        tr.appendChild(tdCheck);
        tr.appendChild(tdName);
        tr.appendChild(tdSize);
        tr.appendChild(tdCollection);
        tr.appendChild(tdTime);
        tr.appendChild(tdStatus);
        tr.appendChild(tdAction);
        tbody.appendChild(tr);
    });

    // 重置全选
    document.getElementById('doc-select-all').checked = false;
    document.getElementById('doc-batch-actions').classList.add('hidden');
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function updateOverviewFileCount(count) {
    const el = document.getElementById('stat-total-files');
    if (el) el.textContent = count.toLocaleString();
}

// ─── 拖拽上传区 ───
function initDropzone() {
    const dropzone = document.getElementById('upload-dropzone');
    let dragCounter = 0;

    dropzone.addEventListener('dragenter', (event) => {
        event.preventDefault();
        dragCounter++;
        dropzone.classList.add('drag-over');
    });

    dropzone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dragCounter--;
        if (dragCounter <= 0) {
            dragCounter = 0;
            dropzone.classList.remove('drag-over');
        }
    });

    dropzone.addEventListener('dragover', (event) => {
        event.preventDefault();
    });

    dropzone.addEventListener('drop', (event) => {
        event.preventDefault();
        dragCounter = 0;
        dropzone.classList.remove('drag-over');

        const files = event.dataTransfer.files;
        if (files && files.length > 0) {
            showUploadModal(files);
        }
    });
}

// ─── 搜索过滤 ───
function filterCollections(keyword) {
    const lowerKeyword = keyword.toLowerCase();
    const cards = document.querySelectorAll('.collection-card');

    let visibleCount = 0;
    cards.forEach(card => {
        const name = (card.dataset.name || '').toLowerCase();
        const isMatch = !keyword || name.includes(lowerKeyword);
        card.style.display = isMatch ? '' : 'none';
        if (isMatch) visibleCount++;
    });

    // 显示/隐藏"无搜索结果"提示
    const grid = document.getElementById('collection-grid');
    let noResult = grid.querySelector('.search-no-result');
    if (visibleCount === 0 && keyword && cachedCollections.length > 0) {
        if (!noResult) {
            noResult = document.createElement('div');
            noResult.className = 'empty-state search-no-result';

            const iconEl = document.createElement('div');
            iconEl.className = 'empty-state-icon';
            iconEl.textContent = '🔍';

            const textEl = document.createElement('div');
            textEl.className = 'empty-state-text';
            textEl.textContent = '未找到匹配的知识库';

            const hintEl = document.createElement('div');
            hintEl.className = 'empty-state-hint';
            hintEl.textContent = '尝试其他关键词';

            noResult.appendChild(iconEl);
            noResult.appendChild(textEl);
            noResult.appendChild(hintEl);
            grid.appendChild(noResult);
        }
    } else if (noResult) {
        noResult.remove();
    }
}

// ─── 加载知识库列表 ───
function renderCollectionSkeleton() {
    const grid = document.getElementById('collection-grid');
    grid.innerHTML = '';
    for (let i = 0; i < 3; i++) {
        const card = document.createElement('div');
        card.className = 'skeleton-card';
        card.innerHTML = `
            <div class="skeleton skeleton-icon"></div>
            <div class="skeleton skeleton-title"></div>
            <div class="skeleton skeleton-desc"></div>
            <div class="skeleton skeleton-stats"></div>
        `;
        grid.appendChild(card);
    }
}

async function loadCollections() {
    const grid = document.getElementById('collection-grid');
    renderCollectionSkeleton();

    try {
        const collections = await apiFetch(API.collections, {}, { retries: 2 });

        if (collections.error) {
            throw new Error(collections.error);
        }

        cachedCollections = collections;
        updateKnowledgeStat(collections);

        if (!collections.length) {
            grid.innerHTML = '';
            const emptyState = document.createElement('div');
            emptyState.className = 'empty-state';

            const iconEl = document.createElement('div');
            iconEl.className = 'empty-state-icon';
            iconEl.textContent = '📭';

            const textEl = document.createElement('div');
            textEl.className = 'empty-state-text';
            textEl.textContent = '暂无知识库集合';

            const hintEl = document.createElement('div');
            hintEl.className = 'empty-state-hint';
            hintEl.textContent = '拖拽文件到上方区域或点击上传，开始构建知识库';

            emptyState.appendChild(iconEl);
            emptyState.appendChild(textEl);
            emptyState.appendChild(hintEl);
            grid.appendChild(emptyState);
            return;
        }

        renderCollectionCards(collections);

    } catch (error) {
        grid.innerHTML = '';
        const emptyState = document.createElement('div');
        emptyState.className = 'empty-state';

        const iconEl = document.createElement('div');
        iconEl.className = 'empty-state-icon';
        iconEl.textContent = '⚠️';

        const textEl = document.createElement('div');
        textEl.className = 'empty-state-text';
        textEl.textContent = '加载失败: ' + error.message;

        const hintEl = document.createElement('div');
        hintEl.className = 'empty-state-hint';
        hintEl.textContent = '请确认 Milvus 服务已启动';

        emptyState.appendChild(iconEl);
        emptyState.appendChild(textEl);
        emptyState.appendChild(hintEl);
        grid.appendChild(emptyState);

        updateKnowledgeStatOffline(error.message);
    }
}

// ─── 渲染卡片（DOM API，防 XSS） ───
function renderCollectionCards(collections) {
    const grid = document.getElementById('collection-grid');
    grid.innerHTML = '';

    collections.forEach(coll => {
        const card = document.createElement('div');
        card.className = 'collection-card';
        card.dataset.name = coll.name;

        // 头部
        const header = document.createElement('div');
        header.className = 'collection-card-header';

        const iconDiv = document.createElement('div');
        iconDiv.className = 'collection-card-icon';
        iconDiv.textContent = '📚';

        const actionsDiv = document.createElement('div');
        actionsDiv.className = 'collection-card-actions';

        const deleteBtn = document.createElement('button');
        deleteBtn.className = 'btn-danger-sm';
        deleteBtn.title = '删除知识库';
        deleteBtn.appendChild(createSvgIcon('delete'));
        deleteBtn.addEventListener('click', async (event) => {
            event.stopPropagation();
            if (confirm(`确定要删除知识库「${coll.name}」吗？此操作不可恢复。`)) {
                await deleteCollection(coll.name);
            }
        });
        actionsDiv.appendChild(deleteBtn);

        header.appendChild(iconDiv);
        header.appendChild(actionsDiv);

        // 名称
        const nameDiv = document.createElement('div');
        nameDiv.className = 'collection-card-name';
        nameDiv.textContent = coll.name;

        // 描述
        const descDiv = document.createElement('div');
        descDiv.className = 'collection-card-desc';
        descDiv.textContent = '向量知识库集合';

        // 统计
        const statsDiv = document.createElement('div');
        statsDiv.className = 'collection-card-stats';

        const statItem = document.createElement('div');
        statItem.className = 'collection-card-stat';
        statItem.appendChild(createSvgIcon('file'));

        const statValue = document.createElement('span');
        statValue.className = 'stat-value';
        statValue.textContent = (coll.num_entities || 0).toLocaleString();
        statItem.appendChild(statValue);
        statItem.appendChild(document.createTextNode(' 条记录'));

        statsDiv.appendChild(statItem);

        card.appendChild(header);
        card.appendChild(nameDiv);
        card.appendChild(descDiv);
        card.appendChild(statsDiv);
        grid.appendChild(card);
    });

    // 恢复搜索过滤状态
    const searchInput = document.getElementById('knowledge-search-input');
    if (searchInput && searchInput.value.trim()) {
        filterCollections(searchInput.value.trim());
    }
}

// ─── 更新统计信息 ───
function updateKnowledgeStat(collections) {
    const totalEntities = collections.reduce((sum, coll) => sum + (coll.num_entities || 0), 0);

    // 页头统计标签
    const statEl = document.getElementById('knowledge-stat');
    if (statEl) {
        statEl.textContent = `${collections.length} 个知识库 · ${totalEntities.toLocaleString()} 条记录`;
    }

    // 概览统计卡片
    const collCountEl = document.getElementById('stat-collection-count');
    const totalRecordsEl = document.getElementById('stat-total-records');
    const dbStatusEl = document.getElementById('stat-db-status');

    if (collCountEl) collCountEl.textContent = collections.length;
    if (totalRecordsEl) totalRecordsEl.textContent = totalEntities.toLocaleString();
    if (dbStatusEl) dbStatusEl.textContent = '在线';
    if (dbStatusEl) dbStatusEl.style.color = 'var(--success)';
}

function updateKnowledgeStatOffline(errorMessage) {
    const statEl = document.getElementById('knowledge-stat');
    if (statEl) statEl.textContent = '连接失败';

    const dbStatusEl = document.getElementById('stat-db-status');
    if (dbStatusEl) {
        dbStatusEl.textContent = '离线';
        dbStatusEl.style.color = 'var(--danger)';
    }
}

// ─── 删除知识库 ───
async function deleteCollection(collectionName) {
    try {
        const result = await apiFetch(`${API.collections}/${encodeURIComponent(collectionName)}`, {
            method: 'DELETE',
        }, { retries: 1, showErrorToast: true });

        if (result.error) {
            throw new Error(result.error);
        }

        showToast(`知识库「${collectionName}」已删除`, 'success');
        loadCollections();
    } catch (error) {
        showToast(`删除失败: ${error.message}`, 'error');
    }
}

// ─── 上传文档 ───
let pendingUploadFiles = null;

function openUploadDialog() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = '.pdf,.docx,.doc,.txt,.md';
    fileInput.multiple = true;

    fileInput.addEventListener('change', () => {
        const files = fileInput.files;
        if (files && files.length > 0) {
            showUploadModal(files);
        }
    });

    fileInput.click();
}

// ─── 上传弹窗 ───
function showUploadModal(files) {
    const allowedExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md'];
    const validFiles = Array.from(files).filter(file => {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        return allowedExtensions.includes(ext);
    });

    if (validFiles.length === 0) {
        showToast('不支持的文件格式，请上传 PDF、Word、TXT 或 Markdown 文件', 'warning');
        return;
    }

    if (validFiles.length < files.length) {
        showToast(`已过滤 ${files.length - validFiles.length} 个不支持的文件`, 'warning');
    }

    // 文件大小限制：单文件 ≤ 50MB，总大小 ≤ 200MB
    const MAX_SINGLE_SIZE = 50 * 1024 * 1024;
    const MAX_TOTAL_SIZE = 200 * 1024 * 1024;
    const oversizedFiles = validFiles.filter(f => f.size > MAX_SINGLE_SIZE);
    if (oversizedFiles.length > 0) {
        const names = oversizedFiles.map(f => f.name).join('、');
        showToast(`以下文件超过 50MB 限制：${names}`, 'warning');
        return;
    }
    const totalSize = validFiles.reduce((sum, f) => sum + f.size, 0);
    if (totalSize > MAX_TOTAL_SIZE) {
        const totalMB = (totalSize / (1024 * 1024)).toFixed(1);
        showToast(`文件总大小 ${totalMB}MB 超过 200MB 限制`, 'warning');
        return;
    }

    pendingUploadFiles = validFiles;

    // 填充知识库下拉选项
    const select = document.getElementById('upload-collection-select');
    select.innerHTML = '<option value="">-- 选择已有知识库 --</option>';
    cachedCollections.forEach(coll => {
        const option = document.createElement('option');
        option.value = coll.name;
        option.textContent = coll.name;
        select.appendChild(option);
    });

    // 清空新建输入框
    document.getElementById('upload-collection-new').value = '';

    // 渲染文件列表（DOM API，防 XSS）
    const fileListEl = document.getElementById('upload-modal-files');
    fileListEl.innerHTML = '';
    validFiles.forEach(file => {
        const sizeKB = (file.size / 1024).toFixed(1);
        const tag = document.createElement('span');
        tag.className = 'upload-file-tag';
        tag.appendChild(createSvgIcon('file'));
        const nameText = document.createTextNode(' ' + file.name + ' ');
        tag.appendChild(nameText);
        const sizeSpan = document.createElement('span');
        sizeSpan.style.color = 'var(--text-muted)';
        sizeSpan.textContent = `(${sizeKB} KB)`;
        tag.appendChild(sizeSpan);
        fileListEl.appendChild(tag);
    });

    // 更新确认按钮状态
    updateUploadConfirmState();

    // 显示弹窗
    document.getElementById('upload-modal-overlay').classList.remove('hidden');
}

function closeUploadModal() {
    document.getElementById('upload-modal-overlay').classList.add('hidden');
    pendingUploadFiles = null;
}

function getSelectedCollectionName() {
    const selectValue = document.getElementById('upload-collection-select').value;
    const newValue = document.getElementById('upload-collection-new').value.trim();
    return newValue || selectValue;
}

function updateUploadConfirmState() {
    const collectionName = getSelectedCollectionName();
    const confirmBtn = document.getElementById('upload-modal-confirm');
    confirmBtn.disabled = !collectionName;
}

function initUploadModal() {
    // 关闭按钮
    document.getElementById('upload-modal-close').addEventListener('click', closeUploadModal);
    document.getElementById('upload-modal-cancel').addEventListener('click', closeUploadModal);

    // 点击遮罩关闭
    document.getElementById('upload-modal-overlay').addEventListener('click', (event) => {
        if (event.target === event.currentTarget) {
            closeUploadModal();
        }
    });

    // 下拉选择变化时：清空新建输入框，更新按钮状态
    document.getElementById('upload-collection-select').addEventListener('change', (event) => {
        if (event.target.value) {
            document.getElementById('upload-collection-new').value = '';
        }
        updateUploadConfirmState();
    });

    // 新建输入框变化时：清空下拉选择，更新按钮状态
    document.getElementById('upload-collection-new').addEventListener('input', (event) => {
        if (event.target.value.trim()) {
            document.getElementById('upload-collection-select').value = '';
        }
        updateUploadConfirmState();
    });

    // 确认上传
    document.getElementById('upload-modal-confirm').addEventListener('click', () => {
        const collectionName = getSelectedCollectionName();
        if (!collectionName || !pendingUploadFiles) return;

        // 先保存文件引用，因为 closeUploadModal 会将 pendingUploadFiles 置为 null
        const filesToUpload = pendingUploadFiles;
        closeUploadModal();
        uploadFiles(filesToUpload, collectionName);
    });
}

// 在 initKnowledgeEvents 之外初始化（弹窗在 body 层级，不依赖页面切换）
document.addEventListener('DOMContentLoaded', initUploadModal);

async function uploadFiles(files, collectionName) {
    const validFiles = Array.from(files);

    // 显示上传进度 UI
    const progressContainer = document.getElementById('upload-progress');
    const progressTitle = document.getElementById('upload-progress-title');
    const progressPercent = document.getElementById('upload-progress-percent');
    const progressFill = document.getElementById('upload-progress-fill');
    const fileListEl = document.getElementById('upload-file-list');
    const dropzone = document.getElementById('upload-dropzone');

    progressContainer.classList.remove('hidden');
    dropzone.classList.add('uploading');
    progressTitle.textContent = `正在上传 ${validFiles.length} 个文件到「${collectionName}」...`;
    progressPercent.textContent = '0%';
    progressFill.style.width = '0%';

    // 渲染文件标签（DOM API，防 XSS）
    fileListEl.innerHTML = '';
    validFiles.forEach(file => {
        const tag = document.createElement('span');
        tag.className = 'upload-file-tag';
        tag.appendChild(createSvgIcon('file'));
        const nameText = document.createTextNode(' ' + file.name);
        tag.appendChild(nameText);
        fileListEl.appendChild(tag);
    });

    // 模拟进度
    let fakeProgress = 0;
    const progressInterval = setInterval(() => {
        fakeProgress = Math.min(fakeProgress + Math.random() * 15, 90);
        progressFill.style.width = fakeProgress + '%';
        progressPercent.textContent = Math.round(fakeProgress) + '%';
    }, 300);

    const formData = new FormData();
    for (const file of validFiles) {
        formData.append('files', file);
    }

    const uploadUrl = `${API.upload}?collection_name=${encodeURIComponent(collectionName)}`;

    try {
        const response = await fetch(uploadUrl, {
            method: 'POST',
            body: formData,
        });

        clearInterval(progressInterval);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `上传失败: ${response.status}`);
        }

        const result = await response.json();

        if (result.error) {
            throw new Error(result.error);
        }

        // 上传成功，开始轮询入库状态
        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';
        progressTitle.textContent = '上传完成，正在处理入库...';

        showToast(result.message || '文件上传成功，入库处理中', 'success');

        // 轮询入库状态
        pollIngestStatus(collectionName, progressContainer, progressTitle, progressPercent, progressFill, dropzone, fileListEl);

    } catch (error) {
        clearInterval(progressInterval);
        progressTitle.textContent = '上传失败';
        progressPercent.textContent = '';
        progressFill.style.width = '100%';
        progressFill.style.background = 'var(--danger)';

        showToast(`上传失败: ${error.message}`, 'error');

        setTimeout(() => {
            progressContainer.classList.add('hidden');
            dropzone.classList.remove('uploading');
            progressFill.style.width = '0%';
            progressFill.style.background = '';
            fileListEl.innerHTML = '';
        }, 3000);
    }
}

// ─── 入库状态轮询 ───
async function pollIngestStatus(collectionName, progressContainer, progressTitle, progressPercent, progressFill, dropzone, fileListEl) {
    const maxAttempts = 60;
    const intervalMs = 3000;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
        await new Promise(resolve => setTimeout(resolve, intervalMs));

        try {
            const statusData = await apiFetch(
                `${API.ingestStatus}/${encodeURIComponent(collectionName)}`,
                {},
                { retries: 0 },
            );

            if (statusData.status === 'completed') {
                progressTitle.textContent = statusData.message || '入库完成！';
                progressFill.style.width = '100%';
                progressFill.style.background = '';
                progressPercent.textContent = '✅';
                showToast(statusData.message || '文档入库完成', 'success');

                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    dropzone.classList.remove('uploading');
                    progressFill.style.width = '0%';
                    fileListEl.innerHTML = '';
                    loadCollections();
                    loadDocuments();
                }, 2000);
                return;
            }

            if (statusData.status === 'failed') {
                progressTitle.textContent = statusData.message || '入库失败';
                progressFill.style.background = 'var(--danger)';
                progressPercent.textContent = '❌';
                showToast(statusData.message || '文档入库失败', 'error');

                setTimeout(() => {
                    progressContainer.classList.add('hidden');
                    dropzone.classList.remove('uploading');
                    progressFill.style.width = '0%';
                    progressFill.style.background = '';
                    fileListEl.innerHTML = '';
                }, 3000);
                return;
            }

            // 仍在处理中，更新进度提示
            const progressValue = Math.min(50 + attempt * 2, 95);
            progressFill.style.width = progressValue + '%';
            progressPercent.textContent = progressValue + '%';
            progressTitle.textContent = statusData.message || '正在处理入库...';

        } catch (error) {
            console.warn('轮询入库状态失败:', error.message);
        }
    }

    // 超时处理
    progressTitle.textContent = '入库处理超时，请稍后在知识库列表中查看';
    progressPercent.textContent = '⏳';
    showToast('入库处理时间较长，请稍后刷新查看', 'warning');

    setTimeout(() => {
        progressContainer.classList.add('hidden');
        dropzone.classList.remove('uploading');
        progressFill.style.width = '0%';
        fileListEl.innerHTML = '';
        loadCollections();
    }, 3000);
}

// ─── 导出需要跨文件调用的函数到全局 ───
window.onKnowledgePageEnter = onKnowledgePageEnter;

})(); // IIFE 结束
