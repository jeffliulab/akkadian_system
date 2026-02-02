// 后端 API 地址 (本地调试默认端口)
const API_URL = "http://localhost:8000/api/v1/translate";

async function handleAction() {
    const textInput = document.getElementById('inputText');
    const modelSelect = document.getElementById('modelSelect');
    const tabletContent = document.getElementById('tabletContent');
    const runBtn = document.getElementById('runBtn');

    const text = textInput.value.trim();
    const model = modelSelect.value;

    // 1. 基础验证：非空且为英文
    // 允许英文字母、数字、常见标点符号
    const isEnglish = /^[A-Za-z0-9\s.,!?'"()\-]+$/.test(text);

    if (!text) {
        return; 
    }

    if (!isEnglish) {
        tabletContent.innerHTML = `<span class="error-text">ERROR: ONLY ENGLISH INPUT ALLOWED</span>`;
        return;
    }

    // 2. UI 状态更新：显示加载中
    runBtn.disabled = true;
    runBtn.innerText = "CARVING...";
    tabletContent.style.opacity = "0.5";

    try {
        // 3. 发送请求给后端
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                model_id: model
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        // 4. 渲染结果
        // 这里直接显示后端传回来的确定性楔形文字
        tabletContent.innerText = data.translation;
        tabletContent.classList.remove('error-text');

    } catch (error) {
        console.error('Fetch error:', error);
        tabletContent.innerHTML = `<span class="error-text">CONNECTION ERROR: Backend Offline</span>`;
    } finally {
        // 5. 恢复 UI 状态
        runBtn.disabled = false;
        runBtn.innerText = "SHOW";
        tabletContent.style.opacity = "1";
    }
}