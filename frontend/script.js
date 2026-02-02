// ✅ 你的后端 API 地址 (确保 Space 处于 Running 状态)
const API_URL = "https://jeffliulab-akkadian-backend.hf.space/api/v1/translate";

document.addEventListener("DOMContentLoaded", () => {
    const runBtn = document.getElementById("runBtn");
    runBtn.addEventListener("click", handleAction);
});

async function handleAction() {
    // 1. 获取 DOM 元素
    const inputField = document.getElementById("inputText");
    const tabletContent = document.getElementById("tabletContent");
    const modelSelect = document.getElementById("modelSelect");
    const runBtn = document.getElementById("runBtn");
    const statusMsg = document.getElementById("statusMsg");

    // 2. 验证输入
    const text = inputField.value.trim();
    if (!text) {
        statusMsg.innerText = "Please enter some text to translate.";
        statusMsg.style.color = "#d9534f"; // 红色警告
        return;
    }

    // 3. 准备发送 (UI 进入加载状态)
    const selectedModel = modelSelect.value;
    const originalBtnText = runBtn.innerText;
    
    runBtn.innerText = "CARVING...";
    runBtn.disabled = true;
    tabletContent.style.opacity = "0.5";
    statusMsg.innerText = selectedModel === "byt5" ? "Consulting the AI Neural Network..." : "Processing...";
    statusMsg.style.color = "#666";

    try {
        // 4. 发送请求给后端
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                model_id: selectedModel
            })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        // 5. 显示结果
        // 后端返回 { "translation": "..." }
        if (data.translation) {
            tabletContent.innerText = data.translation;
        } else {
            tabletContent.innerText = JSON.stringify(data);
        }
        statusMsg.innerText = "Inscription Complete.";
        statusMsg.style.color = "#28a745"; // 绿色成功

    } catch (error) {
        console.error("Error:", error);
        tabletContent.innerText = "Error";
        statusMsg.innerText = "Could not reach the scribe (Network Error).";
        statusMsg.style.color = "#d9534f";
    } finally {
        // 6. 恢复 UI 状态
        runBtn.innerText = originalBtnText;
        runBtn.disabled = false;
        tabletContent.style.opacity = "1";
    }
}