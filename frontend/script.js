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
        statusMsg.style.color = "#d9534f"; 
        return;
    }

    // 3. 准备发送 (UI 进入加载状态)
    const selectedModel = modelSelect.value;
    const originalBtnText = runBtn.innerText;
    
    runBtn.innerText = "CARVING...";
    runBtn.disabled = true;
    
    // 移除之前的动画类，准备重新播放
    tabletContent.classList.remove("animate-reveal");
    tabletContent.style.opacity = "0.3"; // 变暗表示正在处理
    
    statusMsg.innerText = selectedModel === "byt5" ? "Consulting the AI Neural Network..." : "Processing...";
    statusMsg.style.color = "#666"; // 灰色状态字

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
        let resultText = "";
        if (data.translation) {
            resultText = data.translation;
        } else {
            resultText = JSON.stringify(data);
        }

        tabletContent.innerText = resultText;
        
        // 触发动画：强制浏览器重绘 (Reflow)
        void tabletContent.offsetWidth; 
        tabletContent.classList.add("animate-reveal"); // 添加动画类
        tabletContent.style.opacity = "1";

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
    }
}