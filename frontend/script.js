// ✅ 你的 Hugging Face Space 后端 API 地址
const API_URL = "https://jeffliulab-akkadian-backend.hf.space/api/v1/translate";

document.addEventListener("DOMContentLoaded", () => {
    const translateBtn = document.getElementById("translate-btn");
    
    // 绑定点击事件
    translateBtn.addEventListener("click", handleAction);
});

async function handleAction() {
    // 1. 获取 DOM 元素
    const inputField = document.getElementById("input-text");
    const outputArea = document.getElementById("output-text");
    const modelSelect = document.getElementById("model-select");
    const translateBtn = document.getElementById("translate-btn");
    
    // 2. 基础验证
    const text = inputField.value.trim();
    if (!text) {
        // 简单的抖动效果或警告
        outputArea.innerText = "Please inscribe some text on the tablet first.";
        outputArea.style.color = "#8b0000"; // 深红色警告
        return;
    }

    // 获取用户选择的模型 ID ("default" 或 "byt5")
    const selectedModel = modelSelect.value; 

    // 3. 进入“加载中”状态 UI
    const originalBtnText = translateBtn.innerText;
    translateBtn.innerText = "Deciphering...";
    translateBtn.disabled = true;
    translateBtn.style.cursor = "wait";
    
    outputArea.style.color = "inherit"; // 重置颜色
    outputArea.style.opacity = "0.6";
    
    // 根据选择的模型显示不同的加载提示
    if (selectedModel === "byt5") {
        outputArea.innerText = "Consulting the Neural Network Scribe (AI)...";
    } else {
        outputArea.innerText = "Consulting the Standard Algorithm...";
    }

    try {
        // 4. 发送 POST 请求给后端
        const response = await fetch(API_URL, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            // 将文本和模型 ID 打包发送
            body: JSON.stringify({
                text: text,
                model_id: selectedModel 
            })
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        const data = await response.json();

        // 5. 显示翻译结果
        // 后端返回格式预期: { "translation": "...", "model_used": "...", ... }
        if (data.translation) {
             outputArea.innerText = data.translation;
        } else {
             // 兼容性处理
             outputArea.innerText = JSON.stringify(data);
        }

    } catch (error) {
        console.error("Translation Error:", error);
        outputArea.innerText = "Error: The scribe could not be reached. Please check your connection.";
        outputArea.style.color = "#8b0000";
    } finally {
        // 6. 恢复 UI 状态
        outputArea.style.opacity = "1";
        translateBtn.innerText = originalBtnText;
        translateBtn.disabled = false;
        translateBtn.style.cursor = "pointer";
    }
}