# PandaLM Web App

這是一個基於 Streamlit 的網頁應用，用於選擇和比較不同的語言模型。使用者可以輸入指令和內容，並選擇兩個不同的模型來生成回應，最後使用 PandaLM 模型進行評估。

## 功能特點

- **模型選擇**：提供多個語言模型供使用者選擇。
- **生成回應**：根據使用者的指令和輸入內容生成回應。
- **回應比較**：使用 PandaLM 模型對兩個回應進行比較和評估。
- **自定義 CSS**：美化輸出框。

## 安裝

1. 克隆此倉庫到本地：

    ```sh
    git clone https://github.com/kee0812/pandalm-webapp.git
    cd llm-comparison-app
    ```

2. 創建並激活虛擬環境（可選）：

    ```sh
    python -m venv venv
    source venv/bin/activate  # 對於 Windows 系統，請使用 `venv\Scripts\activate`
    ```

3. 安裝所需的依賴包：

    ```sh
    pip install -r requirements.txt
    ```

## 運行應用

1. 在項目目錄下運行以下命令啟動 Streamlit 應用：

    ```sh
    streamlit run app.py
    ```

2. 打開瀏覽器並訪問 `http://localhost:8501`，即可看到應用界面。

## 使用說明

1. **輸入指令**：在輸入框中輸入指令。
2. **輸入內容**：在輸入框中輸入內容。
3. **選擇模型**：從下拉選單中選擇兩個不同的模型。
4. **運行模型**：點擊“運行模型”按鈕，應用將生成兩個模型的回應並顯示在界面上。
5. **查看評估結果**：應用將使用 PandaLM 模型對兩個回應進行比較和評估，並顯示評估結果。

## 程式碼說明

### 模型選項

應用提供了以下模型選項：

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Meta-Llama-3-8B-Instruct`
- `taide/TAIDE-LX-7B-Chat`
- `01-ai/Yi-1.5-6B-Chat`
- `yentinglin/Taiwan-LLM-7B-v2.1-chat`

### LLMModel 類

此類負責加載模型、生成回應和進行評估：

- `__init__(self, model_path)`: 初始化類並加載模型和分詞器。
- `load_model(self)`: 加載模型和分詞器，同時設置特殊標記。
- `generate_response(self, instruction, input_data)`: 根據指令和輸入生成回應。
- `evaluate(self, instruction, input_data, response1, response2)`: 用於 PandaLM 評估兩個模型生成的回應。
- `cleanup(self)`: 釋放資源。

### Streamlit 應用結構

- 標題和輸入框。
- 兩個選擇框用於選擇兩個模型。
- 運行模型按鈕，觸發計算，生成兩個模型的回應，並用 PandaLM 進行評估。

## 引用

如果您使用此應用程式，請引用 PandaLM 的論文：

- Wang, Y., Yu, Z., Zeng, Z., Yang, L., Wang, C., Chen, H., ... & Zhang, Y. (2023). Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. arXiv preprint arXiv:2306.05087. [論文鏈接](https://arxiv.org/abs/2306.05087)

## 注意事項

- 確保您的環境中安裝了 PyTorch 並配置好 CUDA，以利用 GPU 加速模型運行。
- 如果遇到模型加載失敗或生成回應失敗，應用會顯示相應的警告信息。

## 聯繫我們

如果有任何問題或建議，請聯繫我們：[xian081215@gmail.com]

