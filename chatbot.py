import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import gc

# 定義模型選項
model_options = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "taide/TAIDE-LX-7B-Chat",
    "01-ai/Yi-1.5-6B-Chat",
    "yentinglin/Taiwan-LLM-7B-v2.1-chat",
]

@st.cache_resource
class LLMModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer, self.model = self.load_model()
        
    def load_model(self):
        # 加載模型
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            st.warning(f"Error loading model: {e}")
            return None, None

        # 加載分詞器
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        except Exception as e:
            st.warning(f"Error loading tokenizer: {e}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        
        # 設置特殊標記
        if "llama" in self.model_path or "TAIDE" in self.model_path or "PandaLM" in self.model_path:
            tokenizer.add_special_tokens(
                {
                    "eos_token": "</s>",
                    "bos_token": "</s>",
                    "unk_token": "</s>",
                }
            )

        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        model.eval()

        return tokenizer, model

    def generate_response(self, instruction, input_data):
        if self.tokenizer is None or self.model is None:
            return "Model loading failed."
        
        # prompt = f"請依照以下指示用中文回答問題,字數不超過100字。### 指示:{instruction}, ### 問題:{input_data}"
        prompt = f"{instruction}:{input_data}"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation_config = GenerationConfig(
            temperature=0.6,
            top_p=0.95,
            top_k=50,
            num_beams=4,
            do_sample=True,
            early_stopping=True,
            repetition_penalty=1.2,
        )
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config,
                    max_new_tokens=256,
                )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            st.warning(f"Error during generation: {e}")
            response = "Generation failed."
        
        return response.split(f"{prompt}")[-1].strip()
    
    def evaluate(self, instruction, input_data, response1, response2):
        # 用於 PandaLM 評估
        rsp = f"Response 1:\n{response1}\n\nResponse 2:\n{response2}"
        input_sequence = f"Below are two responses for a given task. The task is defined by the Instruction with an Input that provides further context. Evaluate the responses, paying special attention to whether the responses are in Traditional Chinese, of good quality. Generate a short summary reference answer for the task within 100 words.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input_data}\n\n{rsp}\n\n### Evaluation:\n"
        inputs = self.tokenizer(input_sequence, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=GenerationConfig(
                    temperature=0.1,
                    top_p=1,
                    top_k=1,
                    num_beams=4,
                    do_sample=True,
                    early_stopping=True,
                    repetition_penalty=1.2,
                ),
                max_new_tokens=512,
            )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Evaluation:")[-1].strip()

    def cleanup(self):
        # 釋放資源
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()  # 如果使用了 GPU，這行代碼有助於釋放 CUDA 記憶體
        gc.collect()  # 強制進行垃圾回收

# Streamlit 應用程序標題
st.title("LLM 模型選擇網頁")

# 輸入指令
instruction = st.text_input("輸入指令", "")

# 輸入數據
input_data = st.text_area("輸入内容", "")

# 選擇模型
col1, col2 = st.columns(2)
model_option_1 = col1.selectbox("選擇第一個模型", model_options)
model_option_2 = col2.selectbox("選擇第二個模型", model_options)

# 添加自定義CSS
st.markdown(
    """
    <style>
    .output-box {
        border: 1px solid #d3d3d3;
        padding: 10px;
        border-radius: 5px;
        background-color: #f9f9f9;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# 按鈕觸發計算
if st.button("運行模型"):
    if instruction and input_data:
        # 加載和生成模型1的結果
        with st.spinner("正在生成模型1結果..."):
            model1 = LLMModel(model_option_1)
            result_1 = model1.generate_response(instruction, input_data)
            model1.cleanup()
        
        # 加載和生成模型2的結果
        with st.spinner("正在生成模型2結果..."):
            model2 = LLMModel(model_option_2)
            result_2 = model2.generate_response(instruction, input_data)
            model2.cleanup()
        
        # 顯示結果
        col1, col2 = st.columns(2)
        col1.text_area(f"模型 1 ({model_option_1}) 的結果：", result_1, height=200)
        col2.text_area(f"模型 2 ({model_option_2}) 的結果：", result_2, height=200)
        
        # PandaLM 評估
        with st.spinner("PandaLM 評估"):
            pandalm = LLMModel("WeOpenML/PandaLM-7B-v1")
            eval_result = pandalm.evaluate(instruction, input_data, result_1, result_2)
            st.text_area("PandaLM 評估結果：", eval_result, height=200)
            pandalm.cleanup()
    else:
        st.warning("請輸入指令和内容。")
