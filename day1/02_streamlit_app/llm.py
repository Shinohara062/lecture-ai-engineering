# llm.py
import os
import torch
import streamlit as st
import time
from config import MODEL_NAME
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

# モデルをキャッシュして再利用
@st.cache_resource
def load_model():
    """LLMモデルをロードする"""
    try:
        # Hugging Face アクセストークンを取得
        hf_token = st.secrets["huggingface"]["token"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")

        # トークナイザーとモデルの読み込み
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, use_auth_token=hf_token)
        model.to(device)

        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return model, tokenizer

    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None, None

def generate_response(model, tokenizer, user_question):
    """LLMを使用して質問に対する回答を生成する"""
    if model is None or tokenizer is None:
        return "モデルがロードされていないため、回答を生成できません。", 0

    try:
        start_time = time.time()

        # rinnaのフォーマット例に合わせてプロンプトを構築
        prompt = f"ユーザー: {user_question}\nシステム: "

        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        # 生成
        output = model.generate(
            input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

        # プロンプト部分を除去して回答を抽出
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()

        response_time = time.time() - start_time
        return response, response_time

    except Exception as e:
        st.error(f"回答生成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}", 0
