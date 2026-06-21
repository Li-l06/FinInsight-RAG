import os
from langchain_qwq import ChatQwen
from app.config import config


def create_chat_model(streaming: bool = False):
    print(f"[DEBUG] DASHSCOPE_API_KEY in os.environ: {'DASHSCOPE_API_KEY' in os.environ}")
    print(f"[DEBUG] DASHSCOPE_API_BASE: {os.environ.get('DASHSCOPE_API_BASE', 'NOT SET')}")

    model = ChatQwen(
        model=config.dashscope_model,
        temperature=0.7,
        streaming=streaming,
    )
    print(f"[DEBUG] model.api_base: {model.api_base}")
    print(f"[DEBUG] model.api_key is None: {model.api_key is None}")
    if model.api_key:
        print(f"[DEBUG] api_key prefix: {model.api_key.get_secret_value()[:25]}...")
    return model


def create_streaming_chat_model():
    return create_chat_model(streaming=True)