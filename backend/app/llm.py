from langchain_openai import ChatOpenAI

from .config import Settings


def build_llm(settings: Settings):
    if not settings.api_key.get_secret_value():
        raise ValueError("Add VULTR_INFERENCE_API_KEY to backend/back.env before chatting.")
    return ChatOpenAI(
        model=settings.model,
        base_url=settings.base_url,
        api_key=settings.api_key.get_secret_value(),
        temperature=settings.temperature,
        top_p=settings.top_p,
        max_tokens=settings.max_tokens,
        timeout=settings.llm_timeout,
        max_retries=0,
        use_responses_api=False,
        stream_usage=False,
    )
