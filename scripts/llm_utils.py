"""对抗文本生成共用的 LLM API 调用工具，使用 OpenAI 兼容接口调用 GPT-4.1-mini。"""

import json
import os
import time
import requests as req
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# API 配置（读取仓库共用模型配置）
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "model_configs" / "gpt4.1mini_config.json"
with _CONFIG_PATH.open(encoding="utf-8") as _config_file:
    _CONFIG = json.load(_config_file)

_api_info = _CONFIG.get("api_key_info") or {}
_api_keys = _api_info.get("api_keys") or []
_api_key_index = int(_api_info.get("api_key_use", 0))
_config_key = str(_api_keys[_api_key_index]).strip() if _api_keys else ""
API_KEY = os.environ.get("OPENAI_API_KEY", "").strip() or _config_key
if not API_KEY:
    raise RuntimeError(f"No usable API key configured in {_CONFIG_PATH}")

BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or str(_api_info.get("base_url") or "https://api.chatanywhere.tech/v1").strip()
MODEL_NAME = str((_CONFIG.get("model_info") or {}).get("name") or "gpt-4.1-mini").strip()
MAX_RETRIES = 5
TIMEOUT = 180


def call_llm(
    prompt: str,
    system_msg: str = "You are a helpful medical assistant.",
    temperature: float = 0.7,
    max_tokens: int = 2048,
    return_json: bool = False,
) -> Optional[str]:
    """调用 LLM API 并返回文本；请求失败时返回 ``None``。"""
    url = f"{BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload: dict = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    }
    if return_json:
        payload["response_format"] = {"type": "json_object"}

    last_err = ""
    for attempt in range(MAX_RETRIES):
        try:
            resp = req.post(url, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 429:
                wait = min(5 * (attempt + 1), 30)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return content
        except Exception as e:
            last_err = str(e)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    print(f"  [LLM ERROR] {last_err[:200]}")
    return None


# ---------------------------------------------------------------------------
# 提示词构造函数
# ---------------------------------------------------------------------------


def build_blackbox_prompt(query: str, positive: str) -> str:
    """构造黑盒攻击提示词，生成全新的误导性文本 I。"""
    return (
        f"You are helping build training data for a medical retrieval system.\n\n"
        f"Below is a medical QUESTION and the CORRECT medical CONTEXT (e.g., a PubMed abstract).\n\n"
        f"QUESTION:\n{query}\n\n"
        f"CORRECT CONTEXT:\n{positive}\n\n"
        f"Your task: Write a MISLEADING medical passage that:\n"
        f"1. Looks like a real PubMed abstract / medical text (same style, same approximate length as the correct context).\n"
        f"2. Contains keywords and phrases from the QUESTION so it will be retrieved by a dense retriever.\n"
        f"3. Is SEMANTICALLY WRONG — it should support an incorrect answer or contradict the correct context.\n"
        f"4. Does NOT simply copy the correct context with minor edits — it should be a DIFFERENT text.\n\n"
        f"Output ONLY the misleading passage. No prefixes, no explanations, no markdown formatting."
    )
