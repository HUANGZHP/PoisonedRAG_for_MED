# Purpose: Implements src/models/GPT.py in the PoisonedRAG project.

import os
import re

from openai import OpenAI
from .Model import Model


class GPT(Model):
    def __init__(self, config):
        super().__init__(config)
        api_info = config.get("api_key_info", {})
        api_keys = api_info.get("api_keys", [])
        api_pos = int(api_info.get("api_key_use", 0))

        env_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("CHATANYWHERE_API_KEY", "").strip()
        cfg_key = ""
        if len(api_keys) > 0:
            assert (0 <= api_pos < len(api_keys)), "Please enter a valid API key to use"
            cfg_key = str(api_keys[api_pos]).strip()

        api_key = env_key or cfg_key
        if not api_key:
            raise ValueError(
                "No API key found for GPT provider. Set OPENAI_API_KEY (or CHATANYWHERE_API_KEY), "
                "or configure model_configs/*.json -> api_key_info.api_keys."
            )

        base_url = (
            os.environ.get("OPENAI_BASE_URL", "").strip()
            or os.environ.get("CHATANYWHERE_BASE_URL", "").strip()
            or str(api_info.get("base_url", "")).strip()
        )

        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _redact_error(msg: str) -> str:
        # Hide long token-like segments in exception messages.
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-***", msg)

    def query(self, msg):
        try:
            completion = self.client.chat.completions.create(
                model=self.name,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": msg}
                ],
            )
            response = completion.choices[0].message.content
           
        except Exception as e:
            print(f"LLM query failed: {self._redact_error(str(e))}")
            response = ""

        return response