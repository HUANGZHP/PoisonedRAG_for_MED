# Purpose: Implements src/models/GPT.py in the PoisonedRAG project.

import os
import re
import time

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

        params = config.get("params", {})
        self.max_output_tokens = int(params["max_output_tokens"])
        self.reasoning_effort = str(params.get("reasoning_effort", "")).strip().lower()
        self.empty_response_retries = max(0, int(params.get("empty_response_retries", 0)))
        self.empty_response_retry_max_output_tokens = max(
            self.max_output_tokens,
            int(params.get("empty_response_retry_max_output_tokens", self.max_output_tokens)),
        )
        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    @staticmethod
    def _redact_error(msg: str) -> str:
        # Hide long token-like segments in exception messages.
        return re.sub(r"sk-[A-Za-z0-9_-]{12,}", "sk-***", msg)

    @staticmethod
    def _usage_value(completion, name: str, default=None):
        """Read optional SDK usage fields without coupling to a client version."""
        usage = getattr(completion, "usage", None)
        if usage is None:
            return default
        value = getattr(usage, name, None)
        return default if value is None else value

    @staticmethod
    def _reasoning_tokens(completion):
        usage = getattr(completion, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        value = getattr(details, "reasoning_tokens", None) if details is not None else None
        return 0 if value is None else value

    def _chat_completion_kwargs(self, max_output_tokens: int):
        kwargs = {
            "model": self.name,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
        }
        # GPT-5 is a reasoning family. Its output ceiling includes invisible
        # reasoning tokens, so the deprecated max_tokens can yield a completed
        # request with no visible answer. Keep legacy GPT-4.1 experiments
        # byte-for-byte parameter-compatible while using the GPT-5 API fields.
        if self.name.strip().lower().startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_output_tokens
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["max_tokens"] = max_output_tokens
        return kwargs

    def query(self, msg):
        attempts = self.empty_response_retries + 1
        for attempt in range(attempts):
            max_output_tokens = (
                self.max_output_tokens
                if attempt == 0
                else self.empty_response_retry_max_output_tokens
            )
            try:
                kwargs = self._chat_completion_kwargs(max_output_tokens)
                kwargs["messages"].append({"role": "user", "content": msg})
                completion = self.client.chat.completions.create(**kwargs)
                choice = completion.choices[0]
                response = str(getattr(choice.message, "content", "") or "").strip()
                if response:
                    return response

                finish_reason = getattr(choice, "finish_reason", None)
                completion_tokens = self._usage_value(completion, "completion_tokens")
                reasoning_tokens = self._reasoning_tokens(completion)
                print(
                    "LLM empty response: "
                    f"attempt={attempt + 1}/{attempts}, finish_reason={finish_reason}, "
                    f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}, "
                    f"max_output_tokens={max_output_tokens}"
                )
            except Exception as e:
                print(
                    f"LLM query failed: attempt={attempt + 1}/{attempts}, "
                    f"{self._redact_error(str(e))}"
                )

            if attempt + 1 < attempts:
                time.sleep(min(2 ** attempt, 4))

        return ""
