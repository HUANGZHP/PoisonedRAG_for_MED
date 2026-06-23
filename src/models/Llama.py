# Purpose: Implements src/models/Llama.py in the PoisonedRAG project.

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .Model import Model


class Llama(Model):
    """Generic local HuggingFace causal-LM wrapper.

    Despite the class name, this loader is intentionally model-agnostic and can
    serve local Llama/Qwen/DeepSeek-style chat models by path.
    """

    def __init__(self, config):
        super().__init__(config)

        params = config.get("params", {})
        api_info = config.get("api_key_info", {})

        self.max_output_tokens = int(params.get("max_output_tokens", 256))
        self.device = str(params.get("device", "cuda"))
        self.use_chat_template = self._to_bool(params.get("use_chat_template", True))
        self.trust_remote_code = self._to_bool(params.get("trust_remote_code", True))
        self.device_map = params.get("device_map", None)
        self.torch_dtype = self._resolve_dtype(str(params.get("torch_dtype", "float16")))
        self.load_in_8bit = self._to_bool(params.get("load_in_8bit", False))
        self.load_in_4bit = self._to_bool(params.get("load_in_4bit", False))

        self.hf_token = self._resolve_hf_token(api_info)

        tok_kwargs = {
            "trust_remote_code": self.trust_remote_code,
        }
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": self.torch_dtype,
        }

        if self.hf_token:
            tok_kwargs["token"] = self.hf_token
            model_kwargs["token"] = self.hf_token

        if self.device_map:
            model_kwargs["device_map"] = self.device_map

        # Quantization: 8-bit takes priority over 4-bit; these require bitsandbytes.
        if self.load_in_8bit or self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            bnb_kwargs = {}
            if self.load_in_8bit:
                bnb_kwargs["load_in_8bit"] = True
            else:
                bnb_kwargs["load_in_4bit"] = True
            # bnb_4bit_compute_dtype defaults to torch_dtype
            if self.load_in_4bit:
                bnb_kwargs["bnb_4bit_compute_dtype"] = self.torch_dtype
            model_kwargs["quantization_config"] = BitsAndBytesConfig(**bnb_kwargs)
            # bitsandbytes handles device placement; don't pass torch_dtype
            model_kwargs.pop("torch_dtype", None)

        self.tokenizer = self._load_tokenizer_with_fix(tok_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, **model_kwargs)

        # --- Fix pad token ---
        # Llama 3.1-style tokenizers typically have pad_token==None.
        # Use bos_token as pad (distinct from eos_token) to avoid confusing
        # the model's stopping logic during generation.
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.bos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.bos_token_id
            else:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- Preserve model's eos_token_id list ---
        # generation_config may have [128001, 128009]; keep both.
        model_eos = self.model.generation_config.eos_token_id
        if isinstance(model_eos, list):
            self._stop_token_ids = list(model_eos)
        elif model_eos is not None:
            self._stop_token_ids = [model_eos]
        else:
            self._stop_token_ids = [self.tokenizer.eos_token_id]

        # Move model to target device only when not using device_map or quantization
        # (bitsandbytes / device_map handle placement automatically).
        _has_quant = "quantization_config" in model_kwargs
        if self.device_map is None and not _has_quant:
            target = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
            self.model = self.model.to(target)

        self.model.eval()

        try:
            self.input_device = next(self.model.parameters()).device
        except StopIteration:
            self.input_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _load_tokenizer_with_fix(self, tok_kwargs):
        """Load tokenizer, fixing empty special tokens if needed."""
        import tempfile, json, shutil
        from pathlib import Path
        try:
            return AutoTokenizer.from_pretrained(self.name, **tok_kwargs)
        except (RecursionError, RuntimeError):
            pass
        # Llama-1 models may have empty bos/eos/unk="" → fix in a temp copy
        tmp = Path(tempfile.mkdtemp())
        try:
            src = Path(self.name)
            for f in src.iterdir():
                if f.is_file() and not f.name.startswith('pytorch_model'):
                    shutil.copy2(f, tmp / f.name)
            tcfg = tmp / 'tokenizer_config.json'
            if tcfg.exists():
                d = json.loads(tcfg.read_text())
                for k, v in {'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>'}.items():
                    if k in d and (not d[k] or (isinstance(d[k], str) and d[k].strip() == '')):
                        d[k] = v
                d['model_max_length'] = 2048
                tcfg.write_text(json.dumps(d, indent=2))
            return AutoTokenizer.from_pretrained(str(tmp), **tok_kwargs)
        finally:
            shutil.rmtree(str(tmp), ignore_errors=True)

    @staticmethod
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(v)

    @staticmethod
    def _resolve_dtype(name: str):
        n = (name or "").strip().lower()
        if n in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if n in {"fp16", "float16", "half"}:
            return torch.float16
        if n in {"fp32", "float32"}:
            return torch.float32
        return torch.float16

    @staticmethod
    def _resolve_hf_token(api_info):
        env_token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
        if env_token:
            return env_token

        api_keys = api_info.get("api_keys", []) if isinstance(api_info, dict) else []
        api_pos = int(api_info.get("api_key_use", 0)) if isinstance(api_info, dict) else 0
        if isinstance(api_keys, list) and len(api_keys) > 0 and 0 <= api_pos < len(api_keys):
            token = str(api_keys[api_pos]).strip()
            return token or None
        return None

    def _build_inputs(self, msg: str):
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            templated = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            # transformers>=4.46 returns BatchEncoding (not Tensor), extract input_ids
            if isinstance(templated, torch.Tensor):
                return {"input_ids": templated.to(self.input_device)}
            if hasattr(templated, "input_ids"):
                return {"input_ids": templated.input_ids.to(self.input_device)}

        encoded = self.tokenizer(msg, return_tensors="pt")
        return {k: v.to(self.input_device) for k, v in encoded.items()}

    def query(self, msg):
        try:
            model_inputs = self._build_inputs(msg)

            gen_kwargs = {
                "max_new_tokens": self.max_output_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self._stop_token_ids,
            }

            if self.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self.temperature
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                outputs = self.model.generate(**model_inputs, **gen_kwargs)

            input_len = model_inputs["input_ids"].shape[-1]
            completion_ids = outputs[0][input_len:]
            result = self.tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
            return result
        except Exception as e:
            print(f"Local model query failed ({self.name}): {e}")
            return ""