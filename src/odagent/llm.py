"""LLM backends. ``HFBackend`` runs an open-weight chat model locally with transformers
(batched, greedy by default); ``FakeBackend`` is a deterministic stand-in for tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

Messages = list[dict[str, str]]


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    seconds: float  # wall time of the batch divided by the batch size


class Backend(Protocol):
    name: str

    def generate(
        self, prompts: list[Messages], max_new_tokens: int = 200, temperature: float = 0.0
    ) -> list[Generation]: ...


class FakeBackend:
    """Calls ``fn(messages) -> str`` for every prompt; token counts are word counts."""

    def __init__(self, fn: Callable[[Messages], str], name: str = "fake") -> None:
        self.fn = fn
        self.name = name

    def generate(self, prompts, max_new_tokens=200, temperature=0.0):
        out = []
        for msgs in prompts:
            text = self.fn(msgs)
            n_in = sum(len(m["content"].split()) for m in msgs)
            out.append(Generation(text, n_in, len(text.split()), 0.0))
        return out


class HFBackend:
    def __init__(self, model_id: str, batch_size: int = 16, dtype: str = "bfloat16") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.name = model_id
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype) if device == "cuda" else torch.float32
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, prompts, max_new_tokens=200, temperature=0.0):
        results: list[Generation] = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i : i + self.batch_size]
            texts = [
                self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                for m in chunk
            ]
            enc = self.tok(texts, return_tensors="pt", padding=True).to(self.device)
            kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
            if temperature > 0:
                kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
            else:
                kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            t0 = time.perf_counter()
            with self.torch.inference_mode():
                out = self.model.generate(**enc, **kwargs)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            per_item = (time.perf_counter() - t0) / len(chunk)
            new = out[:, enc["input_ids"].shape[1] :]
            for row, attn in zip(new, enc["attention_mask"], strict=True):
                n_out = int((row != self.tok.pad_token_id).sum())
                text = self.tok.decode(row, skip_special_tokens=True)
                results.append(Generation(text, int(attn.sum()), n_out, per_item))
        return results
