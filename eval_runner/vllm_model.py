#!/usr/bin/env python3
"""Cached model wrapper for lm-evaluation-harness.

Wraps the vLLM OpenAI-compatible API with a SQLite prompt cache so that
repeated evaluations are deterministic and skip redundant API calls.
Implements the lm_eval LM interface for seamless harness integration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import requests
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

logger = logging.getLogger(__name__)

CACHE_DB_PATH = Path(__file__).parent / "cache.db"


class PromptCache:
    """SQLite-backed prompt cache keyed on request parameters."""

    def __init__(self, db_path: str | Path = CACHE_DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache "
            "(key TEXT PRIMARY KEY, response TEXT, created_at REAL)"
        )
        self.conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _make_key(**kwargs) -> str:
        raw = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, **kwargs) -> dict | None:
        key = self._make_key(**kwargs)
        row = self.conn.execute(
            "SELECT response FROM cache WHERE key = ?", (key,)
        ).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put(self, response: dict, **kwargs) -> None:
        key = self._make_key(**kwargs)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, response, created_at) VALUES (?, ?, ?)",
            (key, json.dumps(response), time.time()),
        )
        self.conn.commit()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0.0,
        }

    def close(self) -> None:
        self.conn.close()


@register_model("cached_vllm")
class CachedVLLMModel(LM):
    """lm-eval model that queries vLLM's OpenAI-compatible /v1/completions
    endpoint with a local SQLite cache layer."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1/completions",
        model: str = "Qwen/Qwen2.5-3B-Instruct",
        pretrained: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_db: str | None = None,
        batch_size: int | str = 1,
        max_batch_size: int | None = None,
        max_length: int = 4096,
        **kwargs,
    ):
        super().__init__()
        self._base_url = base_url
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._batch_size = int(batch_size) if batch_size else 1
        self._max_length = max_length
        self._cache = PromptCache(cache_db or CACHE_DB_PATH)

    @property
    def eot_token_id(self):
        return None

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_tokens

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return "cpu"

    def tok_encode(self, string: str, **kwargs) -> list[int]:
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens: list[int], **kwargs) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def _call_api(self, prompt: str, max_tokens: int, echo: bool = False,
                  logprobs: int | None = None,
                  prompt_logprobs: int | None = None) -> dict:
        cache_kwargs = dict(
            prompt=prompt, model=self._model, max_tokens=max_tokens,
            temperature=self._temperature, top_p=self._top_p,
            echo=echo, logprobs=logprobs, prompt_logprobs=prompt_logprobs,
        )
        cached = self._cache.get(**cache_kwargs)
        if cached is not None:
            return cached

        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
        }
        if echo:
            payload["echo"] = True
        if logprobs is not None:
            payload["logprobs"] = logprobs
        if prompt_logprobs is not None:
            # vLLM-specific extension: required alongside echo so the server
            # actually computes prompt-side logprobs. Without this, vLLM's
            # OpenAI server splices an empty prompt_logprobs list into the
            # response and crashes with IndexError in _create_completion_logprobs.
            payload["prompt_logprobs"] = prompt_logprobs

        resp = requests.post(self._base_url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        self._cache.put(data, **cache_kwargs)
        return data

    # vLLM's OpenAI server (Metal build) rejects max_tokens=0 and can hit an
    # IndexError in _create_completion_logprobs when generation is empty. We
    # therefore generate exactly one throwaway token and trim it off before
    # scoring the prompt-side logprobs we actually care about.
    _SCORING_MAX_TOKENS = 1

    def loglikelihood(self, requests_list: list) -> list[tuple[float, bool]]:
        results = []
        for req in requests_list:
            context, continuation = req.args
            full_prompt = context + continuation

            data = self._call_api(
                full_prompt,
                max_tokens=self._SCORING_MAX_TOKENS,
                echo=True,
                logprobs=1,
                prompt_logprobs=1,
            )
            choice = data["choices"][0]

            logprobs_data = choice.get("logprobs") or {}
            token_logprobs = list(logprobs_data.get("token_logprobs") or [])
            tokens = list(logprobs_data.get("tokens") or [])
            top_logprobs = list(logprobs_data.get("top_logprobs") or [])

            # Drop the throwaway generated tokens so we score the prompt only.
            if self._SCORING_MAX_TOKENS > 0:
                trim = self._SCORING_MAX_TOKENS
                token_logprobs = token_logprobs[:-trim] if len(token_logprobs) > trim else token_logprobs
                tokens = tokens[:-trim] if len(tokens) > trim else tokens
                top_logprobs = top_logprobs[:-trim] if len(top_logprobs) > trim else top_logprobs

            # Estimate where the continuation starts by reconstructing from tokens
            ctx_len = len(context)
            reconstructed = ""
            cont_start_idx = 0
            for i, tok in enumerate(tokens):
                reconstructed += tok
                if len(reconstructed) >= ctx_len and cont_start_idx == 0:
                    cont_start_idx = i + 1

            cont_logprobs = token_logprobs[cont_start_idx:]
            ll = sum(lp for lp in cont_logprobs if lp is not None)

            # Check if continuation is the greedy (most likely) choice
            is_greedy = True
            for i in range(cont_start_idx, len(tokens)):
                if i < len(top_logprobs) and top_logprobs[i]:
                    top_token = max(top_logprobs[i], key=top_logprobs[i].get)
                    if top_token != tokens[i]:
                        is_greedy = False
                        break

            results.append((ll, is_greedy))
        return results

    def loglikelihood_rolling(self, requests_list: list) -> list[tuple[float, bool]]:
        results = []
        for req in requests_list:
            (text,) = req.args
            data = self._call_api(
                text,
                max_tokens=self._SCORING_MAX_TOKENS,
                echo=True,
                logprobs=1,
                prompt_logprobs=1,
            )
            choice = data["choices"][0]
            logprobs_data = choice.get("logprobs") or {}
            token_logprobs = list(logprobs_data.get("token_logprobs") or [])

            if self._SCORING_MAX_TOKENS > 0 and len(token_logprobs) > self._SCORING_MAX_TOKENS:
                token_logprobs = token_logprobs[: -self._SCORING_MAX_TOKENS]

            ll = sum(lp for lp in token_logprobs if lp is not None)
            results.append((ll, True))
        return results

    def generate_until(self, requests_list: list) -> list[str]:
        results = []
        for req in requests_list:
            context = req.args[0]
            gen_kwargs = req.kwargs if hasattr(req, "kwargs") else {}
            until = gen_kwargs.get("until", gen_kwargs.get("stop", []))
            max_tokens = gen_kwargs.get("max_gen_toks", self._max_tokens)

            payload: dict[str, Any] = {
                "model": self._model,
                "prompt": context,
                "max_tokens": max_tokens,
                "temperature": self._temperature,
                "top_p": self._top_p,
            }
            if until:
                payload["stop"] = until if isinstance(until, list) else [until]

            cache_kwargs = dict(
                prompt=context, model=self._model, max_tokens=max_tokens,
                temperature=self._temperature, top_p=self._top_p,
                stop=str(until),
            )
            cached = self._cache.get(**cache_kwargs)
            if cached is not None:
                data = cached
            else:
                resp = requests.post(self._base_url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                self._cache.put(data, **cache_kwargs)

            text = data["choices"][0]["text"]
            results.append(text)
        return results

    def __del__(self):
        if hasattr(self, "_cache"):
            logger.info("Cache stats: %s", self._cache.stats())
            self._cache.close()
