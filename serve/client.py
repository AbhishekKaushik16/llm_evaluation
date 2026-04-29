#!/usr/bin/env python3
"""Async Python client for the vLLM OpenAI-compatible API.

Supports:
  - Streaming token generation (SSE)
  - Configurable decoding parameters (max_tokens, temperature, top_p, stop)
  - Concurrent request dispatch
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import aiohttp


@dataclass
class GenerationResult:
    text: str
    tokens_generated: int
    elapsed_seconds: float
    finish_reason: str | None = None


@dataclass
class VLLMClient:
    """Async client for vLLM's OpenAI-compatible /v1/completions endpoint."""

    base_url: str = "http://localhost:8000"
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    timeout: float = 120.0
    _session: aiohttp.ClientSession | None = field(default=None, repr=False)

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/completions"

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Non-streaming generation: returns the full result at once."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        session = await self._get_session()
        t0 = time.perf_counter()
        async with session.post(self.completions_url, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
        elapsed = time.perf_counter() - t0

        choice = data["choices"][0]
        return GenerationResult(
            text=choice["text"],
            tokens_generated=data["usage"]["completion_tokens"],
            elapsed_seconds=elapsed,
            finish_reason=choice.get("finish_reason"),
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[str]:
        """Streaming generation: yields tokens as they arrive via SSE."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        session = await self._get_session()
        async with session.post(self.completions_url, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                data_str = decoded[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def generate_stream_full(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> GenerationResult:
        """Streaming generation that collects all tokens and returns a full result with timing."""
        tokens: list[str] = []
        t0 = time.perf_counter()
        ttft: float | None = None
        async for token in self.generate_stream(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            seed=seed,
        ):
            if ttft is None:
                ttft = time.perf_counter() - t0
            tokens.append(token)
        elapsed = time.perf_counter() - t0
        return GenerationResult(
            text="".join(tokens),
            tokens_generated=len(tokens),
            elapsed_seconds=elapsed,
        )

    async def generate_concurrent(
        self,
        prompts: list[str],
        **kwargs,
    ) -> list[GenerationResult]:
        """Fire N concurrent requests and return all results."""
        tasks = [self.generate(p, **kwargs) for p in prompts]
        return await asyncio.gather(*tasks)

    async def __aenter__(self):
        await self._get_session()
        return self

    async def __aexit__(self, *exc):
        await self.close()
