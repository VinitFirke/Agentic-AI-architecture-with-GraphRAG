"""Client for the qwen3-vl VLM served by Ollama (through LangChain), with timing metrics."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from thesis.knowledge.graphrag import NO_THINK_PREFILL


@dataclass
class VLMCall:
    """Result and timing of one VLM request."""

    text: str
    wall_ms: float  # client-side round trip (includes network and image upload)
    server_ms: float  # Ollama total_duration
    prompt_tokens: int
    output_tokens: int
    prompt_eval_ms: float
    eval_ms: float
    done_reason: str

    @property
    def tokens_per_s(self) -> float:
        """Generation throughput (output tokens / generation time)."""
        return self.output_tokens / (self.eval_ms / 1000) if self.eval_ms else 0.0

    @property
    def network_ms(self) -> float:
        """Round-trip time not spent inside Ollama (network, (de)serialisation)."""
        return max(self.wall_ms - self.server_ms, 0.0)


def encode_image(frame_bgr: np.ndarray, max_side: int = 768, quality: int = 85) -> str:
    """Resize so the longest side is max_side and return base64 JPEG."""
    h, w = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        frame_bgr = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("Could not JPEG-encode frame")
    return base64.b64encode(buf).decode("ascii")


class VLMClient:
    """Send an image + prompt to the VLM and return text with latency/throughput metrics."""

    def __init__(self, model: str, host: str, cfg: dict[str, Any]):
        self.cfg = cfg
        self.disable_thinking = cfg.get("disable_thinking", True)
        common = dict(
            model=model,
            base_url=host,
            temperature=cfg.get("temperature", 0.0),
            reasoning=False if self.disable_thinking else None,
            num_predict=cfg.get("num_predict", 300),
            client_kwargs={"timeout": cfg.get("timeout_s", 60)},
            keep_alive="30m",
        )
        self._llm = ChatOllama(**common)
        self._common = common

    def with_schema(self, schema: dict) -> ChatOllama:
        """A ChatOllama instance constrained to a JSON schema (Ollama structured output)."""
        return ChatOllama(**self._common, format=schema)

    def ask(
        self,
        system: str,
        prompt: str,
        image_b64: str | None = None,
        llm: ChatOllama | None = None,
    ) -> VLMCall:
        content: list[dict] = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append(
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}
            )
        msgs = [SystemMessage(system), HumanMessage(content=content)]
        if self.disable_thinking:
            msgs.append(AIMessage(NO_THINK_PREFILL))

        t0 = time.perf_counter()
        resp = (llm or self._llm).invoke(msgs)
        wall = (time.perf_counter() - t0) * 1000
        md = resp.response_metadata
        ns = 1e6
        return VLMCall(
            text=resp.content if isinstance(resp.content, str) else str(resp.content),
            wall_ms=wall,
            server_ms=(md.get("total_duration") or 0) / ns,
            prompt_tokens=md.get("prompt_eval_count") or 0,
            output_tokens=md.get("eval_count") or 0,
            prompt_eval_ms=(md.get("prompt_eval_duration") or 0) / ns,
            eval_ms=(md.get("eval_duration") or 0) / ns,
            done_reason=md.get("done_reason") or "",
        )
