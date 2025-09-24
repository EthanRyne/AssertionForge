# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Add your own LLM client

%%writefile /content/AssertionForge/src/utils_LLM.py

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Add your own LLM client

# utils_LLM.py
from __future__ import annotations
from typing import Any, Dict, Optional, Callable
import os, time, json, re

# Optional fast tokenizer for OpenAI models
try:
    import tiktoken
except Exception:
    tiktoken = None  # it's okay if this is missing

# Use the repo's saver logger if available
try:
    from saver import saver
    log = saver.log_info
except Exception:
    log = print

# =========================
# Public API
# =========================

def get_llm(model_name: str, **llm_args) -> Any:
    name = (model_name or "").lower()

    if name == "callable":
        fn = llm_args.get("fn")
        if not callable(fn):
            raise TypeError("get_llm('callable') requires fn=<callable>.")
        return fn

    agent = {
        "provider": None,
        "model": llm_args.get("model"),
        "system": llm_args.get("system", "You are a hardware verification expert."),
        "temperature": float(llm_args.get("temperature", 0.0)),
        "max_tokens": int(llm_args.get("max_tokens", 1024)),
        "stop": llm_args.get("stop"),
        "metadata": llm_args.get("metadata", {}),
    }

    if name == "openai":
        client = llm_args.get("client")
        if client is None:
            try:
                from openai import OpenAI  # type: ignore
                client = OpenAI(
                          api_key='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',
                          base_url="https://openrouter.ai/api/v1")
            except Exception as e:
                raise RuntimeError("OpenAI client not provided and auto-import failed.") from e
        agent.update({"provider": "openai", "client": client})
        return agent

    if name == "anthropic":
        client = llm_args.get("client")
        if client is None:
            try:
                import anthropic  # type: ignore
                client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except Exception as e:
                raise RuntimeError("Anthropic client not provided and auto-import failed.") from e
        agent.update({"provider": "anthropic", "client": client})
        return agent

    if name in ("http", "rest", "vllm", "tgi"):
        base_url = llm_args.get("base_url")
        if not base_url:
            raise ValueError("get_llm('http') requires base_url=...")
        agent.update({
            "provider": "http",
            "base_url": base_url,
            "headers": llm_args.get("headers", {"Content-Type": "application/json"})
        })
        return agent

    # Heuristic: allow passing client+model without naming provider
    if llm_args.get("client") and agent["model"]:
        agent.update({"provider": "openai", "client": llm_args["client"]})
        return agent

    raise ValueError(f"Unsupported model/provider name: {model_name!r}")


def llm_inference(llm_agent: Any, prompt: str, tag: str) -> str:
    if llm_agent is None:
        raise ValueError("llm_inference: llm_agent is None")

    # Callable agent
    if callable(llm_agent):
        return _ensure_text(llm_agent(prompt, tag))

    # Dict agent (from get_llm)
    if isinstance(llm_agent, dict) and "provider" in llm_agent:
        prov = llm_agent["provider"]
        if prov == "openai":
            return _call_openai(llm_agent, prompt, tag)
        if prov == "anthropic":
            return _call_anthropic(llm_agent, prompt, tag)
        if prov == "http":
            return _call_http(llm_agent, prompt, tag)

    raise TypeError("llm_inference: Unsupported llm_agent. Use get_llm(...) or a callable.")


def count_prompt_tokens(text: str, model_name: Optional[str] = None) -> int:
    """
    Count approximate tokens for a prompt string.

    - If tiktoken is installed and model_name is a known OpenAI model,
      uses the exact tokenizer for that model/family.
    - Otherwise falls back to:
        1) 1 token ≈ 4 characters heuristic
        2) whitespace token count as a lower bound

    Args:
        text: prompt string
        model_name: optional (e.g., "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"). If omitted, uses a general tokenizer if available.

    Returns:
        int: estimated/actual token count
    """
    s = text if text is not None else ""
    s = s.strip()
    if not s:
        return 0

    # Prefer tiktoken if available
    if tiktoken is not None:
        enc = _get_tiktoken_encoding(model_name)
        if enc is not None:
            try:
                return len(enc.encode(s))
            except Exception:
                pass  # fall through to heuristics

    # Heuristic 1: 1 token ~ 4 chars (English-ish average)
    approx = max(1, int(round(len(s) / 4.0)))

    # Heuristic 2: whitespace token count as a floor
    ws_tokens = max(1, len(re.findall(r"\S+", s)))

    # Take the max to be conservative (avoid underestimation)
    return max(approx, ws_tokens)

# =========================
# Provider implementations
# =========================

def _call_openai(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    client = agent["client"]
    model = agent["model"]
    system = agent.get("system", "You are a helpful assistant.")
    temperature = agent.get("temperature", 0.0)
    max_tokens = agent.get("max_tokens", 1024)
    stop = agent.get("stop")
    meta = agent.get("metadata", {})

    if hasattr(client, "chat") and hasattr(client.chat, "completions"):
        return _retry(lambda: _extract_text_openai(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": f"{system} (tag={tag})"},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stop=stop,
            )
        ), f"openai.chat:{model}")

    if hasattr(client, "responses"):
        return _retry(lambda: _extract_text_openai(
            client.responses.create(
                model=model,
                input=prompt,
                metadata={**meta, "tag": tag},
                temperature=temperature,
                max_output_tokens=max_tokens,
                stop_sequences=stop,
                system=system,
            )
        ), f"openai.responses:{model}")

    raise RuntimeError("OpenAI client has no known methods (chat.completions/responses).")


def _call_anthropic(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    client = agent["client"]
    model = agent["model"]
    system = agent.get("system", "You are a helpful assistant.")
    temperature = agent.get("temperature", 0.0)
    max_tokens = agent.get("max_tokens", 1024)
    stop = agent.get("stop")

    if not (hasattr(client, "messages") and hasattr(client.messages, "create")):
        raise RuntimeError("Anthropic client lacks messages.create().")

    def do():
        resp = client.messages.create(
            model=model,
            system=f"{system} (tag={tag})",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop,
        )
        return _extract_text_anthropic(resp)

    return _retry(do, f"anthropic:{model}")


def _call_http(agent: Dict[str, Any], prompt: str, tag: str) -> str:
    import urllib.request
    base_url = agent["base_url"]
    headers = agent.get("headers", {"Content-Type": "application/json"})
    model = agent.get("model", "")
    system = agent.get("system", "You are a helpful assistant.")
    temperature = agent.get("temperature", 0.0)
    max_tokens = agent.get("max_tokens", 1024)
    stop = agent.get("stop")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": f"{system} (tag={tag})"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if stop:
        payload["stop"] = stop

    data = json.dumps(payload).encode("utf-8")

    def do():
        req = urllib.request.Request(base_url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))

        # OpenAI chat-compatible
        try:
            return obj["choices"][0]["message"]["content"]
        except Exception:
            pass
        # Text completions
        try:
            return obj["choices"][0]["text"]
        except Exception:
            pass
        return str(obj)

    return _retry(do, f"http:{base_url}")

# =========================
# Helpers
# =========================

def _retry(fn: Callable[[], str], tag: str, tries: int = 3, backoff: float = 1.5) -> str:
    last = None
    for i in range(1, tries + 1):
        try:
            out = fn()
            return _ensure_text(out)
        except Exception as e:
            last = e
            log(f"[llm_inference] ({tag}) attempt {i}/{tries} failed: {e}")
            if i < tries:
                time.sleep(backoff ** i)
    raise last  # type: ignore[misc]

def _ensure_text(x: Any) -> str:
    if isinstance(x, str):
        return x
    # OpenAI chat.completions
    try: return x.choices[0].message.content  # type: ignore[attr-defined]
    except Exception: pass
    # OpenAI responses
    try: return x.output_text  # type: ignore[attr-defined]
    except Exception: pass
    # Anthropic messages
    try:
        parts = []
        for block in getattr(x, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        if parts: return "".join(parts)
    except Exception:
        pass
    return str(x)

def _extract_text_openai(resp: Any) -> str:
    try: return resp.choices[0].message.content  # chat.completions
    except Exception: pass
    try: return resp.output_text  # responses API
    except Exception: pass
    return _ensure_text(resp)

def _extract_text_anthropic(resp: Any) -> str:
    parts = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", ""))
    return "".join(parts) if parts else _ensure_text(resp)

def _get_tiktoken_encoding(model_name: Optional[str]):
    if tiktoken is None:
        return None
    try:
        if model_name:
            # Known model → pick matching encoding if available
            return tiktoken.encoding_for_model(model_name)
        # Generic fallback (gpt-4o family is usually cl100k_base compatible)
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None
