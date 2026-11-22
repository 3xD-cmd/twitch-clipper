"""
gemini_gateway.py
Quota aware gateway to the Gemini API (google-genai SDK) for the TwitchClipper pipeline.

What it does
  * Discovers usable models dynamically (same exclusion list as before) and ranks them
    per route: scout (Flash, then Lite), critic and rules (Pro, then Flash, never Lite).
  * Keeps a persistent state file on Drive (memory/gemini_state.json):
      ledger   requests per model per Pacific day (Gemini RPD resets at midnight Pacific)
      limits   real RPD / RPM / TPM learned from 429 bodies (quotaValue)
      health   models parked after 404, daily quota, or zero free quota
      thinking which thinking config each model accepts (level, budget, none)
      schema   whether a model accepted response_schema
      aliases  "latest" aliases mapped to the concrete model whose quota they share
  * Classifies every error and reacts:
      404                    park 7 days, next model
      429 PerDay             park until next Pacific midnight (7 days if quotaValue is 0)
      429 PerMinute          wait server delay or 60/RPM, retry same model (max 2)
      bare 429, 5xx, timeout backoff 5 s, 15 s, 40 s plus jitter on the same model, then next
      400 thinking           switch thinking mode and retry, cached per model
      400 schema             drop response_schema for that model, retry
    The retryDelay of a daily quota error is ignored (it is misleading).
  * Paces calls per model (RPM spacing and a TPM window).
  * Uses the SDK HTTP timeout (no abandoned threads) and disables SDK internal retries.
  * Structured output with Pydantic (response_schema), then a local repair chain:
    fences, outer JSON extraction, trailing commas, smart quotes, truncated output
    closing, Python literals, shape wrapping, caller normalizer, lenient item validation.
  * Raises QuotaPause (reason "quota" or "busy") so a stage can checkpoint and stop cleanly.

Public contract
  GeminiGateway(api_key, state_path)
    .generate(route, contents, schema=None, thinking=None, timeout_s=None, ...) -> GatewayResult
    .plan({"scout": 4, "critic": 4}) -> Budget
    .affordable(route) -> int
    .usage_report_ar() -> str
  audio_part(data, mime_type) -> Part
  parse_json_loose(text) -> obj
  Exceptions: QuotaPause, ParseFailure, GatewayError
"""

from __future__ import annotations

import ast
import datetime as _dt
import json
import random
import re
import time
import types as pytypes
import typing
from collections import abc as _abc
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ValidationError

from google import genai
from google.genai import types

try:
    from google.genai import errors as genai_errors
except Exception:  # pragma: no cover
    genai_errors = None  # type: ignore[assignment]

from common import (
    UTC,
    atomic_write_json,
    format_duration_ar,
    load_json_safe,
    log,
    next_pacific_midnight_utc,
    now_utc,
    pacific_date_str,
    pacific_reset_riyadh,
    short_error,
)


# ============================================================================
# Configuration
# ============================================================================

# Battle tested exclusion list (kept) plus a few non text families.
EXCLUDED_MODEL_KEYWORDS = (
    "tts", "image", "audio", "embedding", "live", "veo", "imagen", "lyria", "robotics",
    "computer-use", "banana", "aqa", "deep-research",
)

SEED_MODELS = (
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
)

ROUTES = {
    "scout": {"tiers": ("flash", "lite"), "thinking": "low", "timeout_s": 150, "name_ar": "الكشاف"},
    "critic": {"tiers": ("pro", "flash"), "thinking": "high", "timeout_s": 240, "name_ar": "الناقد"},
    "rules": {"tiers": ("pro", "flash"), "thinking": "high", "timeout_s": 180, "name_ar": "مدرب القواعد"},
    "general": {"tiers": ("flash", "lite", "pro"), "thinking": "low", "timeout_s": 120, "name_ar": "المساعد"},
}

DEFAULT_LIMITS = {"rpm": 5, "rpd": 20, "tpm": 250_000}

TRANSIENT_BACKOFF_S = (5.0, 15.0, 40.0)
MAX_PER_MINUTE_RETRIES = 2
MAX_THINKING_SWITCHES = 2
NOT_FOUND_PARK_DAYS = 7
ZERO_QUOTA_PARK_DAYS = 7
FORBIDDEN_PARK_DAYS = 1
DISCOVERY_TTL_S = 12 * 3600
LEDGER_KEEP_DAYS = 7
DEFAULT_MAX_BUSY_WAIT_S = 600.0

THINKING_BUDGETS = {"low": 1024, "medium": 4096, "high": 8192}
THINKING_MODES = ("level", "budget", "none")

AUDIO_TOKENS_PER_SEC = 25.0
ASSUMED_AUDIO_KBPS = 32.0


# ============================================================================
# Exceptions and results
# ============================================================================

class GatewayError(Exception):
    """Non recoverable failure for this request (bad key, every model rejected the request)."""


class QuotaPause(Exception):
    """Stop cleanly and resume later.
    reason "quota": every model of the route is out of daily quota (reset_at is set).
    reason "busy":  every model stayed overloaded for the whole waiting budget (reset_at is None)."""

    def __init__(self, route: str, reason: str, reset_at: Optional[_dt.datetime], message_ar: str):
        super().__init__(message_ar)
        self.route = route
        self.reason = reason
        self.reset_at = reset_at
        self.message_ar = message_ar


class ParseFailure(Exception):
    """The model answered, but the answer could not be parsed or validated even after
    local repair and one retry. The caller should save the chunk as pending."""

    def __init__(self, route: str, model: str, raw_text: str, detail: str):
        super().__init__(f"{route}/{model}: {detail}")
        self.route = route
        self.model = model
        self.raw_text = raw_text
        self.detail = detail


@dataclass
class GatewayResult:
    data: Any
    raw_text: str
    model: str
    route: str
    attempts: int
    elapsed_s: float
    usage: dict = field(default_factory=dict)
    schema_used: bool = False
    dropped_items: int = 0


@dataclass
class Budget:
    ok: bool
    routes: dict
    estimated: bool
    text_ar: str


@dataclass
class _ErrInfo:
    kind: str
    message: str
    counted: bool = False
    retry_delay: Optional[float] = None
    quota_scope: Optional[str] = None      # "day" or "minute"
    quota_metric: Optional[str] = None     # "requests" or "tokens"
    quota_value: Optional[int] = None
    quota_model: Optional[str] = None


@dataclass
class _Outcome:
    kind: str
    attempts: int
    result: Optional[GatewayResult] = None
    message: str = ""
    raw: str = ""
    model: str = ""


# ============================================================================
# Model name helpers
# ============================================================================

_VERSION_RE = re.compile(r"gemini-(\d+(?:\.\d+)?)")


def model_version(name: str) -> Optional[float]:
    match = _VERSION_RE.search(name or "")
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def model_tier(name: str) -> Optional[str]:
    low = (name or "").lower()
    if "gemini" not in low:
        return None
    if "lite" in low:
        return "lite"
    if "pro" in low:
        return "pro"
    if "flash" in low:
        return "flash"
    return None


def is_preview(name: str) -> bool:
    low = (name or "").lower()
    return "preview" in low or "-exp" in low or "experimental" in low


def is_alias(name: str) -> bool:
    return model_version(name) is None


def is_excluded(name: str) -> bool:
    low = (name or "").lower()
    return any(bad in low for bad in EXCLUDED_MODEL_KEYWORDS)


def _model_sort_key(name: str) -> tuple:
    version = model_version(name)
    return (
        1 if version is None else 0,
        -(version or 0.0),
        1 if is_preview(name) else 0,
        len(name),
        name,
    )


def audio_part(data: bytes, mime_type: str = "audio/mp3"):
    """Inline audio part for multimodal prompts (critic)."""
    return types.Part.from_bytes(data=bytes(data), mime_type=mime_type)


# ============================================================================
# JSON repair chain
# ============================================================================

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_SMART_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _scan_json(text: str) -> tuple:
    """Return (block, open_stack, in_string) for the first JSON object or array in text.
    If the block is complete, open_stack is empty."""
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start < 0:
        return "", [], False
    stack: list = []
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if stack:
                stack.pop()
            if not stack:
                return text[start:i + 1], [], False
    return text[start:], stack, in_str


def _close_truncated(block: str, stack: list, in_str: bool) -> str:
    out = block
    if in_str:
        out += '"'
    out = re.sub(r"[,:\s]+$", "", out)
    out = re.sub(r',\s*"[^"]*"\s*$', "", out)
    for opener in reversed(stack):
        out += "}" if opener == "{" else "]"
    return out


def parse_json_loose(raw: Any) -> Any:
    """Parse model output into a Python object, repairing common defects.
    Raises ValueError when nothing works."""
    if raw is None:
        raise ValueError("empty response")
    if not isinstance(raw, str):
        return raw
    text = _strip_fences(raw)
    if not text:
        raise ValueError("empty response")
    candidates = [text]
    block, stack, in_str = _scan_json(text)
    if block:
        candidates.append(block)
        if stack or in_str:
            candidates.append(_close_truncated(block, stack, in_str))
    seen = set()
    last_err: Optional[Exception] = None
    for cand in candidates:
        for variant in (cand, cand.translate(_SMART_QUOTES)):
            for fixed in (variant, _TRAILING_COMMA_RE.sub(r"\1", variant)):
                if fixed in seen:
                    continue
                seen.add(fixed)
                try:
                    return json.loads(fixed)
                except ValueError as exc:
                    last_err = exc
    for cand in candidates[1:] or candidates:
        try:
            value = ast.literal_eval(_TRAILING_COMMA_RE.sub(r"\1", cand))
            if isinstance(value, (dict, list)):
                return value
        except (ValueError, SyntaxError):
            continue
    raise ValueError(f"unparseable JSON ({short_error(last_err) if last_err else 'no JSON found'})")


# ============================================================================
# Schema helpers (Pydantic v2)
# ============================================================================

def _unwrap_optional(annotation: Any) -> Any:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is getattr(pytypes, "UnionType", None):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _list_fields(schema: type) -> list:
    """[(field_name, item_model_or_None)] for every list field of a Pydantic model."""
    out = []
    for name, info in schema.model_fields.items():
        ann = _unwrap_optional(info.annotation)
        if typing.get_origin(ann) in (list, _abc.Sequence):
            args = typing.get_args(ann)
            item = args[0] if args else None
            item_model = item if isinstance(item, type) and issubclass(item, BaseModel) else None
            out.append((name, item_model))
    return out


def _primary_list_field(schema: type) -> Optional[str]:
    """The list field a bare list most likely belongs to: required fields first,
    then fields whose items are models, then declaration order."""
    lists = _list_fields(schema)
    if not lists:
        return None
    ranked = sorted(
        enumerate(lists),
        key=lambda pair: (
            0 if schema.model_fields[pair[1][0]].is_required() else 1,
            0 if pair[1][1] is not None else 1,
            pair[0],
        ),
    )
    return ranked[0][1][0]


def wrap_to_schema(obj: Any, schema: type) -> Any:
    """Generic shape fixes: a bare list becomes the schema's primary list field; a dict
    missing that field but holding exactly one unknown list value is renamed to it."""
    target = _primary_list_field(schema)
    if target is None:
        return obj
    if isinstance(obj, list):
        return {target: obj}
    if isinstance(obj, dict) and target not in obj:
        known = set(schema.model_fields.keys())
        list_keys = [k for k, v in obj.items() if isinstance(v, list) and k not in known]
        if len(list_keys) == 1:
            fixed = dict(obj)
            fixed[target] = fixed.pop(list_keys[0])
            return fixed
    return obj


def validate_lenient(schema: type, obj: Any) -> tuple:
    """Validate; if the whole object fails, drop only the list items that fail.
    Returns (instance, dropped_count). Raises ValidationError if still invalid."""
    try:
        return schema.model_validate(obj), 0
    except ValidationError as first_err:
        if not isinstance(obj, dict):
            raise first_err
        fixed = dict(obj)
        dropped = 0
        for name, item_model in _list_fields(schema):
            items = fixed.get(name)
            if item_model is None or not isinstance(items, list):
                continue
            kept = []
            for item in items:
                try:
                    item_model.model_validate(item)
                    kept.append(item)
                except ValidationError:
                    dropped += 1
            fixed[name] = kept
        return schema.model_validate(fixed), dropped


# ============================================================================
# Gateway
# ============================================================================

class GeminiGateway:
    def __init__(
        self,
        api_key: str,
        state_path: str,
        max_busy_wait_s: float = DEFAULT_MAX_BUSY_WAIT_S,
        verbose: bool = True,
    ):
        if not api_key or len(str(api_key)) < 10:
            raise GatewayError("❌ مفتاح GEMINI_API_KEY غير موجود أو غير صالح")
        self.api_key = str(api_key)
        self.state_path = state_path
        self.max_busy_wait_s = float(max_busy_wait_s)
        self.verbose = verbose
        self._clients: dict = {}
        self._recent: dict = {}  # bucket -> list[(timestamp, est_tokens)]
        self.state = self._load_state()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> dict:
        state = load_json_safe(self.state_path, default={}, expected_types=(dict,), quiet=True) or {}
        for key in ("ledger", "limits", "health", "thinking", "schema", "aliases"):
            if not isinstance(state.get(key), dict):
                state[key] = {}
        if not isinstance(state.get("models_cache"), dict):
            state["models_cache"] = {}
        state["version"] = 1
        self._prune_ledger(state)
        return state

    @staticmethod
    def _prune_ledger(state: dict) -> None:
        keep = sorted(state["ledger"].keys())[-LEDGER_KEEP_DAYS:]
        state["ledger"] = {k: state["ledger"][k] for k in keep}

    def _save_state(self) -> None:
        try:
            atomic_write_json(self.state_path, self.state, backup=True)
        except Exception as exc:
            log(f"⚠️ تعذر حفظ حالة Gemini في Drive: {short_error(exc)}")

    # ---------------------------------------------------------------- logging

    def _log(self, msg: str) -> None:
        if self.verbose:
            log(msg)

    # ---------------------------------------------------------------- clients

    def _client(self, timeout_s: float):
        key = int(max(10.0, float(timeout_s)) * 1000)
        client = self._clients.get(key)
        if client is not None:
            return client
        http_kwargs: dict = {"timeout": key}
        try:
            http_kwargs["retry_options"] = types.HttpRetryOptions(attempts=1)
        except Exception:
            pass
        try:
            client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(**http_kwargs))
        except Exception:
            try:
                client = genai.Client(api_key=self.api_key, http_options=types.HttpOptions(timeout=key))
            except Exception:
                client = genai.Client(api_key=self.api_key)
        self._clients[key] = client
        return client

    # -------------------------------------------------------------- discovery

    def _discover(self, force: bool = False) -> list:
        cache = self.state.get("models_cache") or {}
        cached_list = cache.get("list") if isinstance(cache.get("list"), list) else None
        fetched_at = cache.get("at")
        fresh = False
        if cached_list and fetched_at:
            try:
                age = (now_utc() - _dt.datetime.fromisoformat(fetched_at)).total_seconds()
                fresh = age < DISCOVERY_TTL_S
            except ValueError:
                fresh = False
        if cached_list and fresh and not force:
            return cached_list

        names: list = []
        try:
            for m in self._client(60).models.list():
                name = (getattr(m, "name", "") or "").split("/")[-1]
                if not name or is_excluded(name) or model_tier(name) is None:
                    continue
                actions = getattr(m, "supported_actions", None)
                if actions and "generateContent" not in actions:
                    continue
                names.append(name)
        except Exception as exc:
            self._log(f"⚠️ تعذر جلب قائمة موديلات Gemini ({short_error(exc, 100)}) نستخدم القائمة الاحتياطية")
        if not names:
            names = list(cached_list or SEED_MODELS)
        names = sorted(set(names), key=_model_sort_key)
        self.state["models_cache"] = {"at": now_utc().isoformat(timespec="seconds"), "list": names}
        self._save_state()
        return names

    # ------------------------------------------------------- buckets / health

    def _bucket(self, model: str) -> str:
        return self.state["aliases"].get(model, model)

    def _limits(self, bucket: str) -> dict:
        learned = self.state["limits"].get(bucket) or {}
        out = dict(DEFAULT_LIMITS)
        for key in ("rpm", "rpd", "tpm"):
            val = learned.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                out[key] = int(val)
        out["learned_rpd"] = isinstance(learned.get("rpd"), (int, float))
        return out

    def _used_today(self, bucket: str) -> int:
        return int(self.state["ledger"].get(pacific_date_str(), {}).get(bucket, 0))

    def _remaining_today(self, bucket: str) -> int:
        return max(0, self._limits(bucket)["rpd"] - self._used_today(bucket))

    def _park(self, model: str, until_utc: _dt.datetime, state: str, reason: str) -> None:
        self.state["health"][model] = {
            "state": state,
            "until": until_utc.astimezone(UTC).isoformat(timespec="seconds"),
            "reason": reason[:200],
        }

    def _parked(self, name: str) -> bool:
        entry = self.state["health"].get(name)
        if not entry:
            return False
        try:
            until = _dt.datetime.fromisoformat(entry.get("until", ""))
        except ValueError:
            return False
        if until.tzinfo is None:
            until = until.replace(tzinfo=UTC)
        if now_utc() >= until:
            self.state["health"].pop(name, None)
            return False
        return True

    def _is_available(self, model: str) -> bool:
        bucket = self._bucket(model)
        if self._parked(model) or self._parked(bucket):
            return False
        lim = self._limits(bucket)
        if lim["learned_rpd"] and self._used_today(bucket) >= lim["rpd"]:
            self._park(bucket, next_pacific_midnight_utc(), "day_quota", "ledger reached learned RPD")
            self._save_state()
            return False
        return True

    def _route_candidates(self, route: str) -> list:
        cfg = ROUTES.get(route)
        if cfg is None:
            raise GatewayError(f"❌ مسار غير معروف: {route}")
        models = self._discover()
        out: list = []
        seen_buckets: set = set()
        for tier in cfg["tiers"]:
            tier_models = sorted((m for m in models if model_tier(m) == tier), key=_model_sort_key)
            for m in tier_models:
                bucket = self._bucket(m)
                if bucket in seen_buckets:
                    continue
                if not self._is_available(m):
                    continue
                seen_buckets.add(bucket)
                out.append(m)
        return out

    # ---------------------------------------------------------------- ledger

    def _record_attempt(self, model: str, est_tokens: int, counted: bool) -> None:
        bucket = self._bucket(model)
        self._recent.setdefault(bucket, []).append((time.time(), int(est_tokens)))
        if counted:
            day = pacific_date_str()
            per_day = self.state["ledger"].setdefault(day, {})
            per_day[bucket] = int(per_day.get(bucket, 0)) + 1
            self._prune_ledger(self.state)
        self._save_state()

    def _pace(self, model: str, est_tokens: int) -> None:
        bucket = self._bucket(model)
        lim = self._limits(bucket)
        rpm = max(1, lim["rpm"])
        tpm = max(1000, lim["tpm"])
        now = time.time()
        window = [(t, tok) for (t, tok) in self._recent.get(bucket, []) if now - t < 60.0]
        self._recent[bucket] = window
        wait = 0.0
        if window:
            wait = max(wait, window[-1][0] + 60.0 / rpm + 0.5 - now)
        if len(window) >= rpm:
            wait = max(wait, window[-rpm][0] + 60.5 - now)
        if window and sum(tok for _, tok in window) + est_tokens > tpm * 0.9:
            wait = max(wait, window[0][0] + 60.5 - now)
        if wait > 0:
            if wait >= 3:
                self._log(f"   ⏳ تنظيم السرعة لـ {model}: انتظار {wait:.0f} ثانية")
            time.sleep(wait)

    # --------------------------------------------------------- token estimate

    @staticmethod
    def _estimate_tokens(contents: Sequence[Any], system_instruction: Optional[str]) -> int:
        total_chars = len(system_instruction or "")
        audio_tokens = 0.0
        for item in contents:
            if isinstance(item, str):
                total_chars += len(item)
            elif isinstance(item, dict) and isinstance(item.get("data"), (bytes, bytearray)):
                secs = item.get("seconds")
                if not isinstance(secs, (int, float)):
                    secs = len(item["data"]) * 8.0 / (ASSUMED_AUDIO_KBPS * 1000.0)
                audio_tokens += float(secs) * AUDIO_TOKENS_PER_SEC
            elif isinstance(item, (bytes, bytearray)):
                audio_tokens += len(item) * 8.0 / (ASSUMED_AUDIO_KBPS * 1000.0) * AUDIO_TOKENS_PER_SEC
            else:
                inline = getattr(item, "inline_data", None)
                data = getattr(inline, "data", None) if inline is not None else None
                if isinstance(data, (bytes, bytearray)):
                    audio_tokens += len(data) * 8.0 / (ASSUMED_AUDIO_KBPS * 1000.0) * AUDIO_TOKENS_PER_SEC
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    total_chars += len(text)
        return int(total_chars / 3.0 + audio_tokens) + 200

    @staticmethod
    def _build_contents(contents: Any) -> list:
        if not isinstance(contents, (list, tuple)):
            contents = [contents]
        parts = []
        for item in contents:
            if isinstance(item, str):
                if item:
                    parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("data"), (bytes, bytearray)):
                parts.append(audio_part(item["data"], item.get("mime_type", "audio/mp3")))
            elif isinstance(item, (bytes, bytearray)):
                parts.append(audio_part(bytes(item), "audio/mp3"))
            elif item is not None:
                parts.append(item)
        if not parts:
            raise GatewayError("❌ الطلب فاضي، ما فيه نص أو صوت نرسله")
        return parts

    # ---------------------------------------------------------------- thinking

    def _default_thinking_mode(self, model: str) -> str:
        version = model_version(model)
        if version is None or version >= 3.0:
            return "level"
        return "budget"

    def _thinking_mode(self, model: str) -> str:
        mode = self.state["thinking"].get(model)
        return mode if mode in THINKING_MODES else self._default_thinking_mode(model)

    def _switch_thinking_mode(self, model: str) -> str:
        current = self._thinking_mode(model)
        nxt = THINKING_MODES[min(len(THINKING_MODES) - 1, THINKING_MODES.index(current) + 1)]
        self.state["thinking"][model] = nxt
        self._save_state()
        return nxt

    def _thinking_config(self, model: str, level: Optional[str]):
        if not level or level == "none":
            return None
        level = level.lower()
        mode = self._thinking_mode(model)
        if mode == "level":
            enum_cls = getattr(types, "ThinkingLevel", None)
            value = getattr(enum_cls, level.upper(), None) if enum_cls is not None else None
            try:
                return types.ThinkingConfig(thinking_level=value if value is not None else level.upper())
            except Exception:
                self.state["thinking"][model] = "budget"
                mode = "budget"
        if mode == "budget":
            try:
                return types.ThinkingConfig(thinking_budget=THINKING_BUDGETS.get(level, 1024))
            except Exception:
                self.state["thinking"][model] = "none"
                return None
        return None

    def _schema_ok(self, model: str) -> bool:
        return self.state["schema"].get(model, True) is not False

    # -------------------------------------------------------------- the call

    def _call(
        self,
        model: str,
        parts: list,
        schema: Optional[type],
        json_mode: bool,
        thinking: Optional[str],
        timeout_s: float,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
    ):
        cfg: dict = {}
        if temperature is not None:
            cfg["temperature"] = float(temperature)
        if max_output_tokens:
            cfg["max_output_tokens"] = int(max_output_tokens)
        if system_instruction:
            cfg["system_instruction"] = system_instruction
        if json_mode:
            cfg["response_mime_type"] = "application/json"
        if schema is not None:
            cfg["response_schema"] = schema
        thinking_cfg = self._thinking_config(model, thinking)
        if thinking_cfg is not None:
            cfg["thinking_config"] = thinking_cfg
        config = types.GenerateContentConfig(**cfg)
        return self._client(timeout_s).models.generate_content(model=model, contents=parts, config=config)

    # ------------------------------------------------------ error classifier

    @staticmethod
    def _detail_items(obj: Any) -> list:
        found: list = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if "@type" in node:
                    found.append(node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(obj)
        return found

    @staticmethod
    def _parse_delay(value: Any) -> Optional[float]:
        if value is None:
            return None
        match = re.search(r"(\d+(?:\.\d+)?)\s*s", str(value))
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def _classify(self, exc: BaseException, schema_in_use: bool) -> _ErrInfo:
        name = exc.__class__.__name__
        text = str(exc)
        low = text.lower()

        is_api = genai_errors is not None and isinstance(exc, genai_errors.APIError)
        if not is_api:
            if "timeout" in name.lower() or "timed out" in low or isinstance(exc, TimeoutError):
                return _ErrInfo("transient", "انتهت مهلة الاتصال (timeout)", counted=True)
            transient_names = ("ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError",
                               "WriteError", "NetworkError", "TransportError", "ProtocolError")
            if any(t.lower() in name.lower() for t in transient_names) or "connection" in low:
                return _ErrInfo("transient", f"مشكلة اتصال ({name})")
            if schema_in_use and ("schema" in low or "pydantic" in low or "json" in low or isinstance(exc, (TypeError, ValueError))):
                return _ErrInfo("schema_invalid", short_error(exc))
            if "thinking" in low:
                return _ErrInfo("thinking_invalid", short_error(exc))
            return _ErrInfo("bad_request", short_error(exc))

        code = getattr(exc, "code", None)
        try:
            code = int(code) if code is not None else None
        except (TypeError, ValueError):
            code = None
        message = str(getattr(exc, "message", "") or "")
        status = str(getattr(exc, "status", "") or "")
        details = getattr(exc, "details", None)
        msg_low = (message + " " + status).lower()

        if code == 404:
            return _ErrInfo("not_found", message[:160] or "404 NOT_FOUND")
        if code == 401:
            return _ErrInfo("fatal", "مفتاح Gemini مرفوض (401). تأكد من GEMINI_API_KEY في Secrets")
        if code == 403:
            return _ErrInfo("forbidden", message[:160] or "403 PERMISSION_DENIED")
        if code == 429:
            info = _ErrInfo("transient", message[:160] or "429 RESOURCE_EXHAUSTED")
            delay = None
            for item in self._detail_items(details):
                typ = str(item.get("@type", ""))
                if typ.endswith("RetryInfo"):
                    delay = self._parse_delay(item.get("retryDelay"))
                if typ.endswith("QuotaFailure"):
                    for v in item.get("violations") or []:
                        qid = str(v.get("quotaId", ""))
                        qmetric = str(v.get("quotaMetric", ""))
                        dims = v.get("quotaDimensions") or {}
                        qval = v.get("quotaValue")
                        try:
                            qval = int(str(qval)) if qval is not None else None
                        except ValueError:
                            qval = None
                        metric = "tokens" if ("token" in qid.lower() or "token" in qmetric.lower()) else "requests"
                        if "PerDay" in qid:
                            info.kind, info.quota_scope = "per_day", "day"
                        elif "PerMinute" in qid and info.kind != "per_day":
                            info.kind, info.quota_scope = "per_minute", "minute"
                        else:
                            continue
                        info.quota_metric = metric
                        info.quota_value = qval
                        info.quota_model = dims.get("model") or info.quota_model
                        if info.kind == "per_day":
                            break
            if info.kind == "transient":
                if "per day" in msg_low or "perday" in msg_low:
                    info.kind, info.quota_scope, info.quota_metric = "per_day", "day", "requests"
                elif "per minute" in msg_low or "perminute" in msg_low:
                    info.kind, info.quota_scope, info.quota_metric = "per_minute", "minute", "requests"
            if delay is None:
                match = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
                if match:
                    delay = float(match.group(1))
            info.retry_delay = delay
            return info
        if code == 400:
            if "thinking" in msg_low:
                return _ErrInfo("thinking_invalid", message[:160])
            if schema_in_use and ("schema" in msg_low or "response_mime_type" in msg_low or "json" in msg_low):
                return _ErrInfo("schema_invalid", message[:160])
            return _ErrInfo("bad_request", message[:160] or "400 INVALID_ARGUMENT")
        if code in (408, 499, 500, 502, 503, 504) or (code is not None and code >= 500):
            return _ErrInfo("transient", f"{code} {status}".strip())
        return _ErrInfo("bad_request", f"{code} {status} {message[:120]}".strip())

    # ------------------------------------------------------- response helpers

    @staticmethod
    def _response_text(resp: Any) -> str:
        try:
            text = resp.text
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
        chunks = []
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for part in getattr(content, "parts", None) or []:
                if getattr(part, "thought", False):
                    continue
                txt = getattr(part, "text", None)
                if isinstance(txt, str):
                    chunks.append(txt)
        return "".join(chunks)

    @staticmethod
    def _usage(resp: Any) -> dict:
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return {}
        out = {}
        for key, attr in (("input", "prompt_token_count"), ("output", "candidates_token_count"),
                          ("thinking", "thoughts_token_count"), ("total", "total_token_count")):
            val = getattr(meta, attr, None)
            if isinstance(val, int):
                out[key] = val
        return out

    @staticmethod
    def _block_reason(resp: Any) -> str:
        feedback = getattr(resp, "prompt_feedback", None)
        reason = getattr(feedback, "block_reason", None) if feedback is not None else None
        if reason:
            return str(reason)
        for cand in getattr(resp, "candidates", None) or []:
            finish = str(getattr(cand, "finish_reason", "") or "")
            if any(k in finish.upper() for k in ("SAFETY", "BLOCK", "PROHIBITED", "RECITATION")):
                return finish
        return ""

    def _parse_response(
        self,
        resp: Any,
        schema: Optional[type],
        json_mode: bool,
        normalizer: Optional[Callable[[Any], Any]],
    ) -> tuple:
        """Return (data, raw_text, dropped_items). Raises ValueError on failure."""
        raw = self._response_text(resp)
        if schema is None and not json_mode:
            if not raw.strip():
                raise ValueError("empty text response")
            return raw.strip(), raw, 0
        if schema is not None:
            parsed = getattr(resp, "parsed", None)
            if isinstance(parsed, schema) and normalizer is None:
                return parsed, raw, 0
        obj = parse_json_loose(raw)
        if schema is not None:
            obj = wrap_to_schema(obj, schema)
        if normalizer is not None:
            obj = normalizer(obj)
        if schema is None:
            return obj, raw, 0
        try:
            instance, dropped = validate_lenient(schema, obj)
        except ValidationError as exc:
            raise ValueError(f"schema validation failed ({exc.error_count()} errors)") from exc
        return instance, raw, dropped

    # ----------------------------------------------------- per model attempt

    def _try_model(
        self,
        model: str,
        route: str,
        parts: list,
        schema: Optional[type],
        json_mode: bool,
        thinking: Optional[str],
        timeout_s: float,
        system_instruction: Optional[str],
        temperature: Optional[float],
        max_output_tokens: Optional[int],
        normalizer: Optional[Callable[[Any], Any]],
        est_tokens: int,
        label: str,
    ) -> _Outcome:
        attempts = 0
        transient_tries = 0
        minute_tries = 0
        thinking_switches = 0
        parse_tries = 0
        timeouts = 0
        last_raw = ""
        tag = f" [{label}]" if label else ""

        while True:
            if not self._is_available(model):
                return _Outcome("parked", attempts, message="parked", model=model)
            self._pace(model, est_tokens)
            use_schema = schema if (schema is not None and self._schema_ok(model)) else None
            mode_ar = {"level": "تفكير", "budget": "تفكير", "none": "بدون تفكير"}[self._thinking_mode(model)]
            level_ar = {"low": "منخفض", "medium": "متوسط", "high": "عميق"}.get((thinking or "").lower(), "")
            self._log(f"🧠{tag} {ROUTES[route]['name_ar']}: الاتصال بـ {model} ({mode_ar} {level_ar})".rstrip())
            attempts += 1
            t0 = time.time()
            try:
                resp = self._call(model, parts, use_schema, json_mode or schema is not None, thinking,
                                  timeout_s, system_instruction, temperature, max_output_tokens)
            except Exception as exc:  # noqa: BLE001
                err = self._classify(exc, schema_in_use=use_schema is not None)
                self._record_attempt(model, est_tokens, counted=err.counted)
                bucket = self._bucket(model)

                if err.kind == "fatal":
                    raise GatewayError(f"❌ {err.message}") from exc

                if err.kind == "not_found":
                    self._park(model, now_utc() + _dt.timedelta(days=NOT_FOUND_PARK_DAYS), "not_found", err.message)
                    self._save_state()
                    self._log(f"   🚫 {model} غير متاح لهذا المفتاح (404)، تم استبعاده {NOT_FOUND_PARK_DAYS} أيام")
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "forbidden":
                    self._park(model, now_utc() + _dt.timedelta(days=FORBIDDEN_PARK_DAYS), "forbidden", err.message)
                    self._save_state()
                    self._log(f"   🚫 {model} مرفوض لهذا المفتاح (403)، تم استبعاده يوم")
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "per_day":
                    target = err.quota_model or model
                    if target != model and target:
                        self.state["aliases"][model] = target
                        bucket = target
                    if err.quota_value is not None and err.quota_metric == "requests":
                        lim = self.state["limits"].setdefault(bucket, {})
                        lim["rpd"] = err.quota_value
                        lim["source"] = "429 quotaValue"
                        lim["updated"] = now_utc().isoformat(timespec="seconds")
                    if err.quota_value == 0:
                        until = now_utc() + _dt.timedelta(days=ZERO_QUOTA_PARK_DAYS)
                        self._park(bucket, until, "no_free_quota", err.message)
                        self._park(model, until, "no_free_quota", err.message)
                        self._log(f"   🔒 {model} ما له حصة مجانية (الحد 0)، تم استبعاده {ZERO_QUOTA_PARK_DAYS} أيام")
                    else:
                        reset_utc = next_pacific_midnight_utc()
                        self._park(bucket, reset_utc, "day_quota", err.message)
                        self._park(model, reset_utc, "day_quota", err.message)
                        limit_txt = f" (الحد {err.quota_value} طلب)" if err.quota_value else ""
                        self._log(f"   📉 خلصت حصة اليوم للموديل {model}{limit_txt}، يرجع {pacific_reset_riyadh()[1]}")
                    self._save_state()
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "per_minute":
                    if err.quota_value:
                        lim = self.state["limits"].setdefault(bucket, {})
                        lim["tpm" if err.quota_metric == "tokens" else "rpm"] = err.quota_value
                        lim["source"] = "429 quotaValue"
                        lim["updated"] = now_utc().isoformat(timespec="seconds")
                        self._save_state()
                    minute_tries += 1
                    if minute_tries > MAX_PER_MINUTE_RETRIES:
                        return _Outcome("transient", attempts, message=err.message, model=model)
                    rpm = max(1, self._limits(bucket)["rpm"])
                    wait = max(err.retry_delay or 0.0, 60.0 / rpm) + random.uniform(1.0, 4.0)
                    self._log(f"   ⏱️ تجاوزنا حد الدقيقة على {model}، ننتظر {wait:.0f} ثانية")
                    time.sleep(wait)
                    continue

                if err.kind == "thinking_invalid":
                    thinking_switches += 1
                    if thinking_switches > MAX_THINKING_SWITCHES:
                        return _Outcome("bad_request", attempts, message=err.message, model=model)
                    new_mode = self._switch_thinking_mode(model)
                    self._log(f"   🔧 {model} ما قبل إعداد التفكير، نجرب الوضع ({new_mode})")
                    continue

                if err.kind == "schema_invalid":
                    if use_schema is not None:
                        self.state["schema"][model] = False
                        self._save_state()
                        self._log(f"   🔧 {model} ما قبل الهيكل الصارم، نكمل بوضع JSON مع الإصلاح المحلي")
                        continue
                    return _Outcome("bad_request", attempts, message=err.message, model=model)

                if err.kind == "transient":
                    if "timeout" in err.message.lower():
                        timeouts += 1
                        self._log(f"   ⏱️ {model} تجاوز {timeout_s:.0f} ثانية")
                        if timeouts >= 2:
                            return _Outcome("transient", attempts, message=err.message, model=model)
                    if transient_tries >= len(TRANSIENT_BACKOFF_S):
                        return _Outcome("transient", attempts, message=err.message, model=model)
                    wait = TRANSIENT_BACKOFF_S[transient_tries] * (1.0 + random.uniform(0.0, 0.3))
                    transient_tries += 1
                    self._log(f"   ⏳ {model} مشغول ({err.message[:60]})، محاولة ثانية بعد {wait:.0f} ثانية")
                    time.sleep(wait)
                    continue

                self._log(f"   ⚠️ {model} رفض الطلب ({err.message[:120]})")
                return _Outcome("bad_request", attempts, message=err.message, model=model)

            # HTTP success: the request counts against quota.
            self._record_attempt(model, est_tokens, counted=True)
            elapsed = time.time() - t0
            blocked = self._block_reason(resp)
            try:
                data, raw, dropped = self._parse_response(resp, use_schema or schema, json_mode, normalizer)
            except ValueError as exc:
                last_raw = self._response_text(resp)
                parse_tries += 1
                reason = blocked or short_error(exc, 120)
                if parse_tries <= 1:
                    self._log(f"   🔁 رد {model} ما انقرأ ({reason})، نعيد المحاولة مرة وحدة")
                    continue
                self._log(f"   ⚠️ رد {model} ما انقرأ بعد محاولتين ({reason})")
                return _Outcome("parse_failed", attempts, message=reason, raw=last_raw, model=model)

            usage = self._usage(resp)
            in_tok = usage.get("input")
            tok_txt = f" | دخل {in_tok / 1000:.1f}k توكن" if isinstance(in_tok, int) else ""
            drop_txt = f" | تم تجاهل {dropped} عنصر غير صالح" if dropped else ""
            self._log(f"✅ استجاب {model} خلال {elapsed:.1f} ثانية{tok_txt}{drop_txt}")
            result = GatewayResult(
                data=data,
                raw_text=raw,
                model=model,
                route=route,
                attempts=attempts,
                elapsed_s=round(elapsed, 2),
                usage=usage,
                schema_used=use_schema is not None,
                dropped_items=dropped,
            )
            return _Outcome("ok", attempts, result=result, model=model)

    # ---------------------------------------------------------------- public

    def generate(
        self,
        route: str,
        contents: Any,
        schema: Optional[type] = None,
        thinking: Optional[str] = None,
        timeout_s: Optional[float] = None,
        system_instruction: Optional[str] = None,
        temperature: Optional[float] = None,
        normalizer: Optional[Callable[[Any], Any]] = None,
        json_mode: Optional[bool] = None,
        est_tokens: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        label: str = "",
    ) -> GatewayResult:
        """Send one request through the route's model chain.

        contents: str, or a list of str / {"data": bytes, "mime_type": "audio/mp3", "seconds": float}
                  / types.Part. Audio dicts are turned into inline parts.
        schema:   Pydantic BaseModel class for structured output (data is an instance).
                  Without schema, json_mode=True returns a parsed object, else plain text.
        normalizer: optional callable applied to the parsed object before validation.

        Raises QuotaPause, ParseFailure or GatewayError."""
        if route not in ROUTES:
            raise GatewayError(f"❌ مسار غير معروف: {route}")
        if schema is not None and not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise GatewayError("❌ schema لازم يكون كلاس Pydantic")
        cfg = ROUTES[route]
        thinking = thinking or cfg["thinking"]
        timeout_s = float(timeout_s or cfg["timeout_s"])
        json_flag = bool(json_mode) if json_mode is not None else schema is not None
        raw_items = contents if isinstance(contents, (list, tuple)) else [contents]
        parts = self._build_contents(raw_items)
        est = int(est_tokens) if est_tokens else self._estimate_tokens(raw_items, system_instruction)

        busy_deadline = time.time() + self.max_busy_wait_s
        total_attempts = 0
        last_msg = ""
        round_no = 0

        while True:
            round_no += 1
            candidates = self._route_candidates(route)
            if not candidates:
                raise self._quota_pause(route)

            any_transient = False
            any_bad = False
            for model in candidates:
                outcome = self._try_model(
                    model, route, parts, schema, json_flag, thinking, timeout_s, system_instruction,
                    temperature, max_output_tokens, normalizer, est, label,
                )
                total_attempts += outcome.attempts
                if outcome.kind == "ok" and outcome.result is not None:
                    outcome.result.attempts = total_attempts
                    return outcome.result
                if outcome.kind == "parse_failed":
                    raise ParseFailure(route, outcome.model, outcome.raw, outcome.message)
                if outcome.kind == "transient":
                    any_transient = True
                elif outcome.kind == "bad_request":
                    any_bad = True
                if outcome.message:
                    last_msg = f"{outcome.model}: {outcome.message}"
                if outcome.kind == "transient":
                    self._log("   ↪️ ننتقل للموديل التالي")

            if not self._route_candidates(route):
                raise self._quota_pause(route)

            if any_transient:
                remaining = busy_deadline - time.time()
                if remaining > 5:
                    wait = min(60.0, remaining)
                    self._log(f"🌩️ كل موديلات {cfg['name_ar']} مشغولة حالياً، ننتظر {wait:.0f} ثانية ونعيد (جولة {round_no + 1})")
                    time.sleep(wait)
                    continue
                raise QuotaPause(
                    route, "busy", None,
                    f"⏸️ سيرفرات Gemini مشغولة جداً حالياً لموديلات {cfg['name_ar']}. "
                    f"التقدم محفوظ، شغّل الخلية مرة ثانية بعد شوي وبيكمل من نفس النقطة.",
                )

            if any_bad:
                raise GatewayError(f"❌ رفضت كل موديلات {cfg['name_ar']} الطلب: {last_msg}")
            raise GatewayError(f"❌ تعذر إكمال طلب {cfg['name_ar']}: {last_msg or 'سبب غير معروف'}")

    def _quota_pause(self, route: str) -> QuotaPause:
        reset_riyadh, reset_text = pacific_reset_riyadh()
        name_ar = ROUTES[route]["name_ar"]
        msg = (
            f"⏸️ خلصت حصة اليوم لموديلات {name_ar}. التقدم محفوظ في Drive، "
            f"شغّل الخلية مرة ثانية {reset_text} وبيكمل من نفس النقطة."
        )
        return QuotaPause(route, "quota", reset_riyadh, msg)

    # ------------------------------------------------------------ budgeting

    def affordable(self, route: str) -> int:
        """Requests the route can still make today across its healthy models
        (estimate when limits were not learned yet)."""
        total = 0
        seen: set = set()
        for model in self._route_candidates(route):
            bucket = self._bucket(model)
            if bucket in seen:
                continue
            seen.add(bucket)
            total += self._remaining_today(bucket)
        return total

    def plan(self, needs: dict) -> Budget:
        """Allocate planned requests per route (in the given order) against the shared
        remaining daily budget of each model bucket."""
        remaining: dict = {}
        routes_out: dict = {}
        estimated = False
        for route, needed in needs.items():
            needed = max(0, int(needed))
            got = 0
            used = []
            for model in self._route_candidates(route):
                bucket = self._bucket(model)
                if bucket not in remaining:
                    remaining[bucket] = self._remaining_today(bucket)
                    if not self._limits(bucket)["learned_rpd"]:
                        estimated = True
                take = min(needed - got, remaining[bucket])
                if take > 0:
                    remaining[bucket] -= take
                    got += take
                    used.append({"model": model, "requests": take, "left_after": remaining[bucket]})
                if got >= needed:
                    break
            routes_out[route] = {"needed": needed, "available": got, "models": used}
        ok = all(r["available"] >= r["needed"] for r in routes_out.values())

        lines = ["📊 خطة طلبات Gemini لهذا التشغيل:"]
        for route, info in routes_out.items():
            name_ar = ROUTES.get(route, {}).get("name_ar", route)
            models_txt = "، ".join(f"{u['model']} ×{u['requests']} (يتبقى {u['left_after']})" for u in info["models"]) or "ما فيه موديل متاح"
            mark = "✅" if info["available"] >= info["needed"] else "⚠️"
            lines.append(f"   {mark} {name_ar}: {info['needed']} طلب ← {models_txt}")
        if estimated:
            lines.append("   ℹ️ بعض الحدود تقديرية (20 طلب يومياً) لين نتعلم الحد الحقيقي من رد Gemini")
        if not ok:
            lines.append(f"   ⏸️ الحصة ما تكفي كاملة اليوم، الحصة ترجع {pacific_reset_riyadh()[1]}")
        return Budget(ok=ok, routes=routes_out, estimated=estimated, text_ar="\n".join(lines))

    def usage_report_ar(self) -> str:
        """Short Arabic report for the Studio: requests used today per model and parked models."""
        day = pacific_date_str()
        used = self.state["ledger"].get(day, {})
        lines = [f"📊 استهلاك Gemini اليوم (يوم الحصة {day}):"]
        if not used:
            lines.append("   ما تم استخدام أي طلب اليوم")
        for bucket, count in sorted(used.items(), key=lambda kv: -kv[1]):
            lim = self._limits(bucket)
            src = "" if lim["learned_rpd"] else " تقديري"
            lines.append(f"   • {bucket}: {count} من {lim['rpd']}{src}")
        parked = []
        for model, entry in list(self.state["health"].items()):
            if self._parked(model):
                try:
                    until = _dt.datetime.fromisoformat(entry["until"])
                    left = format_duration_ar((until - now_utc()).total_seconds())
                except (KeyError, ValueError):
                    left = "?"
                reason_ar = {
                    "not_found": "غير متاح",
                    "day_quota": "خلصت حصته",
                    "no_free_quota": "بدون حصة مجانية",
                    "forbidden": "مرفوض",
                }.get(entry.get("state"), entry.get("state", ""))
                parked.append(f"   • {model}: {reason_ar} (يرجع بعد {left})")
        if parked:
            lines.append("🚫 موديلات مستبعدة مؤقتاً:")
            lines.extend(parked)
        lines.append(f"🔄 الحصة اليومية ترجع {pacific_reset_riyadh()[1]}")
        return "\n".join(lines)

    def reset_model_health(self) -> None:
        """Clear parked models (for example after linking billing)."""
        self.state["health"] = {}
        self._save_state()


# === END OF gemini_gateway.py ===
