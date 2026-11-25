"""
gemini_gateway.py
Quota aware, overload aware gateway to the Gemini API (google-genai SDK) for the
TwitchClipper pipeline. Version 3.

What it does
  * Discovers usable models and ranks them per route. Tier order is fixed per route
    (scout: Flash then Lite; critic, golden, distill, rules: Pro then Flash), and inside a
    tier the live health ranking decides: the route's last winner first (sticky), then
    success rate and EWMA latency, version number only as a tie break.
  * Circuit breaker per model (persisted with expiry):
      strike   500, 502, 503, 504, UNAVAILABLE, overloaded, bare 429, timeout, network error
      on a strike no backoff on that model: go straight to the next candidate; one quick
               retry after 3 s only when it is the last healthy candidate
      open     2 strikes within 10 minutes, or a single 504 or timeout: cooldown 12 minutes
      probe    after the cooldown one call is allowed; success closes the breaker, failure
               reopens it with double the cooldown, capped at 30 minutes
      all open wait for the earliest probe if it fits the busy budget, else QuotaPause("busy")
  * Daily quota handling kept from v2 (404 parks 7 days, 429 PerDay parks until Pacific
    midnight, quotaValue 0 parks 7 days, 429 PerMinute waits), with limits seeded per tier
    and overwritten by real 429 quotaValue numbers.
  * Media content items: video_item / audio_item / image_item dicts (v2 audio dicts and raw
    types.Part objects still work). Video gets VideoMetadata(fps) and the route's media
    resolution; a model that rejects either is remembered (capability cache) and called
    without it.
  * Token estimator per modality (Gemini 3 rates: 70 tokens per video frame at low or medium,
    280 at high, 25 per second of audio, images 280 at low) with EWMA self calibration from
    usage_metadata.prompt_tokens_details.
  * Payload guard: inline bytes are counted with base64 overhead; above 18 MB the largest
    media parts go through the File API (upload, wait for ACTIVE, use, delete).
  * Adaptive timeout: 45 s plus 2 s per 1k input tokens, never below the route floor,
    capped at 180 s.
  * Structured output with Pydantic plus the v2 local repair chain. A response that cannot
    be parsed moves to the next healthy model once before ParseFailure is raised.
  * State saves are batched (at most every 5 s, plus flush()) and merged with the file on
    disk, so Stage 3 and the Studio never overwrite each other's ledger.

Public contract
  GeminiGateway(api_key, state_path, max_busy_wait_s=480.0, verbose=True)
    .generate(route, contents, schema=None, thinking=None, timeout_s=None, ...) -> GatewayResult
    .estimate(contents, system_instruction=None) -> TokenEstimate
    .plan({"scout": 6, "critic": 9}) -> Budget
    .affordable(route) -> int
    .usage_report_ar() / .health_report_ar() -> str
    .reset_model_health() / .flush()
  video_item / audio_item / image_item / audio_part / parse_json_loose
  selftest() -> dict   (offline, fake client, no network)
  Exceptions: QuotaPause, ParseFailure, GatewayError
"""

from __future__ import annotations

import ast
import atexit
import copy
import datetime as _dt
import json
import math
import mimetypes
import os
import random
import re
import tempfile
import threading
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
    clamp,
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

EXCLUDED_MODEL_KEYWORDS = (
    "tts", "image", "audio", "embedding", "live", "veo", "imagen", "lyria", "robotics",
    "computer-use", "banana", "aqa", "deep-research", "omni",
)

SEED_MODELS = (
    "gemini-flash-latest",
    "gemini-3-flash-preview",
    "gemini-flash-lite-latest",
    "gemini-pro-latest",
)

ROUTES = {
    "scout": {"tiers": ("flash", "lite"), "thinking": "low", "min_timeout_s": 60,
              "media": None, "name_ar": "الكشاف"},
    "critic": {"tiers": ("pro", "flash"), "thinking": "high", "min_timeout_s": 90,
               "media": "low", "name_ar": "الناقد"},
    "golden": {"tiers": ("pro", "flash"), "thinking": "high", "min_timeout_s": 90,
               "media": "low", "name_ar": "محلل اللقطات الذهبية"},
    "distill": {"tiers": ("pro", "flash"), "thinking": "high", "min_timeout_s": 90,
                "media": None, "name_ar": "مستخلص الحمض الذهبي"},
    "rules": {"tiers": ("pro", "flash"), "thinking": "high", "min_timeout_s": 90,
              "media": None, "name_ar": "مدرب القواعد"},
    "general": {"tiers": ("flash", "lite", "pro"), "thinking": "low", "min_timeout_s": 45,
                "media": None, "name_ar": "المساعد"},
}

BREAKER = {
    "strike_window_s": 600,
    "strikes_to_open": 2,
    "cooldown_s": 720,
    "max_cooldown_s": 1800,
    "quick_retry_s": 3,
    "probe_grace_s": 1800,
    "round_pause_s": 5,
}

SEED_LIMITS = {
    "flash": {"rpm": 5, "rpd": 20, "tpm": 250_000},
    "lite": {"rpm": 10, "rpd": 500, "tpm": 250_000},
    "pro": {"rpm": 5, "rpd": 20, "tpm": 250_000},
}

STATE_VERSION = 2
STATE_SECTIONS = ("ledger", "limits", "health", "thinking", "schema", "aliases", "models_cache",
                  "breaker", "perf", "sticky", "calib", "caps")

MAX_PER_MINUTE_RETRIES = 2
MAX_THINKING_SWITCHES = 2
NOT_FOUND_PARK_DAYS = 7
ZERO_QUOTA_PARK_DAYS = 7
FORBIDDEN_PARK_DAYS = 1
DISCOVERY_TTL_S = 12 * 3600
LEDGER_KEEP_DAYS = 7
DEFAULT_MAX_BUSY_WAIT_S = 480.0
SAVE_INTERVAL_S = 5.0

THINKING_BUDGETS = {"low": 1024, "medium": 4096, "high": 8192}
THINKING_MODES = ("level", "budget", "none")

# Token rates (Gemini 3 family)
VIDEO_FRAME_TOKENS = {"low": 70, "medium": 70, "high": 280, None: 70}
IMAGE_TOKENS = {"low": 280, "medium": 560, "high": 1120, None: 1120}
AUDIO_TOKENS_PER_SEC = 25.0
TEXT_CHARS_PER_TOKEN = 3.0
REQUEST_OVERHEAD_TOKENS = 200
ASSUMED_AUDIO_KBPS = 32.0
ASSUMED_VIDEO_KBPS = 300.0
MODALITIES = ("text", "audio", "video", "image")
CALIB_ALPHA = 0.3
CALIB_RANGE = (0.25, 4.0)

INLINE_LIMIT_BYTES = 18 * 1024 * 1024
UPLOAD_ACTIVE_TIMEOUT_S = 180.0

TIMEOUT_BASE_S = 45.0
TIMEOUT_PER_1K_S = 2.0
TIMEOUT_CAP_S = 180.0


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
    local repair and one fallback model. The caller should save the work as pending."""

    def __init__(self, route: str, model: str, raw_text: str, detail: str):
        super().__init__(f"{route}/{model}: {detail}")
        self.route = route
        self.model = model
        self.raw_text = raw_text
        self.detail = detail


@dataclass
class TokenEstimate:
    text: int
    audio: int
    video: int
    image: int
    total: int
    payload_bytes: int


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
    usage_by_modality: dict = field(default_factory=dict)


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
    hard: bool = False
    retry_delay: Optional[float] = None
    quota_scope: Optional[str] = None
    quota_metric: Optional[str] = None
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
    """lite, pro or flash. A Gemini model with no tier word in its name (for example a
    Gemini 4 codename) joins the flash tier; 404 or zero quota parks it automatically."""
    low = (name or "").lower()
    if "gemini" not in low:
        return None
    if "lite" in low:
        return "lite"
    if "pro" in low:
        return "pro"
    return "flash"


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


# ============================================================================
# Content items
# ============================================================================

def video_item(data: bytes, seconds: float, fps: float = 2.0, mime_type: str = "video/mp4") -> dict:
    """Video with its own audio track for multimodal prompts (critic, golden)."""
    return {"kind": "video", "data": bytes(data), "mime_type": mime_type,
            "seconds": float(seconds) if seconds is not None else None,
            "fps": float(fps) if fps else None}


def audio_item(data: bytes, seconds: float, mime_type: str = "audio/mp3") -> dict:
    return {"kind": "audio", "data": bytes(data), "mime_type": mime_type,
            "seconds": float(seconds) if seconds is not None else None, "fps": None}


def image_item(data: bytes, mime_type: str = "image/jpeg") -> dict:
    return {"kind": "image", "data": bytes(data), "mime_type": mime_type, "seconds": None, "fps": None}


def audio_part(data: bytes, mime_type: str = "audio/mp3"):
    """Inline audio part (kept from v2)."""
    return types.Part.from_bytes(data=bytes(data), mime_type=mime_type)


def _kind_from_mime(mime: str) -> str:
    low = str(mime or "").lower()
    if low.startswith("video/"):
        return "video"
    if low.startswith("image/"):
        return "image"
    return "audio"


def _normalize_items(contents: Any) -> list:
    """Turn the caller's contents into a list of normalized items:
    {"kind": "text", "text"} | {"kind": "audio"|"video"|"image", "data", "mime_type", "seconds", "fps"}
    | {"kind": "part", "part"}."""
    if not isinstance(contents, (list, tuple)):
        contents = [contents]
    items = []
    for item in contents:
        if item is None:
            continue
        if isinstance(item, str):
            if item:
                items.append({"kind": "text", "text": item})
        elif isinstance(item, dict) and isinstance(item.get("data"), (bytes, bytearray)):
            mime = str(item.get("mime_type") or "audio/mp3")
            kind = str(item.get("kind") or _kind_from_mime(mime)).lower()
            if kind not in ("audio", "video", "image"):
                kind = _kind_from_mime(mime)
            secs = item.get("seconds")
            fps = item.get("fps")
            items.append({
                "kind": kind, "data": bytes(item["data"]), "mime_type": mime,
                "seconds": float(secs) if isinstance(secs, (int, float)) and secs > 0 else None,
                "fps": float(fps) if isinstance(fps, (int, float)) and fps > 0 else None,
            })
        elif isinstance(item, (bytes, bytearray)):
            items.append({"kind": "audio", "data": bytes(item), "mime_type": "audio/mp3", "seconds": None, "fps": None})
        else:
            items.append({"kind": "part", "part": item})
    if not items:
        raise GatewayError("❌ الطلب فاضي، ما فيه نص أو وسائط نرسلها")
    return items


# ============================================================================
# JSON repair chain (kept from v2)
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
# Schema helpers (Pydantic v2, kept from v2)
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
# Small helpers
# ============================================================================

def _modality_name(value: Any) -> str:
    text = str(getattr(value, "name", value) or "").lower()
    return text.split(".")[-1]


def _iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), UTC).isoformat(timespec="seconds")


def _ts(value: Any) -> Optional[float]:
    if not value:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        moment = _dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.timestamp()


def _api_fields(exc: BaseException) -> tuple:
    """(is_api, code, message, status, details) for SDK API errors and duck typed fakes."""
    code = getattr(exc, "code", None)
    is_api = genai_errors is not None and isinstance(exc, genai_errors.APIError)
    try:
        code = int(code) if code is not None and not isinstance(code, bool) else None
    except (TypeError, ValueError):
        code = None
    if not is_api and code is not None and hasattr(exc, "status"):
        is_api = True
    message = str(getattr(exc, "message", "") or "")
    status = str(getattr(exc, "status", "") or "")
    details = getattr(exc, "details", None)
    return is_api, code, message, status, details


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
        self.inline_limit_bytes = INLINE_LIMIT_BYTES
        self._clients: dict = {}
        self._client_factory: Optional[Callable[[int], Any]] = None
        self._now: Callable[[], float] = time.time
        self._sleep: Callable[[float], None] = time.sleep
        self._recent: dict = {}
        self._lock = threading.RLock()
        self._touched: set = set()
        self._ledger_delta: dict = {}
        self._dirty = False
        self._last_save = 0.0
        self.state = self._normalize_state(load_json_safe(self.state_path, default={}, expected_types=(dict,), quiet=True))
        try:
            atexit.register(self.flush)
        except Exception:
            pass

    # ------------------------------------------------------------------ state

    @staticmethod
    def _normalize_state(raw: Any) -> dict:
        state = dict(raw) if isinstance(raw, dict) else {}
        for key in STATE_SECTIONS:
            if not isinstance(state.get(key), dict):
                state[key] = {}
        state["version"] = STATE_VERSION
        keep = sorted(state["ledger"].keys())[-LEDGER_KEEP_DAYS:]
        state["ledger"] = {k: state["ledger"][k] for k in keep if isinstance(state["ledger"][k], dict)}
        for mod in MODALITIES:
            val = state["calib"].get(mod)
            if not isinstance(val, (int, float)) or not (CALIB_RANGE[0] <= val <= CALIB_RANGE[1]):
                state["calib"][mod] = 1.0
        return state

    def _touch(self, section: str, key: str) -> None:
        with self._lock:
            self._touched.add((section, str(key)))
            self._dirty = True

    def _save_state(self) -> bool:
        """Merge my changes into the file on disk, then write it atomically."""
        with self._lock:
            disk = self._normalize_state(load_json_safe(self.state_path, default={}, expected_types=(dict,), quiet=True))
            for section, key in self._touched:
                mine = self.state.get(section, {})
                if key in mine:
                    disk[section][key] = copy.deepcopy(mine[key])
                else:
                    disk[section].pop(key, None)
            for day, buckets in self._ledger_delta.items():
                per_day = disk["ledger"].setdefault(day, {})
                for bucket, count in buckets.items():
                    per_day[bucket] = int(per_day.get(bucket, 0) or 0) + int(count)
            disk = self._normalize_state(disk)
            try:
                atomic_write_json(self.state_path, disk, backup=True)
            except Exception as exc:
                log(f"⚠️ تعذر حفظ حالة Gemini في Drive: {short_error(exc)}")
                return False
            self.state = disk
            self._touched.clear()
            self._ledger_delta.clear()
            self._dirty = False
            self._last_save = self._now()
            return True

    def _maybe_save(self, force: bool = False) -> None:
        with self._lock:
            if not self._dirty:
                return
            if force or self._now() - self._last_save >= SAVE_INTERVAL_S:
                self._save_state()

    def flush(self) -> None:
        """Write pending state now (called at the end of a stage and on exit)."""
        try:
            self._maybe_save(force=True)
        except Exception:
            pass

    # ---------------------------------------------------------------- logging

    def _log(self, msg: str) -> None:
        if self.verbose:
            log(msg)

    # ---------------------------------------------------------------- clients

    def _client(self, timeout_s: float):
        key = int(max(10.0, float(timeout_s)) * 1000)
        if self._client_factory is not None:
            return self._client_factory(key)
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
        fetched_at = _ts(cache.get("at"))
        fresh = bool(cached_list) and fetched_at is not None and (now_utc().timestamp() - fetched_at) < DISCOVERY_TTL_S
        if fresh and not force:
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
        self.state["models_cache"]["list"] = names
        self.state["models_cache"]["at"] = now_utc().isoformat(timespec="seconds")
        self._touch("models_cache", "list")
        self._touch("models_cache", "at")
        self._maybe_save()
        return names

    # ------------------------------------------------------- buckets / limits

    def _bucket(self, model: str) -> str:
        return self.state["aliases"].get(model, model)

    def _limits(self, bucket: str) -> dict:
        tier = model_tier(bucket) or "flash"
        out = dict(SEED_LIMITS.get(tier, SEED_LIMITS["flash"]))
        learned = self.state["limits"].get(bucket) or {}
        for key in ("rpm", "rpd", "tpm"):
            val = learned.get(key)
            if isinstance(val, (int, float)) and val >= 0:
                out[key] = int(val)
        out["learned_rpd"] = isinstance(learned.get("rpd"), (int, float))
        return out

    def _set_limit(self, bucket: str, key: str, value: int) -> None:
        lim = self.state["limits"].setdefault(bucket, {})
        lim[key] = int(value)
        lim["source"] = "429 quotaValue"
        lim["updated"] = now_utc().isoformat(timespec="seconds")
        self._touch("limits", bucket)

    def _used_today(self, bucket: str) -> int:
        return int(self.state["ledger"].get(pacific_date_str(), {}).get(bucket, 0) or 0)

    def _remaining_today(self, bucket: str) -> int:
        return max(0, self._limits(bucket)["rpd"] - self._used_today(bucket))

    # ------------------------------------------------------------- parking

    def _park(self, model: str, until_utc: _dt.datetime, state: str, reason: str) -> None:
        self.state["health"][model] = {
            "state": state,
            "until": until_utc.astimezone(UTC).isoformat(timespec="seconds"),
            "reason": str(reason)[:200],
        }
        self._touch("health", model)

    def _parked(self, name: str) -> bool:
        entry = self.state["health"].get(name)
        if not entry:
            return False
        until = _ts(entry.get("until"))
        if until is None:
            return False
        if now_utc().timestamp() >= until:
            self.state["health"].pop(name, None)
            self._touch("health", name)
            return False
        return True

    def _is_available(self, model: str) -> bool:
        """Not parked for daily quota, 404, 403 or zero quota."""
        bucket = self._bucket(model)
        if self._parked(model) or self._parked(bucket):
            return False
        lim = self._limits(bucket)
        if lim["learned_rpd"] and self._used_today(bucket) >= lim["rpd"]:
            self._park(bucket, next_pacific_midnight_utc(), "day_quota", "ledger reached learned RPD")
            return False
        return True

    # ------------------------------------------------------------- breaker

    def _breaker_open_until(self, model: str) -> Optional[float]:
        entry = self.state["breaker"].get(model)
        if not entry:
            return None
        return _ts(entry.get("open_until"))

    def _breaker_blocked(self, model: str) -> bool:
        until = self._breaker_open_until(model)
        return until is not None and self._now() < until

    def _strike(self, model: str, message: str, hard: bool) -> bool:
        """Record an overload strike. Returns True when the breaker is (now) open."""
        now = self._now()
        entry = dict(self.state["breaker"].get(model) or {})
        opened_before = bool(entry.get("opened"))
        until = _ts(entry.get("open_until"))
        cooldown = int(entry.get("cooldown_s") or BREAKER["cooldown_s"])
        if until is not None and now < until:
            entry["last_error"] = str(message)[:160]
            self.state["breaker"][model] = entry
            self._touch("breaker", model)
            return True
        probe = opened_before and until is not None and until <= now < until + BREAKER["probe_grace_s"]
        if opened_before and not probe and (until is None or now >= until + BREAKER["probe_grace_s"]):
            entry = {}
            opened_before = False
            cooldown = BREAKER["cooldown_s"]
        strikes = [float(s) for s in entry.get("strikes") or [] if now - float(s) < BREAKER["strike_window_s"]]
        strikes.append(now)
        open_now = False
        if probe:
            cooldown = min(BREAKER["max_cooldown_s"], cooldown * 2)
            open_now = True
        elif hard or len(strikes) >= BREAKER["strikes_to_open"]:
            cooldown = BREAKER["cooldown_s"]
            open_now = True
        entry["last_error"] = str(message)[:160]
        entry["cooldown_s"] = int(cooldown)
        if open_now:
            entry["open_until"] = _iso(now + cooldown)
            entry["opened"] = True
            entry["strikes"] = []
        else:
            entry["strikes"] = strikes
            entry.setdefault("opened", False)
            entry.setdefault("open_until", "")
        self.state["breaker"][model] = entry
        self._touch("breaker", model)
        perf = dict(self.state["perf"].get(model) or {})
        perf["fail"] = int(perf.get("fail", 0) or 0) + 1
        self.state["perf"][model] = perf
        self._touch("perf", model)
        for route, sticky_model in list(self.state["sticky"].items()):
            if sticky_model == model:
                self.state["sticky"].pop(route, None)
                self._touch("sticky", route)
        if open_now:
            mins = max(1, int(round(cooldown / 60.0)))
            why = "ما رد بالوقت" if hard else "مشغول"
            self._log(f"   🔌 قاطع الدائرة: {model} {why} ({str(message)[:40]})، أوقفناه {mins} دقيقة ونكمل بموديل ثاني")
        return open_now

    def _health_success(self, model: str, latency_s: float) -> None:
        if model in self.state["breaker"]:
            self.state["breaker"].pop(model, None)
            self._touch("breaker", model)
        perf = dict(self.state["perf"].get(model) or {})
        perf["ok"] = int(perf.get("ok", 0) or 0) + 1
        old = perf.get("ewma_latency_s")
        perf["ewma_latency_s"] = round(float(latency_s) if not isinstance(old, (int, float))
                                       else 0.7 * float(old) + 0.3 * float(latency_s), 2)
        perf["last_ok"] = _iso(self._now())
        self.state["perf"][model] = perf
        self._touch("perf", model)

    def _set_sticky(self, route: str, model: str) -> None:
        if self.state["sticky"].get(route) != model:
            self.state["sticky"][route] = model
            self._touch("sticky", route)

    def _rank_key(self, route: str, model: str) -> tuple:
        perf = self.state["perf"].get(model) or {}
        ok = int(perf.get("ok", 0) or 0)
        fail = int(perf.get("fail", 0) or 0)
        rate = (ok + 1.0) / (ok + fail + 2.0)
        lat = perf.get("ewma_latency_s")
        lat = float(lat) if isinstance(lat, (int, float)) else 30.0
        entry = self.state["breaker"].get(model) or {}
        recent = 1 if entry.get("strikes") else 0
        sticky = 0 if self.state["sticky"].get(route) == model else 1
        return (sticky, recent, -round(rate, 1), int(lat // 10), _model_sort_key(model))

    # ---------------------------------------------------------- candidates

    def _route_models(self, route: str) -> list:
        """Available models of a route (not parked), tier order, ranked, one per quota bucket.
        Breaker state is not filtered here."""
        cfg = ROUTES.get(route)
        if cfg is None:
            raise GatewayError(f"❌ مسار غير معروف: {route}")
        models = self._discover()
        out: list = []
        seen_buckets: set = set()
        for tier in cfg["tiers"]:
            tier_models = [m for m in models if model_tier(m) == tier and self._is_available(m)]
            tier_models.sort(key=lambda m: self._rank_key(route, m))
            for m in tier_models:
                bucket = self._bucket(m)
                if bucket in seen_buckets:
                    continue
                seen_buckets.add(bucket)
                out.append(m)
        return out

    def _route_candidates(self, route: str) -> list:
        return [m for m in self._route_models(route) if not self._breaker_blocked(m)]

    def _blocked_candidates(self, route: str) -> list:
        return [m for m in self._route_models(route) if self._breaker_blocked(m)]

    # ---------------------------------------------------------------- ledger

    def _record_attempt(self, model: str, est_tokens: int, counted: bool) -> None:
        bucket = self._bucket(model)
        self._recent.setdefault(bucket, []).append((self._now(), int(est_tokens)))
        if counted:
            day = pacific_date_str()
            per_day = self.state["ledger"].setdefault(day, {})
            per_day[bucket] = int(per_day.get(bucket, 0) or 0) + 1
            delta = self._ledger_delta.setdefault(day, {})
            delta[bucket] = int(delta.get(bucket, 0)) + 1
            self._dirty = True
        self._maybe_save()

    def _pace(self, model: str, est_tokens: int) -> None:
        bucket = self._bucket(model)
        lim = self._limits(bucket)
        rpm = max(1, lim["rpm"])
        tpm = max(1000, lim["tpm"])
        now = self._now()
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
            self._sleep(wait)

    # --------------------------------------------------------- token estimate

    def _estimate_raw(self, items: list, system_instruction: Optional[str], media_res: Optional[str]) -> tuple:
        """(raw tokens per modality before calibration, inline payload bytes)."""
        raw = {m: 0.0 for m in MODALITIES}
        chars = len(system_instruction or "")
        payload = len((system_instruction or "").encode("utf-8"))
        frame_tok = VIDEO_FRAME_TOKENS.get(media_res, 70)
        image_tok = IMAGE_TOKENS.get(media_res, 1120)
        for it in items:
            kind = it["kind"]
            if kind == "text":
                chars += len(it["text"])
                payload += len(it["text"].encode("utf-8"))
            elif kind == "audio":
                secs = it.get("seconds") or len(it["data"]) * 8.0 / (ASSUMED_AUDIO_KBPS * 1000.0)
                raw["audio"] += secs * AUDIO_TOKENS_PER_SEC
                payload += int(len(it["data"]) * 4 / 3)
            elif kind == "video":
                secs = it.get("seconds") or len(it["data"]) * 8.0 / (ASSUMED_VIDEO_KBPS * 1000.0)
                fps = it.get("fps") or 1.0
                raw["video"] += math.ceil(secs * fps) * frame_tok
                raw["audio"] += secs * AUDIO_TOKENS_PER_SEC
                payload += int(len(it["data"]) * 4 / 3)
            elif kind == "image":
                raw["image"] += image_tok
                payload += int(len(it["data"]) * 4 / 3)
            else:
                part = it["part"]
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    chars += len(text)
                    payload += len(text.encode("utf-8"))
                inline = getattr(part, "inline_data", None)
                data = getattr(inline, "data", None) if inline is not None else None
                if isinstance(data, (bytes, bytearray)):
                    pkind = _kind_from_mime(getattr(inline, "mime_type", "") or "")
                    payload += int(len(data) * 4 / 3)
                    if pkind == "image":
                        raw["image"] += image_tok
                    elif pkind == "video":
                        secs = len(data) * 8.0 / (ASSUMED_VIDEO_KBPS * 1000.0)
                        raw["video"] += math.ceil(secs) * frame_tok
                        raw["audio"] += secs * AUDIO_TOKENS_PER_SEC
                    else:
                        raw["audio"] += len(data) * 8.0 / (ASSUMED_AUDIO_KBPS * 1000.0) * AUDIO_TOKENS_PER_SEC
        raw["text"] = chars / TEXT_CHARS_PER_TOKEN
        return raw, payload

    def _calibrated(self, raw: dict, payload: int) -> TokenEstimate:
        calib = self.state["calib"]
        vals = {m: int(round(raw[m] * float(calib.get(m, 1.0) or 1.0))) for m in MODALITIES}
        total = sum(vals.values()) + REQUEST_OVERHEAD_TOKENS
        return TokenEstimate(text=vals["text"], audio=vals["audio"], video=vals["video"], image=vals["image"],
                             total=int(total), payload_bytes=int(payload))

    def estimate(self, contents: Any, system_instruction: Optional[str] = None,
                 media_resolution: Optional[str] = None) -> TokenEstimate:
        """Calibrated token estimate per modality plus inline payload bytes (base64 included)."""
        items = _normalize_items(contents)
        raw, payload = self._estimate_raw(items, system_instruction, media_resolution)
        return self._calibrated(raw, payload)

    def _calibrate(self, raw: dict, usage_by_modality: dict) -> None:
        for mod in MODALITIES:
            est = float(raw.get(mod, 0.0) or 0.0)
            actual = usage_by_modality.get(mod)
            if est < 50 or not isinstance(actual, (int, float)) or actual <= 0:
                continue
            ratio = clamp(float(actual) / est, CALIB_RANGE[0], CALIB_RANGE[1])
            old = float(self.state["calib"].get(mod, 1.0) or 1.0)
            new = clamp((1.0 - CALIB_ALPHA) * old + CALIB_ALPHA * ratio, CALIB_RANGE[0], CALIB_RANGE[1])
            self.state["calib"][mod] = round(new, 4)
            self._touch("calib", mod)

    def _adaptive_timeout(self, route: str, tokens: int) -> float:
        floor = float(ROUTES[route].get("min_timeout_s", 45))
        val = TIMEOUT_BASE_S + TIMEOUT_PER_1K_S * (max(0, int(tokens)) / 1000.0)
        return float(min(TIMEOUT_CAP_S, max(floor, val)))

    # ------------------------------------------------------------ File API

    @staticmethod
    def _file_state(f: Any) -> str:
        st = getattr(f, "state", None)
        if st is None:
            return ""
        return str(getattr(st, "name", st) or "").upper().split(".")[-1]

    def _upload_bytes(self, data: bytes, mime: str) -> dict:
        suffix = mimetypes.guess_extension(mime or "") or ".bin"
        fd, tmp = tempfile.mkstemp(prefix="tc_upload_", suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            client = self._client(300)
            try:
                up = client.files.upload(file=tmp, config={"mime_type": mime})
            except TypeError:
                up = client.files.upload(path=tmp, config={"mime_type": mime})
            t0 = self._now()
            while True:
                state = self._file_state(up)
                if state in ("", "ACTIVE") or "ACTIVE" in state:
                    break
                if "FAIL" in state:
                    raise GatewayError("❌ رفع الملف إلى Gemini فشل أثناء المعالجة")
                if self._now() - t0 > UPLOAD_ACTIVE_TIMEOUT_S:
                    raise GatewayError("❌ رفع الملف إلى Gemini أخذ وقت أطول من المسموح")
                self._sleep(3)
                up = client.files.get(name=up.name)
            return {"name": getattr(up, "name", ""), "uri": getattr(up, "uri", ""),
                    "mime_type": getattr(up, "mime_type", None) or mime}
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _prepare_uploads(self, items: list, payload_bytes: int) -> dict:
        """Upload the largest media items until the inline payload fits the guard."""
        if payload_bytes <= self.inline_limit_bytes:
            return {}
        order = sorted((i for i, it in enumerate(items) if it["kind"] in ("audio", "video", "image")),
                       key=lambda i: -len(items[i]["data"]))
        uploads: dict = {}
        remaining = payload_bytes
        mb = payload_bytes / (1024 * 1024)
        self._log(f"   📦 حجم الطلب {mb:.1f} MB أكبر من الحد الآمن، نرفع الوسائط الكبيرة عبر File API")
        try:
            for i in order:
                if remaining <= self.inline_limit_bytes:
                    break
                uploads[i] = self._upload_bytes(items[i]["data"], items[i]["mime_type"])
                remaining -= int(len(items[i]["data"]) * 4 / 3)
        except Exception:
            self._cleanup_uploads(uploads)
            raise
        if remaining > self.inline_limit_bytes:
            self._cleanup_uploads(uploads)
            raise GatewayError("❌ الطلب كبير جداً حتى بعد رفع الوسائط، صغّر عدد المرشحين في الدفعة")
        return uploads

    def _cleanup_uploads(self, uploads: dict) -> None:
        if not uploads:
            return
        client = self._client(60)
        for info in uploads.values():
            try:
                if info.get("name"):
                    client.files.delete(name=info["name"])
            except Exception:
                pass

    # ------------------------------------------------------- capabilities

    def _cap(self, model: str, name: str) -> bool:
        return (self.state["caps"].get(model) or {}).get(name, True) is not False

    def _set_cap(self, model: str, name: str, value: bool) -> None:
        caps = dict(self.state["caps"].get(model) or {})
        caps[name] = bool(value)
        self.state["caps"][model] = caps
        self._touch("caps", model)

    # ------------------------------------------------------------ parts

    def _video_part(self, it: dict, model: str, upload: Optional[dict]):
        fps = it.get("fps")
        use_vm = bool(fps) and self._cap(model, "video_metadata")
        try:
            vm = types.VideoMetadata(fps=float(fps)) if use_vm else None
        except Exception:
            vm = None
        try:
            if upload:
                kwargs = {"file_data": types.FileData(file_uri=upload["uri"], mime_type=upload["mime_type"])}
            else:
                kwargs = {"inline_data": types.Blob(data=it["data"], mime_type=it["mime_type"])}
            if vm is not None:
                kwargs["video_metadata"] = vm
            return types.Part(**kwargs)
        except Exception:
            if upload:
                return types.Part.from_uri(file_uri=upload["uri"], mime_type=upload["mime_type"])
            return types.Part.from_bytes(data=it["data"], mime_type=it["mime_type"])

    def _build_parts(self, items: list, model: str, uploads: dict) -> list:
        parts = []
        for i, it in enumerate(items):
            kind = it["kind"]
            if kind == "text":
                parts.append(it["text"])
            elif kind == "video":
                parts.append(self._video_part(it, model, uploads.get(i)))
            elif kind in ("audio", "image"):
                up = uploads.get(i)
                if up:
                    parts.append(types.Part.from_uri(file_uri=up["uri"], mime_type=up["mime_type"]))
                else:
                    parts.append(types.Part.from_bytes(data=it["data"], mime_type=it["mime_type"]))
            else:
                parts.append(it["part"])
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
        self._touch("thinking", model)
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
                self._touch("thinking", model)
                mode = "budget"
        if mode == "budget":
            try:
                return types.ThinkingConfig(thinking_budget=THINKING_BUDGETS.get(level, 1024))
            except Exception:
                self.state["thinking"][model] = "none"
                self._touch("thinking", model)
                return None
        return None

    def _schema_ok(self, model: str) -> bool:
        return self.state["schema"].get(model, True) is not False

    # -------------------------------------------------------------- the call

    def _make_config(self, model: str, schema: Optional[type], json_mode: bool, thinking: Optional[str],
                     system_instruction: Optional[str], temperature: Optional[float],
                     max_output_tokens: Optional[int], media_res: Optional[str]):
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
        if media_res and self._cap(model, "media_resolution"):
            enum_cls = getattr(types, "MediaResolution", None)
            value = getattr(enum_cls, f"MEDIA_RESOLUTION_{media_res.upper()}", None) if enum_cls is not None else None
            if value is not None:
                cfg["media_resolution"] = value
        try:
            return types.GenerateContentConfig(**cfg)
        except Exception:
            if "media_resolution" in cfg:
                cfg.pop("media_resolution", None)
                self._set_cap(model, "media_resolution", False)
                return types.GenerateContentConfig(**cfg)
            raise

    def _call(self, model: str, parts: list, schema: Optional[type], json_mode: bool, thinking: Optional[str],
              timeout_s: float, system_instruction: Optional[str], temperature: Optional[float],
              max_output_tokens: Optional[int], media_res: Optional[str]):
        config = self._make_config(model, schema, json_mode, thinking, system_instruction,
                                   temperature, max_output_tokens, media_res)
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
        is_api, code, message, status, details = _api_fields(exc)

        if not is_api:
            if "timeout" in name.lower() or "timed out" in low or isinstance(exc, TimeoutError):
                return _ErrInfo("transient", "انتهت مهلة الاتصال (timeout)", counted=True, hard=True)
            transient_names = ("ConnectError", "ConnectionError", "RemoteProtocolError", "ReadError",
                               "WriteError", "NetworkError", "TransportError", "ProtocolError")
            if any(t.lower() in name.lower() for t in transient_names) or "connection" in low:
                return _ErrInfo("transient", f"مشكلة اتصال ({name})")
            if "video_metadata" in low or "videometadata" in low:
                return _ErrInfo("vm_invalid", short_error(exc))
            if "media_resolution" in low or "mediaresolution" in low:
                return _ErrInfo("mr_invalid", short_error(exc))
            if schema_in_use and ("schema" in low or "pydantic" in low or "json" in low or isinstance(exc, (TypeError, ValueError))):
                return _ErrInfo("schema_invalid", short_error(exc))
            if "thinking" in low:
                return _ErrInfo("thinking_invalid", short_error(exc))
            return _ErrInfo("bad_request", short_error(exc))

        msg_low = (message + " " + status + " " + text).lower()

        if code == 404:
            return _ErrInfo("not_found", message[:160] or "404 NOT_FOUND")
        if code == 401:
            return _ErrInfo("fatal", "مفتاح Gemini مرفوض (401). تأكد من GEMINI_API_KEY في Secrets")
        if code == 403:
            return _ErrInfo("forbidden", message[:160] or "403 PERMISSION_DENIED")
        if code == 429:
            info = _ErrInfo("overloaded", message[:160] or "429 RESOURCE_EXHAUSTED")
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
            if info.kind == "overloaded":
                if "per day" in msg_low or "perday" in msg_low:
                    info.kind, info.quota_scope, info.quota_metric = "per_day", "day", "requests"
                elif "per minute" in msg_low or "perminute" in msg_low:
                    info.kind, info.quota_scope, info.quota_metric = "per_minute", "minute", "requests"
            if delay is None:
                match = re.search(r"retry in\s+(\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE)
                if match:
                    delay = float(match.group(1))
            info.retry_delay = delay
            if info.kind == "overloaded":
                info.kind = "transient"
                info.message = f"429 {status or 'RESOURCE_EXHAUSTED'}".strip()
            return info
        if code == 400:
            if "video_metadata" in msg_low or "videometadata" in msg_low or "fps" in msg_low:
                return _ErrInfo("vm_invalid", message[:160])
            if "media_resolution" in msg_low or "mediaresolution" in msg_low:
                return _ErrInfo("mr_invalid", message[:160])
            if "thinking" in msg_low:
                return _ErrInfo("thinking_invalid", message[:160])
            if schema_in_use and ("schema" in msg_low or "response_mime_type" in msg_low or "json" in msg_low):
                return _ErrInfo("schema_invalid", message[:160])
            return _ErrInfo("bad_request", message[:160] or "400 INVALID_ARGUMENT")
        if code in (408, 499, 500, 502, 503, 504) or (code is not None and code >= 500):
            hard = code in (408, 504) or "deadline" in msg_low
            return _ErrInfo("transient", f"{code} {status}".strip(), counted=code in (408, 504), hard=hard)
        if "overloaded" in msg_low or "unavailable" in msg_low:
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
    def _usage(resp: Any) -> tuple:
        """(usage dict as in v2, tokens per modality)."""
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return {}, {}
        out = {}
        for key, attr in (("input", "prompt_token_count"), ("output", "candidates_token_count"),
                          ("thinking", "thoughts_token_count"), ("total", "total_token_count")):
            val = getattr(meta, attr, None)
            if isinstance(val, int):
                out[key] = val
        by_mod: dict = {}
        for detail in getattr(meta, "prompt_tokens_details", None) or []:
            mod = _modality_name(getattr(detail, "modality", ""))
            count = getattr(detail, "token_count", None)
            if mod and isinstance(count, int):
                by_mod[mod] = by_mod.get(mod, 0) + count
        return out, by_mod

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

    def _parse_response(self, resp: Any, schema: Optional[type], json_mode: bool,
                        normalizer: Optional[Callable[[Any], Any]]) -> tuple:
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

    def _try_model(self, model: str, route: str, items: list, uploads: dict, raw_est: dict,
                   schema: Optional[type], json_mode: bool, thinking: Optional[str], timeout_s: float,
                   system_instruction: Optional[str], temperature: Optional[float],
                   max_output_tokens: Optional[int], normalizer: Optional[Callable[[Any], Any]],
                   est_tokens: int, label: str, media_res: Optional[str], is_last: bool) -> _Outcome:
        attempts = 0
        minute_tries = 0
        thinking_switches = 0
        caps_fixes = 0
        quick_used = False
        tag = f" [{label}]" if label else ""

        while True:
            if not self._is_available(model) or self._breaker_blocked(model):
                return _Outcome("parked", attempts, message="parked", model=model)
            self._pace(model, est_tokens)
            use_schema = schema if (schema is not None and self._schema_ok(model)) else None
            mode_ar = {"level": "تفكير", "budget": "تفكير", "none": "بدون تفكير"}[self._thinking_mode(model)]
            level_ar = {"low": "منخفض", "medium": "متوسط", "high": "عميق"}.get((thinking or "").lower(), "")
            self._log(f"🧠{tag} {ROUTES[route]['name_ar']}: الاتصال بـ {model} ({mode_ar} {level_ar})".rstrip())
            parts = self._build_parts(items, model, uploads)
            attempts += 1
            t0 = self._now()
            try:
                resp = self._call(model, parts, use_schema, json_mode or schema is not None, thinking,
                                  timeout_s, system_instruction, temperature, max_output_tokens, media_res)
            except Exception as exc:  # noqa: BLE001
                err = self._classify(exc, schema_in_use=use_schema is not None)
                self._record_attempt(model, est_tokens, counted=err.counted)
                bucket = self._bucket(model)

                if err.kind == "fatal":
                    raise GatewayError(f"❌ {err.message}") from exc

                if err.kind == "not_found":
                    self._park(model, now_utc() + _dt.timedelta(days=NOT_FOUND_PARK_DAYS), "not_found", err.message)
                    self._log(f"   🚫 {model} غير متاح لهذا المفتاح (404)، تم استبعاده {NOT_FOUND_PARK_DAYS} أيام")
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "forbidden":
                    self._park(model, now_utc() + _dt.timedelta(days=FORBIDDEN_PARK_DAYS), "forbidden", err.message)
                    self._log(f"   🚫 {model} مرفوض لهذا المفتاح (403)، تم استبعاده يوم")
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "per_day":
                    target = err.quota_model or model
                    if target and target != model:
                        self.state["aliases"][model] = target
                        self._touch("aliases", model)
                        bucket = target
                    if err.quota_value is not None and err.quota_metric == "requests":
                        self._set_limit(bucket, "rpd", err.quota_value)
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
                    return _Outcome("parked", attempts, message=err.message, model=model)

                if err.kind == "per_minute":
                    if err.quota_value:
                        self._set_limit(bucket, "tpm" if err.quota_metric == "tokens" else "rpm", err.quota_value)
                    minute_tries += 1
                    if minute_tries > MAX_PER_MINUTE_RETRIES:
                        return _Outcome("transient", attempts, message=err.message, model=model)
                    rpm = max(1, self._limits(bucket)["rpm"])
                    wait = max(err.retry_delay or 0.0, 60.0 / rpm) + random.uniform(1.0, 4.0)
                    self._log(f"   ⏱️ تجاوزنا حد الدقيقة على {model}، ننتظر {wait:.0f} ثانية")
                    self._sleep(wait)
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
                        self._touch("schema", model)
                        self._log(f"   🔧 {model} ما قبل الهيكل الصارم، نكمل بوضع JSON مع الإصلاح المحلي")
                        continue
                    return _Outcome("bad_request", attempts, message=err.message, model=model)

                if err.kind in ("vm_invalid", "mr_invalid"):
                    cap = "video_metadata" if err.kind == "vm_invalid" else "media_resolution"
                    caps_fixes += 1
                    if caps_fixes > 2 or not self._cap(model, cap):
                        return _Outcome("bad_request", attempts, message=err.message, model=model)
                    self._set_cap(model, cap, False)
                    what = "معدل الإطارات للفيديو" if cap == "video_metadata" else "دقة الوسائط"
                    self._log(f"   🔧 {model} ما قبل إعداد {what}، نعيد الطلب بدونه ونتذكر هذا")
                    continue

                if err.kind == "transient":
                    opened = self._strike(model, err.message, hard=err.hard)
                    if not opened and is_last and not quick_used:
                        quick_used = True
                        self._log(f"   ⏳ {model} مشغول ({err.message[:50]})، محاولة سريعة بعد {BREAKER['quick_retry_s']} ثواني لأنه آخر موديل متاح")
                        self._sleep(BREAKER["quick_retry_s"])
                        continue
                    if not opened:
                        self._log(f"   ↪️ {model} مشغول ({err.message[:50]})، ننتقل مباشرة للي بعده")
                    return _Outcome("transient", attempts, message=err.message, model=model)

                self._log(f"   ⚠️ {model} رفض الطلب ({err.message[:120]})")
                return _Outcome("bad_request", attempts, message=err.message, model=model)

            # HTTP success: the request counts against quota and the model is healthy.
            elapsed = self._now() - t0
            self._record_attempt(model, est_tokens, counted=True)
            self._health_success(model, elapsed)
            blocked = self._block_reason(resp)
            try:
                data, raw, dropped = self._parse_response(resp, use_schema or schema, json_mode, normalizer)
            except ValueError as exc:
                reason = blocked or short_error(exc, 120)
                self._log(f"   ⚠️ رد {model} ما انقرأ ({reason})")
                return _Outcome("parse_failed", attempts, message=reason, raw=self._response_text(resp), model=model)

            usage, by_mod = self._usage(resp)
            if by_mod:
                self._calibrate(raw_est, by_mod)
            in_tok = usage.get("input")
            tok_txt = f" | دخل {in_tok / 1000:.1f}k توكن" if isinstance(in_tok, int) else ""
            drop_txt = f" | تم تجاهل {dropped} عنصر غير صالح" if dropped else ""
            self._log(f"✅ استجاب {model} خلال {elapsed:.1f} ثانية{tok_txt}{drop_txt}")
            result = GatewayResult(
                data=data, raw_text=raw, model=model, route=route, attempts=attempts,
                elapsed_s=round(elapsed, 2), usage=usage, schema_used=use_schema is not None,
                dropped_items=dropped, usage_by_modality=by_mod,
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
        """Send one request through the route's healthy model chain.

        contents: str, or a list of str / video_item / audio_item / image_item dicts / v2 audio
                  dicts {"data", "mime_type", "seconds"} / raw bytes (mp3) / types.Part.
        schema:   Pydantic BaseModel class for structured output (data is an instance).
                  Without schema, json_mode=True returns a parsed object, else plain text.
        normalizer: optional callable applied to the parsed object before validation.

        Raises QuotaPause, ParseFailure or GatewayError."""
        if route not in ROUTES:
            raise GatewayError(f"❌ مسار غير معروف: {route}")
        if schema is not None and not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            raise GatewayError("❌ schema لازم يكون كلاس Pydantic")
        cfg = ROUTES[route]
        media_res = cfg.get("media")
        thinking = thinking or cfg["thinking"]
        json_flag = bool(json_mode) if json_mode is not None else schema is not None
        items = _normalize_items(contents)
        raw_est, payload = self._estimate_raw(items, system_instruction, media_res)
        estimate = self._calibrated(raw_est, payload)
        est = int(est_tokens) if est_tokens else estimate.total
        timeout = float(timeout_s) if timeout_s else self._adaptive_timeout(route, est)

        deadline = self._now() + self.max_busy_wait_s
        total_attempts = 0
        last_msg = ""
        any_bad = False
        skip: set = set()
        parse_fallback_used = False
        last_parse: Optional[_Outcome] = None
        announced_blocked = False
        uploads: dict = {}
        try:
            uploads = self._prepare_uploads(items, payload)
            while True:
                candidates = [m for m in self._route_candidates(route) if m not in skip]
                if not announced_blocked:
                    blocked_now = [m for m in self._blocked_candidates(route) if m not in skip]
                    if blocked_now:
                        self._log(f"   ⏭️ تخطينا موديلات موقفة مؤقتاً: {'، '.join(blocked_now)}")
                    announced_blocked = True
                if not candidates:
                    if last_parse is not None:
                        raise ParseFailure(route, last_parse.model, last_parse.raw, last_parse.message)
                    blocked = [m for m in self._blocked_candidates(route) if m not in skip]
                    if blocked:
                        wake = min(self._breaker_open_until(m) or self._now() for m in blocked)
                        wait = max(1.0, wake - self._now() + 1.0)
                        if self._now() + wait <= deadline:
                            self._log(f"🌩️ كل موديلات {cfg['name_ar']} موقفة مؤقتاً، ننتظر {format_duration_ar(wait)} ونجرب أقربها")
                            self._sleep(wait)
                            continue
                        raise QuotaPause(
                            route, "busy", None,
                            f"⏸️ سيرفرات Gemini مشغولة جداً حالياً لموديلات {cfg['name_ar']}. "
                            f"التقدم محفوظ، شغّل الخلية مرة ثانية بعد {format_duration_ar(wake - self._now())} تقريباً وبيكمل من نفس النقطة.",
                        )
                    if any_bad:
                        raise GatewayError(f"❌ رفضت كل موديلات {cfg['name_ar']} الطلب: {last_msg}")
                    raise self._quota_pause(route)

                any_transient = False
                for idx, model in enumerate(candidates):
                    outcome = self._try_model(
                        model, route, items, uploads, raw_est, schema, json_flag, thinking, timeout,
                        system_instruction, temperature, max_output_tokens, normalizer, est, label,
                        media_res, is_last=(idx == len(candidates) - 1),
                    )
                    total_attempts += outcome.attempts
                    if outcome.kind == "ok" and outcome.result is not None:
                        outcome.result.attempts = total_attempts
                        self._set_sticky(route, model)
                        return outcome.result
                    if outcome.message:
                        last_msg = f"{outcome.model}: {outcome.message}"
                    if outcome.kind == "parse_failed":
                        skip.add(model)
                        last_parse = outcome
                        if parse_fallback_used:
                            raise ParseFailure(route, outcome.model, outcome.raw, outcome.message)
                        parse_fallback_used = True
                        self._log("   🔁 نجرب موديل ثاني مرة وحدة بسبب الرد اللي ما انقرأ")
                        continue
                    if outcome.kind == "bad_request":
                        skip.add(model)
                        any_bad = True
                    elif outcome.kind == "transient":
                        any_transient = True

                if any_transient:
                    if self._now() >= deadline:
                        raise QuotaPause(
                            route, "busy", None,
                            f"⏸️ سيرفرات Gemini مشغولة جداً حالياً لموديلات {cfg['name_ar']}. "
                            "التقدم محفوظ، شغّل الخلية مرة ثانية بعد شوي وبيكمل من نفس النقطة.",
                        )
                    self._sleep(BREAKER["round_pause_s"])
        except (QuotaPause, GatewayError):
            self.flush()
            raise
        finally:
            self._cleanup_uploads(uploads)
            self._maybe_save()

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
        """Requests the route can still make today across its models (breaker state ignored,
        open breakers recover). Estimate when limits were not learned yet."""
        total = 0
        seen: set = set()
        for model in self._route_models(route):
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
            for model in self._route_models(route):
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
        self._maybe_save()

        lines = ["📊 خطة طلبات Gemini لهذا التشغيل:"]
        for route, info in routes_out.items():
            name_ar = ROUTES.get(route, {}).get("name_ar", route)
            models_txt = "، ".join(f"{u['model']} ×{u['requests']} (يتبقى {u['left_after']})" for u in info["models"]) or "ما فيه موديل متاح"
            mark = "✅" if info["available"] >= info["needed"] else "⚠️"
            lines.append(f"   {mark} {name_ar}: {info['needed']} طلب ← {models_txt}")
        if estimated:
            lines.append("   ℹ️ بعض الحدود تقديرية (Flash 20 وLite 500 طلب يومياً) لين نتعلم الحد الحقيقي من رد Gemini")
        if not ok:
            lines.append(f"   ⏸️ الحصة ما تكفي كاملة اليوم، الحصة ترجع {pacific_reset_riyadh()[1]}")
        return Budget(ok=ok, routes=routes_out, estimated=estimated, text_ar="\n".join(lines))

    # ------------------------------------------------------------ reports

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
                until = _ts(entry.get("until"))
                left = format_duration_ar(until - now_utc().timestamp()) if until else "?"
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

    def health_report_ar(self) -> str:
        """Arabic report of breaker state, success rate, latency and sticky routes."""
        now = self._now()
        lines = ["🩺 صحة موديلات Gemini:"]
        names = sorted(set(self.state["perf"].keys()) | set(self.state["breaker"].keys()), key=_model_sort_key)
        if not names:
            lines.append("   ما فيه بيانات بعد، أول طلب بيبني الصورة")
        for model in names:
            perf = self.state["perf"].get(model) or {}
            entry = self.state["breaker"].get(model) or {}
            until = _ts(entry.get("open_until"))
            if until is not None and now < until:
                state_txt = f"🔴 موقف مؤقتاً، يرجع بعد {format_duration_ar(until - now)}"
            elif entry.get("opened") and until is not None and now < until + BREAKER["probe_grace_s"]:
                state_txt = "🟡 جاهز لمحاولة تجريبية"
            elif entry.get("strikes"):
                state_txt = f"🟠 تعثر {len(entry['strikes'])} مرة مؤخراً"
            else:
                state_txt = "🟢 شغال"
            ok = int(perf.get("ok", 0) or 0)
            fail = int(perf.get("fail", 0) or 0)
            lat = perf.get("ewma_latency_s")
            lat_txt = f" | متوسط الرد {float(lat):.0f} ثانية" if isinstance(lat, (int, float)) else ""
            lines.append(f"   • {model}: {state_txt} | نجح {ok} وتعثر {fail}{lat_txt}")
        if self.state["sticky"]:
            lines.append("📌 الموديل المفضل لكل مسار (آخر واحد نجح):")
            for route, model in sorted(self.state["sticky"].items()):
                lines.append(f"   • {ROUTES.get(route, {}).get('name_ar', route)}: {model}")
        calib = self.state["calib"]
        lines.append("🎯 معايرة تقدير التوكن: " + "، ".join(
            f"{name} ×{float(calib.get(key, 1.0)):.2f}"
            for key, name in (("text", "النص"), ("audio", "الصوت"), ("video", "الفيديو"), ("image", "الصور"))))
        return "\n".join(lines)

    def reset_model_health(self) -> None:
        """Clear parked models and open breakers (for example after linking billing)."""
        for section in ("health", "breaker"):
            for key in list(self.state[section].keys()):
                self.state[section].pop(key, None)
                self._touch(section, key)
        self.flush()


# ============================================================================
# Offline self test (fake client, no network, no real sleeping)
# ============================================================================

class _FakeAPIError(Exception):
    def __init__(self, code: int, status: str, message: str):
        super().__init__(f"{code} {status}. {message}")
        self.code = code
        self.status = status
        self.message = message
        self.details = None


class _FakeObj:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


def _fake_response(text: str, by_mod: Optional[dict] = None, prompt_tokens: int = 1000) -> Any:
    details = [_FakeObj(modality=_FakeObj(name=k.upper()), token_count=v) for k, v in (by_mod or {}).items()]
    usage = _FakeObj(prompt_token_count=prompt_tokens, candidates_token_count=50, thoughts_token_count=0,
                     total_token_count=prompt_tokens + 50, prompt_tokens_details=details)
    return _FakeObj(text=text, candidates=[], usage_metadata=usage, parsed=None, prompt_feedback=None)


class _FakeModels:
    def __init__(self, names: Sequence[str], handlers: dict):
        self.names = list(names)
        self.handlers = handlers
        self.calls: list = []

    def list(self):
        return [_FakeObj(name=f"models/{n}", supported_actions=["generateContent"]) for n in self.names]

    def generate_content(self, model: str, contents: Any, config: Any = None):
        self.calls.append((model, contents, config))
        handler = self.handlers.get(model)
        if handler is None:
            raise _FakeAPIError(404, "NOT_FOUND", "model not found")
        return handler(model, contents, config)


class _FakeFiles:
    def __init__(self):
        self.uploaded: list = []
        self.deleted: list = []

    def upload(self, file: Any = None, config: Any = None, **_kw: Any):
        name = f"files/fake{len(self.uploaded)}"
        mime = (config or {}).get("mime_type") if isinstance(config, dict) else None
        self.uploaded.append(name)
        return _FakeObj(name=name, uri=f"https://fake/{name}", mime_type=mime, state=_FakeObj(name="PROCESSING"))

    def get(self, name: str):
        return _FakeObj(name=name, uri=f"https://fake/{name}", mime_type="video/mp4", state=_FakeObj(name="ACTIVE"))

    def delete(self, name: str):
        self.deleted.append(name)


class _FakeClient:
    def __init__(self, names: Sequence[str], handlers: dict):
        self.models = _FakeModels(names, handlers)
        self.files = _FakeFiles()


class _SelfTestSchema(BaseModel):
    items: list[str] = []


def _ok_json(_m: str, _c: Any, _cfg: Any):
    return _fake_response('{"items": ["تمام"]}')


def _busy(_m: str, _c: Any, _cfg: Any):
    raise _FakeAPIError(503, "UNAVAILABLE", "The model is overloaded. Please try again later.")


def _garbage(_m: str, _c: Any, _cfg: Any):
    return _fake_response("هذا مو JSON أبداً")


def _make_test_gateway(folder: str, name: str, names: Sequence[str], handlers: dict,
                       max_busy_wait_s: float = 60.0) -> tuple:
    clock = {"t": 1_800_000_000.0}
    gw = GeminiGateway("selftest_key_0000", os.path.join(folder, f"{name}.json"),
                       max_busy_wait_s=max_busy_wait_s, verbose=False)
    client = _FakeClient(names, handlers)
    gw._client_factory = lambda _timeout_ms: client
    gw._now = lambda: clock["t"]

    def fake_sleep(seconds: float) -> None:
        clock["t"] += max(0.0, float(seconds))

    gw._sleep = fake_sleep
    return gw, client, clock


def selftest(workdir: Optional[str] = None) -> dict:
    """Offline checks with a fake Gemini client (no network, no quota):
      1. ranking and stickiness: busy newest models are skipped, the winner is reused first
      2. breaker opens after 2 strikes, cooldown 12 min, doubles on a failed probe, caps at 30 min
      3. all models busy beyond the budget raises QuotaPause("busy") quickly
      4. a response that cannot be parsed moves to the next model once, then ParseFailure
      5. a model that rejects video_metadata is called again without it and remembered
      6. self calibration moves the video factor from usage_metadata
      7. payload above the guard goes through the File API and is deleted after
      8. two gateways on one state file merge their ledgers
    Returns a dict of results with an "all_ok" flag."""
    folder = workdir or tempfile.mkdtemp(prefix="tc_gateway_selftest_")
    os.makedirs(folder, exist_ok=True)
    results: dict = {}
    log("🧪 فحص ذاتي لبوابة Gemini (عميل وهمي بدون إنترنت)...")

    def mark(name: str, ok: bool, detail: str) -> None:
        results[name] = {"ok": bool(ok), "detail": detail}
        log(f"   {'✅' if ok else '❌'} {name}: {detail}")

    flash = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3-flash-preview"]
    lite = ["gemini-3.1-flash-lite"]

    # 1. ranking and stickiness
    try:
        handlers = {"gemini-3.8-flash": _busy, "gemini-3.7-flash": _busy,
                    "gemini-3-flash-preview": _ok_json, "gemini-3.1-flash-lite": _ok_json}
        gw, client, _clock = _make_test_gateway(folder, "rank", flash + lite, handlers)
        r1 = gw.generate("scout", ["اختبار"], schema=_SelfTestSchema)
        n1 = len(client.models.calls)
        r2 = gw.generate("scout", ["اختبار"], schema=_SelfTestSchema)
        n2 = len(client.models.calls) - n1
        ok = r1.model == "gemini-3-flash-preview" and r2.model == "gemini-3-flash-preview" and n1 == 3 and n2 == 1
        mark("ranking_sticky", ok, f"الطلب الأول {n1} محاولات، الثاني {n2} محاولة مباشرة على {r2.model}")
    except Exception as exc:
        mark("ranking_sticky", False, f"خطأ: {short_error(exc)}")

    # 2. breaker timing
    try:
        gw, _client, clock = _make_test_gateway(folder, "breaker", flash, {})
        m = "gemini-3.8-flash"
        first = gw._strike(m, "503", hard=False)
        second = gw._strike(m, "503", hard=False)
        cd1 = gw.state["breaker"][m]["cooldown_s"]
        blocked = gw._breaker_blocked(m)
        clock["t"] += cd1 + 1
        half_open = not gw._breaker_blocked(m)
        gw._strike(m, "503", hard=False)
        cd2 = gw.state["breaker"][m]["cooldown_s"]
        clock["t"] += cd2 + 1
        gw._strike(m, "503", hard=False)
        cd3 = gw.state["breaker"][m]["cooldown_s"]
        clock["t"] += cd3 + 1
        gw._health_success(m, 5.0)
        closed = m not in gw.state["breaker"]
        ok = (not first and second and blocked and half_open and cd1 == 720 and cd2 == 1440
              and cd3 == 1800 and closed)
        mark("breaker", ok, f"يفتح بعد ضربتين، التوقف {cd1} ثم {cd2} ثم {cd3} ثانية، ويقفل بعد النجاح")
    except Exception as exc:
        mark("breaker", False, f"خطأ: {short_error(exc)}")

    # 3. all busy -> QuotaPause busy
    try:
        handlers = {n: _busy for n in flash + lite}
        gw, client, clock = _make_test_gateway(folder, "busy", flash + lite, handlers, max_busy_wait_s=60.0)
        t0 = clock["t"]
        reason = ""
        try:
            gw.generate("scout", ["اختبار"], schema=_SelfTestSchema)
        except QuotaPause as pause:
            reason = pause.reason
        spent = clock["t"] - t0
        all_open = all(gw._breaker_blocked(n) for n in flash + lite)
        ok = reason == "busy" and all_open and spent < 200
        mark("all_busy", ok, f"توقف نظيف بسبب {reason or 'لا شي'} بعد {spent:.0f} ثانية وهمية و{len(client.models.calls)} محاولة")
    except Exception as exc:
        mark("all_busy", False, f"خطأ: {short_error(exc)}")

    # 4. parse fallback
    try:
        handlers = {"gemini-3.8-flash": _garbage, "gemini-3.7-flash": _ok_json}
        gw, _client, _clock = _make_test_gateway(folder, "parse", flash[:2], handlers)
        r = gw.generate("general", ["اختبار"], schema=_SelfTestSchema)
        fallback_ok = r.model == "gemini-3.7-flash" and r.data.items == ["تمام"]
        handlers2 = {"gemini-3.8-flash": _garbage, "gemini-3.7-flash": _garbage}
        gw2, client2, _clock2 = _make_test_gateway(folder, "parse2", flash[:2], handlers2)
        raised = False
        try:
            gw2.generate("general", ["اختبار"], schema=_SelfTestSchema)
        except ParseFailure:
            raised = True
        ok = fallback_ok and raised and len(client2.models.calls) == 2
        mark("parse_fallback", ok, "الرد التالف ينتقل لموديل ثاني مرة وحدة، وبعدها يرفع ParseFailure")
    except Exception as exc:
        mark("parse_fallback", False, f"خطأ: {short_error(exc)}")

    # 5. video_metadata capability cache
    try:
        seen_vm: list = []

        def strict_vm(_m: str, contents: Any, _cfg: Any):
            has_vm = any(getattr(p, "video_metadata", None) is not None for p in contents if not isinstance(p, str))
            seen_vm.append(has_vm)
            if has_vm:
                raise _FakeAPIError(400, "INVALID_ARGUMENT", "Invalid value at 'video_metadata.fps'")
            return _fake_response('{"items": ["فيديو"]}')

        gw, _client, _clock = _make_test_gateway(folder, "caps", ["gemini-3-flash-preview"],
                                                 {"gemini-3-flash-preview": strict_vm})
        vid = video_item(b"\x00" * 4000, seconds=10.0, fps=2.0)
        r = gw.generate("critic", ["شوف المقطع", vid], schema=_SelfTestSchema)
        ok = (r.data.items == ["فيديو"] and seen_vm == [True, False]
              and gw.state["caps"]["gemini-3-flash-preview"]["video_metadata"] is False)
        mark("caps_cache", ok, "الموديل رفض معدل الإطارات، أعدنا الطلب بدونه وتذكرنا ذلك")
    except Exception as exc:
        mark("caps_cache", False, f"خطأ: {short_error(exc)}")

    # 6. calibration
    try:
        def heavy_video(_m: str, _c: Any, _cfg: Any):
            return _fake_response('{"items": ["ok"]}', by_mod={"video": 20000, "audio": 250, "text": 30})

        gw, _client, _clock = _make_test_gateway(folder, "calib", ["gemini-3-flash-preview"],
                                                 {"gemini-3-flash-preview": heavy_video})
        before = gw.estimate(["نص", video_item(b"\x00" * 4000, seconds=10.0, fps=2.0)], media_resolution="low")
        gw.generate("critic", ["نص", video_item(b"\x00" * 4000, seconds=10.0, fps=2.0)], schema=_SelfTestSchema)
        factor = gw.state["calib"]["video"]
        after = gw.estimate(["نص", video_item(b"\x00" * 4000, seconds=10.0, fps=2.0)], media_resolution="low")
        ok = abs(factor - 1.9) < 0.01 and before.video == 1400 and after.video > before.video
        mark("calibration", ok, f"معامل الفيديو صار ×{factor:.2f} (التقدير {before.video} ثم {after.video} توكن)")
    except Exception as exc:
        mark("calibration", False, f"خطأ: {short_error(exc)}")

    # 7. payload guard and File API
    try:
        used_file: list = []

        def file_aware(_m: str, contents: Any, _cfg: Any):
            used_file.append(any(getattr(p, "file_data", None) is not None for p in contents if not isinstance(p, str)))
            return _fake_response('{"items": ["ملف"]}')

        gw, client, _clock = _make_test_gateway(folder, "upload", ["gemini-3-flash-preview"],
                                                {"gemini-3-flash-preview": file_aware})
        gw.inline_limit_bytes = 2000
        gw.generate("critic", ["مقطع كبير", video_item(b"\x01" * 6000, seconds=20.0, fps=2.0)], schema=_SelfTestSchema)
        ok = client.files.uploaded == client.files.deleted and len(client.files.uploaded) == 1 and used_file == [True]
        mark("payload_guard", ok, f"رفعنا {len(client.files.uploaded)} ملف عبر File API وحذفناه بعد الاستخدام")
    except Exception as exc:
        mark("payload_guard", False, f"خطأ: {short_error(exc)}")

    # 8. merged state saves
    try:
        handlers = {"gemini-3-flash-preview": _ok_json}
        gw_a, _ca, _cla = _make_test_gateway(folder, "merge", ["gemini-3-flash-preview"], handlers)
        gw_b, _cb, _clb = _make_test_gateway(folder, "merge", ["gemini-3-flash-preview"], handlers)
        gw_a.generate("scout", ["أ"], schema=_SelfTestSchema)
        gw_b.generate("scout", ["ب"], schema=_SelfTestSchema)
        gw_a.flush()
        gw_b.flush()
        disk = load_json_safe(os.path.join(folder, "merge.json"), default={}, quiet=True) or {}
        count = int(((disk.get("ledger") or {}).get(pacific_date_str()) or {}).get("gemini-3-flash-preview", 0))
        ok = count == 2
        mark("state_merge", ok, f"السجل على Drive فيه {count} طلب من جلستين بدون ما وحدة تمسح الثانية")
    except Exception as exc:
        mark("state_merge", False, f"خطأ: {short_error(exc)}")

    all_ok = all(v["ok"] for v in results.values())
    results["all_ok"] = all_ok
    log("🎉 كل فحوصات البوابة ناجحة" if all_ok else "⚠️ فيه فحوصات فشلت في البوابة")
    return results


# === END OF gemini_gateway.py ===
