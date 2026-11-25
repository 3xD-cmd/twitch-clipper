"""
golden_engine.py
Golden Clips training for the TwitchClipper pipeline (v3, new module).

The editor drops 8 to 15 of his best finished TikTok / Reels edits of this streamer into
TwitchClipper/memory/golden_clips/. Gemini watches each edit once (picture and sound),
ignores the CapCut post production (zooms, memes, SFX, music, captions) and reverse
engineers the raw clip selection: what happened in game, why second 0 is the hook, what
she said and with what energy, where jump cuts removed dead air, and what closed the clip.
Every analysis is cached for good in memory/golden_examples.json, so adding clips later only
analyzes the new files. One text request distills all analyses into the Golden DNA, and
local Python computes the numeric stats used by taste_engine.apply_golden_priors.

Public contract
  GOLDEN_EXTS, MOMENT_TYPES, ENDING_TYPES
  GoldenBeat, GoldenAnalysis, GoldenArchetype, GoldenDNA   (lenient Pydantic models)
  golden_key(path) -> str                                  (SHA1 of size + first 8 MB + last 1 MB)
  scan_golden(memory, data=None) -> [{"path", "name", "key", "size", "state"}]
  load_golden(memory) -> dict / save_golden(memory, data) -> str
  train_golden(drive_root, api_key, max_new=None, redistill=False, gateway=None) -> dict
  golden_stats(golden) -> {"n", "duration_s", "hook_latency_s", "ending_types", "cuts_per_min", "moment_types"}
  golden_context(golden, route, category=None, moment_type=None, max_chars=4000) -> str
  set_excluded(memory, key, excluded) -> dict
  golden_summary_ar(golden) -> str
  selftest(workdir=None) -> dict   (offline, fake gateway, synthetic clips)

Safety
  * memory/golden_clips/ is read only for the pipeline: files are hashed and transcoded from,
    never moved, renamed or deleted. Transcodes live in /tmp and are removed after each clip.
  * Analyses are never pruned: a deleted video keeps its analysis with file_missing=True.
  * golden_examples.json is written atomically with a .bak copy after every clip, so a
    QuotaPause or a crash resumes at the next file.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import time
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator

from common import (
    GOLDEN_SCHEMA,
    TMP_ROOT,
    MemoryPaths,
    atomic_write_json,
    coerce_int,
    coerce_number,
    ensure_dir,
    load_json_safe,
    log,
    mmss_to_sec,
    normalize_arabic,
    now_iso,
    sec_to_mmss,
    short_error,
)
from gemini_gateway import GatewayError, GeminiGateway, ParseFailure, QuotaPause, video_item
from media_engine import transcode_for_gemini


# ============================================================================
# Constants
# ============================================================================

GOLDEN_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
MOMENT_TYPES = ("فشل_بصري", "غدر_ومقلب", "ضحك_هستيري", "سالفة", "تفاعل_شات", "رعب_وفجعة", "لعب_قوي")
ENDING_TYPES = ("قفلة_كلام", "ذروة_ضحك", "ردة_فعل", "صمت_مضحك")

MOMENT_TYPES_AR = {
    "فشل_بصري": "فشل أو حادث بصري (طيحة، دفة بالغلط، موت مضحك)",
    "غدر_ومقلب": "غدر أو مقلب في اللعبة",
    "ضحك_هستيري": "ضحك هستيري هو نجم اللقطة",
    "سالفة": "سالفة أو كلام مضحك",
    "تفاعل_شات": "تفاعل مع الشات",
    "رعب_وفجعة": "رعب أو فجعة",
    "لعب_قوي": "لعب قوي أو فوز",
}
ENDING_TYPES_AR = {
    "قفلة_كلام": "قفلة بجملة",
    "ذروة_ضحك": "على ذروة الضحك",
    "ردة_فعل": "على ردة فعل",
    "صمت_مضحك": "على صمت مضحك",
}

# Studio and taste categories -> moment types used to pick similar golden examples
CATEGORY_TO_MOMENTS = {
    "ضحك": ("ضحك_هستيري", "فشل_بصري"),
    "مقلب_وغدر": ("غدر_ومقلب",),
    "سالفة": ("سالفة",),
    "تفاعل_شات": ("تفاعل_شات",),
    "صراخ_وتوتر": ("رعب_وفجعة",),
    "قيمينق": ("لعب_قوي", "فشل_بصري"),
}

TRANSCODE_FPS = 3.0
TRANSCODE_SHORT_SIDE = 480
TRANSCODE_AUDIO_KBPS = 64
TRANSCODE_MAX_SECONDS = 240.0
HASH_HEAD_BYTES = 8 * 1024 * 1024
HASH_TAIL_BYTES = 1 * 1024 * 1024
MAX_ATTEMPTS = 3
SCOUT_CHAR_CAP = 2700
DISTILL_ITEM_CHARS = 1800


# ============================================================================
# Lenient helpers
# ============================================================================

def _text(value: Any, limit: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "، ".join(str(v) for v in value if str(v).strip())
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text[:limit]


def _text_list(value: Any, max_items: int = 8, limit: int = 240) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[\n؛;]+|(?:^|\s)[•\*]\s*", value)
    elif isinstance(value, (list, tuple)):
        parts = []
        for v in value:
            if isinstance(v, dict):
                v = " ".join(str(x) for x in v.values())
            parts.append(str(v))
    else:
        parts = [str(value)]
    out = []
    for p in parts:
        t = _text(p, limit).strip(" .،")
        if t and t not in out:
            out.append(t)
        if len(out) >= max_items:
            break
    return out


def _norm_choice(value: Any, choices: Sequence[str], default: str) -> str:
    raw = normalize_arabic(str(value or "")).replace(" ", "_").strip("_")
    if not raw:
        return default
    normed = {normalize_arabic(c): c for c in choices}
    if raw in normed:
        return normed[raw]
    for key, choice in normed.items():
        if key in raw or raw in key:
            return choice
    raw_parts = set(raw.split("_"))
    best, best_hits = default, 0
    for key, choice in normed.items():
        hits = len(raw_parts & set(key.split("_")))
        if hits > best_hits:
            best, best_hits = choice, hits
    return best


def _mmss(value: Any) -> str:
    if isinstance(value, (int, float)):
        return sec_to_mmss(value)
    sec = mmss_to_sec(value)
    return sec_to_mmss(sec) if sec is not None else _text(value, 12)


# ============================================================================
# Pydantic models (Gemini response schemas)
# ============================================================================

class GoldenBeat(BaseModel):
    model_config = ConfigDict(extra="ignore")
    at: str = ""
    what: str = ""

    @field_validator("at", mode="before")
    @classmethod
    def _at(cls, v: Any) -> str:
        return _mmss(v)

    @field_validator("what", mode="before")
    @classmethod
    def _what(cls, v: Any) -> str:
        return _text(v, 200)


class GoldenAnalysis(BaseModel):
    model_config = ConfigDict(extra="ignore")
    game: str = ""
    moment_type: str = "ضحك_هستيري"
    visual_story: str = ""
    hook_at_zero: str = ""
    hook_latency_s: float = 0.0
    first_words: str = ""
    key_lines: list[str] = []
    vocal_energy: str = ""
    reaction: str = ""
    jump_cuts: list[GoldenBeat] = []
    ending: str = ""
    ending_type: str = "ردة_فعل"
    raw_duration_guess_s: float = 0.0
    pacing: str = ""
    why_it_works: str = ""
    selection_rules: list[str] = []
    postprod_ignored: list[str] = []

    @field_validator("game", "visual_story", "hook_at_zero", "first_words", "vocal_energy", "reaction",
                     "ending", "pacing", "why_it_works", mode="before")
    @classmethod
    def _plain(cls, v: Any) -> str:
        return _text(v, 600)

    @field_validator("moment_type", mode="before")
    @classmethod
    def _moment(cls, v: Any) -> str:
        return _norm_choice(v, MOMENT_TYPES, "ضحك_هستيري")

    @field_validator("ending_type", mode="before")
    @classmethod
    def _ending(cls, v: Any) -> str:
        return _norm_choice(v, ENDING_TYPES, "ردة_فعل")

    @field_validator("hook_latency_s", "raw_duration_guess_s", mode="before")
    @classmethod
    def _seconds(cls, v: Any) -> float:
        if isinstance(v, str) and ":" in v:
            sec = mmss_to_sec(v)
            return float(sec) if sec is not None else 0.0
        num = coerce_number(v, 0.0, fraction_scale=None)
        return float(max(0.0, num or 0.0))

    @field_validator("key_lines", mode="before")
    @classmethod
    def _lines(cls, v: Any) -> list:
        return _text_list(v, 4, 200)

    @field_validator("selection_rules", mode="before")
    @classmethod
    def _rules(cls, v: Any) -> list:
        return _text_list(v, 5, 240)

    @field_validator("postprod_ignored", mode="before")
    @classmethod
    def _post(cls, v: Any) -> list:
        return _text_list(v, 10, 80)

    @field_validator("jump_cuts", mode="before")
    @classmethod
    def _cuts(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, str):
            v = [{"at": m.group(0), "what": ""} for m in re.finditer(r"\d{1,2}:\d{2}", v)]
        out = []
        for item in v if isinstance(v, (list, tuple)) else []:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (str, int, float)):
                out.append({"at": item, "what": ""})
        return out[:30]


class GoldenArchetype(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str = ""
    signature: str = ""
    how_to_spot: str = ""

    @field_validator("name", "signature", "how_to_spot", mode="before")
    @classmethod
    def _plain(cls, v: Any) -> str:
        return _text(v, 400)


class GoldenDNA(BaseModel):
    model_config = ConfigDict(extra="ignore")
    principles: list[str] = []
    archetypes: list[GoldenArchetype] = []
    hook_rules: list[str] = []
    cut_rules: list[str] = []
    ending_rules: list[str] = []
    avoid: list[str] = []

    @field_validator("principles", mode="before")
    @classmethod
    def _principles(cls, v: Any) -> list:
        return _text_list(v, 10, 260)

    @field_validator("hook_rules", "cut_rules", "ending_rules", "avoid", mode="before")
    @classmethod
    def _rules(cls, v: Any) -> list:
        return _text_list(v, 6, 240)

    @field_validator("archetypes", mode="before")
    @classmethod
    def _arch(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, dict):
            v = [v]
        out = []
        for item in v if isinstance(v, (list, tuple)) else []:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, str) and item.strip():
                out.append({"name": item})
        return out[:6]


def _unwrap(obj: Any) -> Any:
    """Normalizer for model output: unwrap {"analysis": {...}} or a one item list."""
    if isinstance(obj, list):
        return obj[0] if obj and isinstance(obj[0], dict) else {}
    if isinstance(obj, dict) and len(obj) == 1:
        only = next(iter(obj.values()))
        if isinstance(only, dict):
            return only
    return obj


# ============================================================================
# Identity, scanning, storage
# ============================================================================

def golden_key(path: str) -> str:
    """Rename proof identity: SHA1 of the file size, its first 8 MB and its last 1 MB."""
    size = os.path.getsize(path)
    h = hashlib.sha1()
    h.update(f"{size}:".encode("ascii"))
    with open(path, "rb") as f:
        h.update(f.read(HASH_HEAD_BYTES))
        if size > HASH_HEAD_BYTES:
            f.seek(max(HASH_HEAD_BYTES, size - HASH_TAIL_BYTES))
            h.update(f.read(HASH_TAIL_BYTES))
    return h.hexdigest()


def _empty_golden() -> dict:
    return {"schema": GOLDEN_SCHEMA, "items": {}, "dna": {}, "stats": {}, "index": {}, "updated_at": ""}


def _normalize_item(key: str, raw: Any) -> dict:
    item = dict(raw) if isinstance(raw, dict) else {}
    item["key"] = str(item.get("key") or key)
    item["file"] = str(item.get("file") or "")
    item["size"] = coerce_int(item.get("size"), 0, lo=0)
    item["file_missing"] = bool(item.get("file_missing", False))
    item["excluded"] = bool(item.get("excluded", False))
    status = str(item.get("status") or "new")
    item["status"] = status if status in ("new", "done", "failed") else "new"
    item["error"] = str(item.get("error") or "")
    item["attempts"] = coerce_int(item.get("attempts"), 0, lo=0)
    item.setdefault("model", "")
    item.setdefault("added_at", "")
    item.setdefault("analyzed_at", "")
    dur = coerce_number(item.get("duration_s"), None)
    item["duration_s"] = round(float(dur), 2) if dur else 0.0
    item["transcode"] = dict(item.get("transcode") or {})
    item["usage"] = dict(item.get("usage") or {})
    analysis = item.get("analysis")
    if isinstance(analysis, dict) and analysis:
        try:
            item["analysis"] = GoldenAnalysis.model_validate(analysis).model_dump()
        except Exception:
            item["analysis"] = {}
    else:
        item["analysis"] = {}
    if item["status"] == "done" and not item["analysis"]:
        item["status"] = "new"
    return item


def load_golden(memory: MemoryPaths) -> dict:
    """golden_examples.json with every key present and every item normalized."""
    raw = load_json_safe(memory.golden_examples, default={}, expected_types=(dict,), quiet=True) or {}
    data = _empty_golden()
    for key in ("dna", "stats", "index"):
        if isinstance(raw.get(key), dict):
            data[key] = raw[key]
    data["updated_at"] = str(raw.get("updated_at") or "")
    items = raw.get("items") if isinstance(raw.get("items"), dict) else {}
    data["items"] = {str(k): _normalize_item(str(k), v) for k, v in items.items()}
    for k, v in raw.items():
        if k not in data:
            data[k] = v
    return data


def save_golden(memory: MemoryPaths, data: dict) -> str:
    """Atomic save with .bak. Only golden_examples.json is written; the clips folder is never touched."""
    memory.ensure()
    out = dict(data)
    out["schema"] = GOLDEN_SCHEMA
    out["updated_at"] = now_iso()
    data["updated_at"] = out["updated_at"]
    return atomic_write_json(memory.golden_examples, out, backup=True)


def scan_golden(memory: MemoryPaths, data: Optional[dict] = None) -> list:
    """Video files in memory/golden_clips (subfolders included, hidden files skipped).
    Keys come from a (size, mtime) cache in data["index"], so unchanged files are not
    hashed again. state: new | done | failed | excluded | duplicate."""
    folder = memory.golden_dir
    if data is None:
        data = load_golden(memory)
    index = data.setdefault("index", {})
    items = data.get("items") or {}
    found = []
    if not os.path.isdir(folder):
        return found
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for name in sorted(files):
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in GOLDEN_EXTS:
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder)
            try:
                st = os.stat(path)
            except OSError:
                continue
            if st.st_size < 1024:
                continue
            cached = index.get(rel) if isinstance(index.get(rel), dict) else None
            if cached and cached.get("size") == st.st_size and cached.get("mtime_ns") == st.st_mtime_ns and cached.get("key"):
                key = str(cached["key"])
            else:
                try:
                    key = golden_key(path)
                except OSError as exc:
                    log(f"⚠️ تعذر قراءة {rel}: {short_error(exc)}")
                    continue
                index[rel] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "key": key}
            found.append({"path": path, "name": rel, "key": key, "size": st.st_size, "state": "new"})
    for rel in list(index.keys()):
        if not os.path.isfile(os.path.join(folder, rel)):
            index.pop(rel, None)
    seen: set = set()
    for f in found:
        item = items.get(f["key"])
        if f["key"] in seen:
            f["state"] = "duplicate"
        elif item and item.get("excluded"):
            f["state"] = "excluded"
        elif item and item.get("status") == "done":
            f["state"] = "done"
        elif item and item.get("status") == "failed":
            f["state"] = "failed"
        seen.add(f["key"])
    return found


def _sync_items(data: dict, files: Sequence[dict]) -> None:
    """Create items for new files, refresh names after renames, flag missing files.
    Items are never removed."""
    items = data.setdefault("items", {})
    present = {}
    for f in files:
        if f["state"] == "duplicate":
            continue
        present[f["key"]] = f
    for key, f in present.items():
        item = items.get(key)
        if item is None:
            item = _normalize_item(key, {"key": key, "file": f["name"], "size": f["size"], "added_at": now_iso()})
            items[key] = item
        item["file"] = f["name"]
        item["size"] = f["size"]
        item["file_missing"] = False
    for key, item in items.items():
        if key not in present:
            item["file_missing"] = True


# ============================================================================
# Prompts (Arabic)
# ============================================================================

ANALYSIS_SYSTEM = """أنت محلل محتوى قصير محترف (تيك توك وريلز) ومتخصص في لقطات البث المباشر.
بتشوف مقطع جاهز عمله ممنتج محترف لستريمر سعودية (بنت) تلعب ألعاب مثل روبلوكس. المقطع مقصوص من بث تويتش طويل.
مهمتك تفهم ليش الممنتج اختار هذي اللحظة الخام من البث بالذات، مو تقييم المونتاج.

قواعد مهمة:
1. تجاهل تماماً كل شي أضافه الممنتج في CapCut: الزووم، الميمز، الستيكرات، المؤثرات الصوتية، الموسيقى، الكتابة والترجمة على الشاشة، الإيموجي، التسريع والتبطيء. اكتب اللي تجاهلته في postprod_ignored.
2. ركز على اللحظة الخام: وش صار في اللعبة بصرياً، وش قالت الستريمر وبأي طاقة صوت، ووش كانت ردة فعلها.
3. الجمب كت (jump cut) تعرفه من قفزة مفاجئة في الصورة أو الصوت داخل نفس المشهد. حدد مكانه ووش انشال (سكوت، كلام زايد، انتظار).
4. الأوقات كلها بصيغة MM:SS من بداية المقطع.
5. اكتب بالعربي، جمل قصيرة وواضحة، بدون مبالغة."""

ANALYSIS_FIELDS = """املأ الحقول كذا:
• game: اسم اللعبة أو نوعها
• moment_type: واحد من {moments}
• visual_story: وش صار في الشاشة خطوة بخطوة (بدون المونتاج)
• hook_at_zero: ليش المقطع يبدأ بالضبط من الثانية 0 (وش اللي يمسك المشاهد)
• hook_latency_s: كم ثانية من البداية لين أول حركة مهمة أو أول كلمة (رقم)
• first_words: أول كلام قالته الستريمر حرفياً
• key_lines: أهم جمل قالتها (حرفياً، 4 بالكثير)
• vocal_energy: طاقة صوتها (هادي، يتصاعد، انفجار) مع وصف قصير
• reaction: نوع ردة فعلها (ضحك هستيري، صراخ، صدمة، اعتذار...)
• jump_cuts: قائمة {{"at": "MM:SS", "what": "وش انشال"}}
• ending: وش اللي قفل المقطع
• ending_type: واحد من {endings}
• raw_duration_guess_s: كم ثانية تتوقع كانت اللحظة الخام في البث قبل القص (رقم)
• pacing: إيقاع المقطع باختصار
• why_it_works: ليش المقطع ناجح في جملة أو جملتين
• selection_rules: من 2 إلى 5 قواعد عامة لاختيار لقطات مثل هذي من بث طويل
• postprod_ignored: عناصر المونتاج اللي شفتها وتجاهلتها"""

DISTILL_SYSTEM = """أنت تستخلص "الحمض الذهبي" لذوق ممنتج محترف من تحليلات لقطاته الجاهزة.
المطلوب قواعد عملية تساعد كشاف ذكي وناقد يختارون لقطات خام جديدة من بث تويتش طويل لنفس الستريمر.
اكتب بالعربي، كل قاعدة جملة قصيرة وقابلة للتطبيق، بدون كلام عام أو مبالغة.
لا تخترع شي مو موجود في التحليلات."""


def _analysis_prompt(index: int, total: int, name: str, seconds: float) -> str:
    return (
        f"حلل المقطع الذهبي رقم {index} من {total} (اسم الملف: {name}، مدته {seconds:.0f} ثانية).\n\n"
        + ANALYSIS_FIELDS.format(moments="، ".join(MOMENT_TYPES), endings="، ".join(ENDING_TYPES))
    )


def _analysis_digest(item: dict, limit: int = DISTILL_ITEM_CHARS) -> str:
    a = item.get("analysis") or {}
    cuts = "، ".join(f"{c.get('at', '')} {c.get('what', '')}".strip() for c in (a.get("jump_cuts") or [])[:6])
    parts = [
        f"🎬 {item.get('file') or item.get('key', '')[:10]} ({item.get('duration_s', 0):.0f}ث، {a.get('moment_type', '')}، اللعبة: {a.get('game', '')})",
        f"   الشاشة: {a.get('visual_story', '')}",
        f"   الهوك: {a.get('hook_at_zero', '')} (بعد {float(a.get('hook_latency_s') or 0):.1f}ث)",
        f"   الكلام: {' | '.join(a.get('key_lines') or [])}",
        f"   الطاقة: {a.get('vocal_energy', '')} | ردة الفعل: {a.get('reaction', '')}",
        f"   القص: {cuts or 'بدون قص واضح'} | الإيقاع: {a.get('pacing', '')}",
        f"   القفلة ({a.get('ending_type', '')}): {a.get('ending', '')}",
        f"   ليش ناجح: {a.get('why_it_works', '')}",
        "   قواعد: " + " | ".join(a.get("selection_rules") or []),
    ]
    text = "\n".join(p for p in parts if p.strip())
    return text[:limit]


def _distill_prompt(items: Sequence[dict], stats: dict) -> str:
    lines = [f"هذي تحليلات {len(items)} لقطة ذهبية عملها الممنتج بنفسه:", ""]
    lines.extend(_analysis_digest(it) for it in items)
    lines.append("")
    lines.append("أرقام محسوبة من المقاطع: " + _stats_line(stats))
    lines.append("")
    lines.append("استخلص:")
    lines.append("• principles: من 5 إلى 8 مبادئ للذوق")
    lines.append("• archetypes: من 3 إلى 6 أنماط لقطات (name، signature مثال يميزه من التحليلات، how_to_spot كيف نلقطه في البث من الكلام والصوت والشاشة)")
    lines.append("• hook_rules: كيف يختار ثانية البداية")
    lines.append("• cut_rules: وش يشيل بالقص ووش يخلي")
    lines.append("• ending_rules: وين يقفل اللقطة")
    lines.append("• avoid: وش يتجنب")
    return "\n".join(lines)


# ============================================================================
# Stats (local math, no model)
# ============================================================================

def _usable_items(golden: dict) -> list:
    return [it for it in (golden.get("items") or {}).values()
            if isinstance(it, dict) and it.get("status") == "done" and it.get("analysis") and not it.get("excluded")]


def golden_stats(golden: dict) -> dict:
    """Numbers from the analyzed, non excluded golden clips:
    n, duration_s [p25, p50, p75] of the finished edits, median hook_latency_s,
    ending_types and moment_types counts, mean cuts_per_min."""
    items = _usable_items(golden)
    if not items:
        return {"n": 0, "duration_s": [], "hook_latency_s": None, "ending_types": {},
                "cuts_per_min": None, "moment_types": {}}
    durations = [float(it.get("duration_s") or 0.0) for it in items if float(it.get("duration_s") or 0.0) > 0]
    hooks = [float((it["analysis"].get("hook_latency_s") or 0.0)) for it in items]
    endings: dict = {}
    moments: dict = {}
    cpm = []
    for it in items:
        a = it["analysis"]
        endings[a.get("ending_type", "ردة_فعل")] = endings.get(a.get("ending_type", "ردة_فعل"), 0) + 1
        moments[a.get("moment_type", "ضحك_هستيري")] = moments.get(a.get("moment_type", "ضحك_هستيري"), 0) + 1
        dur = float(it.get("duration_s") or 0.0)
        if dur > 0:
            cpm.append(len(a.get("jump_cuts") or []) / (dur / 60.0))
    dur_q = [round(float(np.percentile(durations, q)), 1) for q in (25, 50, 75)] if durations else []
    return {
        "n": len(items),
        "duration_s": dur_q,
        "hook_latency_s": round(float(np.median(hooks)), 2) if hooks else None,
        "ending_types": dict(sorted(endings.items(), key=lambda kv: -kv[1])),
        "cuts_per_min": round(float(np.mean(cpm)), 2) if cpm else None,
        "moment_types": dict(sorted(moments.items(), key=lambda kv: -kv[1])),
    }


def _stats_line(stats: dict) -> str:
    if not stats or not stats.get("n"):
        return "ما فيه أرقام بعد"
    parts = []
    dur = stats.get("duration_s") or []
    if len(dur) == 3:
        parts.append(f"المدة غالباً من {dur[0]:.0f} إلى {dur[2]:.0f} ثانية (الوسط {dur[1]:.0f})")
    if stats.get("hook_latency_s") is not None:
        parts.append(f"الهوك بعد {stats['hook_latency_s']:.1f} ثانية تقريباً")
    if stats.get("cuts_per_min") is not None:
        parts.append(f"{stats['cuts_per_min']:.1f} قصة بالدقيقة")
    endings = stats.get("ending_types") or {}
    if endings:
        total = sum(endings.values()) or 1
        parts.append("القفلات: " + "، ".join(f"{ENDING_TYPES_AR.get(k, k)} {v * 100 // total}%" for k, v in endings.items()))
    return " | ".join(parts)


# ============================================================================
# Training
# ============================================================================

def _distill(gw: Any, golden: dict) -> bool:
    items = _usable_items(golden)
    if len(items) < 2:
        return False
    prompt = _distill_prompt(items, golden.get("stats") or golden_stats(golden))
    res = gw.generate("distill", [prompt], schema=GoldenDNA, system_instruction=DISTILL_SYSTEM,
                      normalizer=_unwrap, label="الحمض الذهبي")
    dna = res.data.model_dump() if isinstance(res.data, BaseModel) else GoldenDNA.model_validate(res.data).model_dump()
    golden["dna"] = {"from_keys": sorted(it["key"] for it in items), "model": res.model, "at": now_iso(), "data": dna}
    return True


def _dna_stale(golden: dict) -> bool:
    current = sorted(it["key"] for it in _usable_items(golden))
    dna = golden.get("dna") or {}
    return bool(current) and sorted(dna.get("from_keys") or []) != current


def train_golden(drive_root: str, api_key: str, max_new: Optional[int] = None,
                 redistill: bool = False, gateway: Optional[Any] = None) -> dict:
    """Analyze new golden clips (one Gemini request each), then distill the Golden DNA
    (one request) and recompute the stats locally. Incremental: analyzed files are skipped,
    renamed files keep their analysis, failed files are retried up to 3 times.
    Returns {"analyzed", "failed", "skipped", "distilled", "paused"}."""
    t0 = time.time()
    memory = MemoryPaths(drive_root).ensure()
    golden = load_golden(memory)
    files = scan_golden(memory, golden)
    _sync_items(golden, files)
    summary = {"analyzed": 0, "failed": 0, "skipped": 0, "distilled": False, "paused": None}

    pending = []
    for f in files:
        item = golden["items"].get(f["key"])
        if f["state"] in ("done", "excluded", "duplicate") or item is None:
            summary["skipped"] += 1
            continue
        if item.get("attempts", 0) >= MAX_ATTEMPTS and item.get("status") == "failed":
            summary["skipped"] += 1
            continue
        pending.append(f)
    if max_new is not None:
        extra = pending[max(0, int(max_new)):]
        summary["skipped"] += len(extra)
        pending = pending[:max(0, int(max_new))]

    total_items = len(golden["items"])
    log(f"🌟 اللقطات الذهبية: {len(files)} ملف في المجلد، {total_items} محفوظ في الذاكرة، {len(pending)} جديد للتحليل")
    if not files and not total_items:
        log(f"📂 حط أفضل مقاطعك الجاهزة (8 إلى 15 مقطع) في: {memory.golden_dir}")
        save_golden(memory, golden)
        return summary

    gw_box: list = [gateway]

    def get_gw() -> Any:
        if gw_box[0] is None:
            gw_box[0] = GeminiGateway(api_key, memory.gemini_state)
        return gw_box[0]

    work = ensure_dir(os.path.join(TMP_ROOT, "golden"))
    for i, f in enumerate(pending, start=1):
        item = golden["items"][f["key"]]
        out = os.path.join(work, f"{f['key'][:16]}.mp4")
        log(f"🎬 تحليل المقطع الذهبي {i}/{len(pending)}: {f['name']}")
        try:
            try:
                tinfo = transcode_for_gemini(f["path"], out, fps=TRANSCODE_FPS, short_side=TRANSCODE_SHORT_SIDE,
                                             audio_kbps=TRANSCODE_AUDIO_KBPS, max_seconds=TRANSCODE_MAX_SECONDS)
            except Exception as exc:
                item["status"] = "failed"
                item["attempts"] = item.get("attempts", 0) + 1
                item["error"] = f"تحويل الفيديو فشل: {short_error(exc)}"
                summary["failed"] += 1
                log(f"   ❌ {item['error']}")
                save_golden(memory, golden)
                continue
            if not tinfo.get("has_audio"):
                log("   ⚠️ المقطع ما فيه صوت، التحليل بيعتمد على الصورة فقط")
            with open(tinfo["path"], "rb") as fh:
                payload = fh.read()
            contents = [_analysis_prompt(i, len(pending), f["name"], tinfo["seconds"]),
                        video_item(payload, seconds=tinfo["seconds"], fps=TRANSCODE_FPS)]
            try:
                res = get_gw().generate("golden", contents, schema=GoldenAnalysis, system_instruction=ANALYSIS_SYSTEM,
                                  normalizer=_unwrap, label=f"ذهبي {i}/{len(pending)}")
            except QuotaPause as pause:
                summary["paused"] = pause.message_ar
                log(pause.message_ar)
                break
            except ParseFailure as exc:
                item["status"] = "failed"
                item["attempts"] = item.get("attempts", 0) + 1
                item["error"] = f"رد Gemini ما انقرأ: {exc.detail[:120]}"
                summary["failed"] += 1
                log(f"   ❌ {item['error']}")
                save_golden(memory, golden)
                continue
            except GatewayError as exc:
                if "مفتاح" in str(exc):
                    raise
                item["status"] = "failed"
                item["attempts"] = item.get("attempts", 0) + 1
                item["error"] = short_error(exc, 160)
                summary["failed"] += 1
                log(f"   ❌ {item['error']}")
                save_golden(memory, golden)
                continue
            analysis = res.data if isinstance(res.data, GoldenAnalysis) else GoldenAnalysis.model_validate(res.data)
            usage = dict(res.usage or {})
            usage.update({k: v for k, v in (res.usage_by_modality or {}).items() if k in ("video", "audio", "text")})
            item.update({
                "status": "done", "error": "", "model": res.model, "analyzed_at": now_iso(),
                "duration_s": round(float(tinfo["seconds"]), 2),
                "transcode": {"fps": TRANSCODE_FPS, "height": tinfo.get("height"), "width": tinfo.get("width"),
                              "bytes": tinfo.get("bytes"), "truncated": bool(tinfo.get("truncated"))},
                "usage": usage,
                "analysis": analysis.model_dump(),
            })
            item["attempts"] = item.get("attempts", 0) + 1
            summary["analyzed"] += 1
            ignored = "، ".join(analysis.postprod_ignored[:4]) or "ما ذكر أي مونتاج"
            log(f"   ✅ {MOMENT_TYPES_AR.get(analysis.moment_type, analysis.moment_type)} | "
                f"القفلة: {ENDING_TYPES_AR.get(analysis.ending_type, analysis.ending_type)} | تجاهل: {ignored}")
            golden["stats"] = golden_stats(golden)
            save_golden(memory, golden)
        finally:
            if os.path.exists(out):
                try:
                    os.remove(out)
                except OSError:
                    pass

    golden["stats"] = golden_stats(golden)
    save_golden(memory, golden)

    if summary["paused"] is None and (redistill or summary["analyzed"] > 0 or _dna_stale(golden)):
        usable = len(_usable_items(golden))
        if usable >= 2:
            log(f"🧬 استخلاص الحمض الذهبي من {usable} لقطة...")
            try:
                summary["distilled"] = _distill(get_gw(), golden)
                save_golden(memory, golden)
                if summary["distilled"]:
                    log("   ✅ الحمض الذهبي جاهز ومحفوظ")
            except QuotaPause as pause:
                summary["paused"] = pause.message_ar
                log(pause.message_ar)
            except (ParseFailure, GatewayError) as exc:
                if isinstance(exc, GatewayError) and "مفتاح" in str(exc):
                    raise
                log(f"   ⚠️ الاستخلاص ما نجح هالمرة ({short_error(exc, 100)})، الحمض القديم باقي ونعيد المحاولة بالتشغيل الجاي")
        else:
            log("ℹ️ نحتاج لقطتين ذهبيتين محللة على الأقل عشان نستخلص الحمض الذهبي")

    flush = getattr(gw_box[0], "flush", None) if gw_box[0] is not None else None
    if callable(flush):
        flush()
    try:
        shutil.rmtree(work, ignore_errors=True)
    except Exception:
        pass
    log(f"🏁 خلصنا: حللنا {summary['analyzed']}، فشل {summary['failed']}، تخطينا {summary['skipped']}"
        f"{'، والحمض الذهبي تحدث' if summary['distilled'] else ''} (خلال {time.time() - t0:.0f} ثانية)")
    return summary


# ============================================================================
# Context for Scout and Critic
# ============================================================================

def _similar_items(golden: dict, category: Optional[str], moment_type: Optional[str], limit: int) -> list:
    items = _usable_items(golden)
    wanted = set()
    if moment_type:
        wanted.add(_norm_choice(moment_type, MOMENT_TYPES, ""))
    if category:
        wanted.update(CATEGORY_TO_MOMENTS.get(str(category).strip(), ()))
    wanted.discard("")

    def score(it: dict) -> tuple:
        mt = it["analysis"].get("moment_type")
        exact = 2 if moment_type and mt == _norm_choice(moment_type, MOMENT_TYPES, "") else 0
        near = 1 if mt in wanted else 0
        return (-(exact + near), -len(it["analysis"].get("why_it_works") or ""), str(it.get("analyzed_at") or ""))

    return sorted(items, key=score)[:limit]


def _example_block(it: dict) -> str:
    a = it["analysis"]
    lines = [
        f"⭐ «{it.get('file') or 'مقطع'}» ({MOMENT_TYPES_AR.get(a.get('moment_type'), a.get('moment_type'))}، {it.get('duration_s', 0):.0f}ث)",
        f"   الشاشة: {_text(a.get('visual_story'), 260)}",
        f"   الهوك: {_text(a.get('hook_at_zero'), 160)}",
        f"   القفلة: {_text(a.get('ending'), 160)}",
        f"   ليش ناجح: {_text(a.get('why_it_works'), 200)}",
    ]
    return "\n".join(lines)


def _rules_from_items(items: Sequence[dict], limit: int) -> list:
    counts: dict = {}
    for it in items:
        for rule in it["analysis"].get("selection_rules") or []:
            counts[rule] = counts.get(rule, 0) + 1
    return [r for r, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:limit]


def _cap(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    return (cut[:nl] if nl > max_chars * 0.6 else cut).rstrip() + "\n…"


def golden_context(golden: dict, route: str, category: Optional[str] = None,
                   moment_type: Optional[str] = None, max_chars: int = 4000) -> str:
    """Arabic Golden DNA block for prompts.
    route "scout": compact (about 900 tokens): principles, archetypes with how to spot them, stats.
    route "critic" (and any other route): full DNA plus the 2 golden examples most similar to
    the candidate's category or moment type."""
    if not isinstance(golden, dict):
        return ""
    items = _usable_items(golden)
    dna = ((golden.get("dna") or {}).get("data")) or {}
    stats = golden.get("stats") or golden_stats(golden)
    if not items and not dna:
        return ""
    n = len(items)
    lines = [f"🌟 الحمض الذهبي (مستخلص من {n} لقطة جاهزة عملها الممنتج بنفسه، طبقه بدون ما تنسخه حرفياً):"]
    principles = dna.get("principles") or _rules_from_items(items, 6)
    archetypes = dna.get("archetypes") or []

    if route == "scout":
        cap = min(int(max_chars), SCOUT_CHAR_CAP)
        if principles:
            lines.append("مبادئ:")
            lines.extend(f"   • {p}" for p in principles[:6])
        if archetypes:
            lines.append("أنماط لقطاته المفضلة وكيف تلقطها في البث:")
            for a in archetypes[:5]:
                spot = a.get("how_to_spot") or a.get("signature") or ""
                lines.append(f"   • {a.get('name', '')}: {spot}")
        lines.append("📊 " + _stats_line(stats))
        return _cap("\n".join(lines), cap)

    cap = int(max_chars)
    if principles:
        lines.append("مبادئ:")
        lines.extend(f"   • {p}" for p in principles[:8])
    if archetypes:
        lines.append("الأنماط:")
        for a in archetypes[:6]:
            lines.append(f"   • {a.get('name', '')}: {a.get('signature', '')} (تلقطها من: {a.get('how_to_spot', '')})")
    for key, title in (("hook_rules", "البداية"), ("cut_rules", "القص"), ("ending_rules", "القفلة"), ("avoid", "تجنب")):
        rules = dna.get(key) or []
        if rules:
            lines.append(f"{title}: " + " | ".join(rules[:5]))
    lines.append("📊 " + _stats_line(stats))
    examples = _similar_items(golden, category, moment_type, 2)
    if examples:
        lines.append("أقرب لقطات ذهبية لهذا النوع:")
        lines.extend(_example_block(it) for it in examples)
    return _cap("\n".join(lines), cap)


# ============================================================================
# Studio helpers
# ============================================================================

def set_excluded(memory: MemoryPaths, key: str, excluded: bool) -> dict:
    """Include or exclude one analyzed clip from the DNA, stats and examples. The analysis
    itself is kept. The DNA is marked stale and refreshed on the next training run.
    Returns the updated golden data."""
    golden = load_golden(memory)
    item = golden["items"].get(str(key))
    if item is None:
        raise KeyError(f"المقطع {key} غير موجود في الذاكرة الذهبية")
    item["excluded"] = bool(excluded)
    golden["stats"] = golden_stats(golden)
    save_golden(memory, golden)
    state = "استبعدنا" if excluded else "رجعنا"
    log(f"🌟 {state} المقطع {item.get('file') or key[:10]} {'من' if excluded else 'إلى'} الحمض الذهبي (يتحدث بالتدريب الجاي)")
    return golden


def golden_summary_ar(golden: dict) -> str:
    """Arabic summary for the Studio Golden tab."""
    items = list((golden.get("items") or {}).values()) if isinstance(golden, dict) else []
    if not items:
        return "🌟 ما فيه لقطات ذهبية بعد. حط أفضل 8 إلى 15 مقطع جاهز في مجلد memory/golden_clips وشغّل خلية التدريب."
    done = [it for it in items if it.get("status") == "done"]
    failed = [it for it in items if it.get("status") == "failed"]
    missing = [it for it in items if it.get("file_missing")]
    excluded = [it for it in items if it.get("excluded")]
    new = [it for it in items if it.get("status") == "new" and not it.get("file_missing")]
    lines = [f"🌟 اللقطات الذهبية: {len(done)} محللة من {len(items)}"
             + (f"، {len(new)} تنتظر التحليل" if new else "")
             + (f"، {len(failed)} فشلت" if failed else "")
             + (f"، {len(excluded)} مستبعدة" if excluded else "")
             + (f"، {len(missing)} ملفها محذوف (التحليل محفوظ)" if missing else "")]
    stats = golden.get("stats") or golden_stats(golden)
    if stats.get("n"):
        lines.append("📊 " + _stats_line(stats))
        moments = stats.get("moment_types") or {}
        if moments:
            lines.append("🎭 الأنواع: " + "، ".join(f"{MOMENT_TYPES_AR.get(k, k)} ×{v}" for k, v in moments.items()))
    dna = ((golden.get("dna") or {}).get("data")) or {}
    if dna.get("principles"):
        lines.append("🧬 أهم مبادئ الحمض الذهبي:")
        lines.extend(f"   • {p}" for p in dna["principles"][:4])
        if _dna_stale(golden):
            lines.append("   ℹ️ فيه تغييرات بعد آخر استخلاص، شغّل التدريب عشان يتحدث")
    elif done:
        lines.append("🧬 الحمض الذهبي ما انستخلص بعد (يحتاج لقطتين محللة على الأقل)")
    lines.append("")
    for it in sorted(items, key=lambda x: str(x.get("file") or "")):
        a = it.get("analysis") or {}
        flags = []
        if it.get("excluded"):
            flags.append("مستبعد")
        if it.get("file_missing"):
            flags.append("الملف محذوف")
        if it.get("status") == "failed":
            flags.append(f"فشل: {it.get('error', '')[:60]}")
        if it.get("status") == "new":
            flags.append("ينتظر")
        flag_txt = f" [{'، '.join(flags)}]" if flags else ""
        if a:
            ignored = "، ".join((a.get("postprod_ignored") or [])[:3]) or "ما ذكر"
            lines.append(f"• {it.get('file') or it['key'][:10]}{flag_txt}: {MOMENT_TYPES_AR.get(a.get('moment_type'), a.get('moment_type'))}"
                         f" | القفلة {ENDING_TYPES_AR.get(a.get('ending_type'), a.get('ending_type'))} | تجاهل: {ignored}")
        else:
            lines.append(f"• {it.get('file') or it['key'][:10]}{flag_txt}")
    return "\n".join(lines)


# ============================================================================
# Offline self test
# ============================================================================

class _FakeResult:
    def __init__(self, data: Any, model: str):
        self.data = data
        self.model = model
        self.usage = {"input": 9000, "output": 600}
        self.usage_by_modality = {"video": 7000, "audio": 1100, "text": 900}


class _FakeGateway:
    """Stand in for GeminiGateway: answers from canned data, can pause after N calls."""

    def __init__(self, pause_after: Optional[int] = None):
        self.calls: list = []
        self.pause_after = pause_after

    def generate(self, route: str, contents: Any, schema: Any = None, system_instruction: Optional[str] = None,
                 normalizer: Any = None, label: str = "", **_kw: Any) -> _FakeResult:
        self.calls.append(route)
        if self.pause_after is not None and len(self.calls) > self.pause_after:
            raise QuotaPause(route, "quota", None, "⏸️ (اختبار) خلصت الحصة")
        if route == "golden":
            n = len([c for c in self.calls if c == "golden"])
            raw = {"analysis": {
                "game": "روبلوكس", "moment_type": "فشل بصري" if n % 2 else "غدر ومقلب",
                "visual_story": "اشترت أداة دفع وبغت تدف شخص غريب ودفت صديقتها بالغلط",
                "hook_at_zero": "تبدأ لحظة شراء الأداة", "hook_latency_s": "00:01",
                "first_words": "خلني اجرب", "key_lines": ["لا لا سوري", "والله ما قصدت"],
                "vocal_energy": "يتصاعد لين انفجار ضحك", "reaction": "ضحك هستيري واعتذار",
                "jump_cuts": [{"at": 6, "what": "انتظار"}, "00:12"], "ending": "تعيدها مرة ثانية وتنفجر ضحك",
                "ending_type": "ذروة الضحك", "raw_duration_guess_s": 55, "pacing": "سريع",
                "why_it_works": "فشل بصري واضح مع ردة فعل صادقة", "selection_rules": "ابدأ قبل الحركة\nاقفل على الضحكة",
                "postprod_ignored": ["زووم", "ميم", "كتابة"],
            }}
            data = schema.model_validate(normalizer(raw) if normalizer else raw)
            return _FakeResult(data, "fake-golden")
        raw = {"principles": ["الحركة البصرية قبل الكلام", "القفلة على الضحكة"],
               "archetypes": [{"name": "الدفة الغلط", "signature": "دفت صديقتها", "how_to_spot": "ضحك فجأة بعد سكوت + سوري"}],
               "hook_rules": ["ابدأ قبل الحركة بثانية"], "cut_rules": ["شيل الانتظار"],
               "ending_rules": ["اقفل بعد الضحكة"], "avoid": ["سوالف طويلة بدون ذروة"]}
        return _FakeResult(schema.model_validate(raw), "fake-distill")

    def flush(self) -> None:
        return None


def selftest(workdir: Optional[str] = None) -> dict:
    """Offline checks with synthetic clips and a fake gateway (no Gemini quota used):
      1. first run analyzes every clip, distills the DNA, writes stats
      2. second run analyzes nothing (incremental)
      3. a renamed file keeps its analysis (same key)
      4. a deleted file keeps its analysis with file_missing=True
      5. a quota pause mid run saves the finished clips and resumes later
      6. stats match what taste_engine.apply_golden_priors expects, contexts fit their caps
      7. the golden_clips folder is untouched (same files, same bytes)
    Returns a dict of results with an "all_ok" flag."""
    from common import run_command

    root = workdir or tempfile.mkdtemp(prefix="tc_golden_selftest_")
    memory = MemoryPaths(root).ensure()
    results: dict = {}
    log("🧪 فحص ذاتي للقطات الذهبية (مقاطع صناعية وبوابة وهمية)...")

    def mark(name: str, ok: bool, detail: str) -> None:
        results[name] = {"ok": bool(ok), "detail": detail}
        log(f"   {'✅' if ok else '❌'} {name}: {detail}")

    try:
        for i, (size, tone) in enumerate((("720x1280", 440), ("1080x1920", 660), ("720x1280", 880))):
            path = os.path.join(memory.golden_dir, f"edit_{i + 1}.mp4")
            if not os.path.isfile(path):
                res = run_command(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                                   f"testsrc2=s={size}:r=30", "-f", "lavfi", "-i", f"sine=f={tone}:r=48000",
                                   "-t", str(8 + 2 * i), "-c:v", "libx264", "-preset", "ultrafast",
                                   "-pix_fmt", "yuv420p", "-c:a", "aac", path], timeout=300)
                if not res.ok:
                    raise RuntimeError(res.last_error_line())
        before = {n: golden_key(os.path.join(memory.golden_dir, n)) for n in os.listdir(memory.golden_dir)}

        gw = _FakeGateway(pause_after=2)
        s1 = train_golden(root, "selftest_key_0000", gateway=gw)
        g = load_golden(memory)
        mark("pause_resume_saved", s1["paused"] is not None and s1["analyzed"] == 2
             and sum(1 for it in g["items"].values() if it["status"] == "done") == 2,
             f"توقف بعد {s1['analyzed']} مقطع وحفظهم")

        gw2 = _FakeGateway()
        s2 = train_golden(root, "selftest_key_0000", gateway=gw2)
        g = load_golden(memory)
        mark("first_run", s2["analyzed"] == 1 and s2["distilled"] and g["dna"].get("data", {}).get("principles"),
             f"كمل المقطع الباقي واستخلص الحمض ({gw2.calls})")

        gw3 = _FakeGateway()
        s3 = train_golden(root, "selftest_key_0000", gateway=gw3)
        mark("incremental", s3["analyzed"] == 0 and not gw3.calls, f"التشغيل الثاني ما أرسل أي طلب ({len(gw3.calls)})")

        os.rename(os.path.join(memory.golden_dir, "edit_1.mp4"), os.path.join(memory.golden_dir, "renamed.mp4"))
        gw4 = _FakeGateway()
        s4 = train_golden(root, "selftest_key_0000", gateway=gw4)
        g = load_golden(memory)
        renamed = [it for it in g["items"].values() if it["file"] == "renamed.mp4"]
        mark("rename", s4["analyzed"] == 0 and renamed and renamed[0]["status"] == "done",
             "الملف بعد تغيير اسمه احتفظ بتحليله")
        os.rename(os.path.join(memory.golden_dir, "renamed.mp4"), os.path.join(memory.golden_dir, "edit_1.mp4"))

        hidden = os.path.join(root, "hidden_edit_3.mp4")
        shutil.move(os.path.join(memory.golden_dir, "edit_3.mp4"), hidden)
        train_golden(root, "selftest_key_0000", gateway=_FakeGateway())
        g = load_golden(memory)
        missing = [it for it in g["items"].values() if it["file_missing"]]
        mark("missing_kept", len(missing) == 1 and missing[0]["status"] == "done" and len(g["items"]) == 3,
             "الملف المحذوف بقى تحليله في الذاكرة (file_missing)")
        shutil.move(hidden, os.path.join(memory.golden_dir, "edit_3.mp4"))

        stats = golden_stats(g)
        stats_ok = (stats["n"] == 3 and len(stats["duration_s"]) == 3 and stats["hook_latency_s"] == 1.0
                    and stats["cuts_per_min"] and set(stats["ending_types"]) <= set(ENDING_TYPES)
                    and set(stats["moment_types"]) <= set(MOMENT_TYPES))
        scout_ctx = golden_context(g, "scout")
        critic_ctx = golden_context(g, "critic", category="مقلب_وغدر")
        ctx_ok = 0 < len(scout_ctx) <= SCOUT_CHAR_CAP + 2 and 0 < len(critic_ctx) <= 4002 and "⭐" in critic_ctx
        mark("stats_context", stats_ok and ctx_ok,
             f"المدة {stats['duration_s']}، الهوك {stats['hook_latency_s']}ث، {stats['cuts_per_min']} قصة/دقيقة، "
             f"سياق الكشاف {len(scout_ctx)} حرف والناقد {len(critic_ctx)} حرف")

        key = next(iter(g["items"]))
        g2 = set_excluded(memory, key, True)
        mark("exclude", golden_stats(g2)["n"] == 2 and _dna_stale(g2), "الاستبعاد يحدث الأرقام ويطلب استخلاص جديد")
        set_excluded(memory, key, False)

        after = {n: golden_key(os.path.join(memory.golden_dir, n)) for n in os.listdir(memory.golden_dir)}
        mark("folder_untouched", before == after, "مجلد اللقطات الذهبية ما تغير فيه أي ملف")
        summary = golden_summary_ar(load_golden(memory))
        mark("summary", "🌟" in summary and "🧬" in summary, summary.splitlines()[0])
    except Exception as exc:
        mark("golden", False, f"خطأ: {short_error(exc)}")

    all_ok = all(v["ok"] for v in results.values())
    results["all_ok"] = all_ok
    log("🎉 كل فحوصات اللقطات الذهبية ناجحة" if all_ok else "⚠️ فيه فحوصات فشلت في اللقطات الذهبية")
    return results


# === END OF golden_engine.py ===
