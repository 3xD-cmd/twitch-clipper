"""
common.py
Shared helpers for the TwitchClipper pipeline (Stage 2, Stage 3, Stage 4 and the Colab notebook).

Contents
  * Logging in Arabic
  * Atomic, Drive safe file writes (temp file in the same folder, fsync, os.replace)
  * JSON loading with automatic fallback to the last good ".bak" copy
  * Time formatting and tolerant parsing (seconds, HH:MM:SS, Arabic Indic digits)
  * Number coercion for LLM output ("9/10", "٨", "8.5")
  * Arabic text normalization for lexicon matching
  * Safe file names (no emoji, no bidi marks, byte capped)
  * Pacific midnight quota reset expressed in Riyadh time
  * Path layout dataclasses (VodPaths, MemoryPaths) and VOD discovery
  * Manifest v2 and feedback v2 loaders with legacy (v1) conversion
  * Guarded deletion that can never touch TwitchClipper/memory
  * Subprocess helper and module reload helper for Colab runtime switches

This module has no third party dependency except an optional numpy import
used only to serialize numpy values to JSON.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime as _dt
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Optional, Sequence

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore[assignment]


# ============================================================================
# Constants
# ============================================================================

DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/TwitchClipper"
TMP_ROOT = "/tmp/twitch_clipper"

TRANSCRIPT_SCHEMA = 2
MANIFEST_SCHEMA = 2
FEEDBACK_SCHEMA = 2

BACKUP_SUFFIX = ".bak"

VALID_RATINGS = ("pending", "liked", "disliked")

PROJECT_MODULES = (
    "common",
    "gemini_gateway",
    "media_engine",
    "taste_engine",
    "stage2_ingest",
    "stage3_clipper",
    "stage4_studio",
)

RIYADH_TZ = _dt.timezone(_dt.timedelta(hours=3), name="AST")
UTC = _dt.timezone.utc

_AR_WEEKDAYS = ("الاثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت", "الأحد")
_AR_MONTHS = (
    "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
    "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
)


# ============================================================================
# Logging
# ============================================================================

def log(msg: str) -> None:
    """Print one line immediately (Colab buffers output otherwise)."""
    print(msg, flush=True)


def short_error(exc: BaseException, max_len: int = 180) -> str:
    """First line of an exception message, trimmed, for compact Arabic logs."""
    text = str(exc).strip() or exc.__class__.__name__
    first = text.splitlines()[0] if text else exc.__class__.__name__
    return first[:max_len]


def log_exception(context_ar: str, exc: BaseException) -> None:
    log(f"❌ {context_ar}: {short_error(exc)}")


def elapsed_str(t0: float) -> str:
    """Human readable elapsed time since t0 (time.time())."""
    return format_duration_ar(max(0.0, time.time() - t0))


# ============================================================================
# JSON serialization helpers
# ============================================================================

_NUMPY = None
_NUMPY_CHECKED = False


def _numpy():
    global _NUMPY, _NUMPY_CHECKED
    if not _NUMPY_CHECKED:
        _NUMPY_CHECKED = True
        try:
            import numpy as np  # noqa: WPS433
            _NUMPY = np
        except Exception:
            _NUMPY = None
    return _NUMPY


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types, tuples, sets, dates, paths, dataclasses and
    pydantic models into plain JSON types. NaN and infinity become None so the
    output is always strict JSON."""
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.time)):
        return obj.isoformat()
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    np = _numpy()
    if np is not None:
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return val if math.isfinite(val) else None
        if isinstance(obj, np.ndarray):
            return to_jsonable(obj.tolist())
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        try:
            return to_jsonable(dump())
        except Exception:
            pass
    return str(obj)


# ============================================================================
# Atomic file writes (safe on Google Drive FUSE)
# ============================================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _tmp_path_for(path: str) -> str:
    folder = os.path.dirname(os.path.abspath(path))
    base = os.path.basename(path)
    token = f"{os.getpid()}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
    return os.path.join(folder, f".{base}.{token}.tmp")


def _remove_quietly(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _is_valid_json_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            json.load(f)
        return True
    except (OSError, ValueError, UnicodeDecodeError):
        return False


def _backup_existing(path: str, validate_json: bool) -> None:
    """Copy the current file to path.bak before it is replaced. A corrupt JSON file
    is never allowed to overwrite a good backup."""
    if not os.path.isfile(path):
        return
    if validate_json and not _is_valid_json_file(path):
        return
    try:
        shutil.copyfile(path, path + BACKUP_SUFFIX)
    except OSError as exc:
        log(f"⚠️ تعذر حفظ نسخة احتياطية من {os.path.basename(path)}: {short_error(exc)}")


def atomic_write_bytes(path: str, data: bytes, backup: bool = False, _validate_json_backup: bool = False) -> str:
    """Write bytes to a temp file in the same folder, fsync it, then os.replace it over
    the target. Readers see either the old file or the new file, never half a file."""
    path = os.fspath(path)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    tmp = _tmp_path_for(path)
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        if backup:
            _backup_existing(path, validate_json=_validate_json_backup)
        os.replace(tmp, path)
    except BaseException:
        _remove_quietly(tmp)
        raise
    return path


def atomic_write_text(path: str, text: str, encoding: str = "utf-8", backup: bool = False) -> str:
    """Atomic text write. Use encoding="utf-8-sig" to write a BOM."""
    return atomic_write_bytes(path, str(text).encode(encoding), backup=backup)


def atomic_write_json(path: str, data: Any, indent: Optional[int] = 2, backup: bool = True) -> str:
    """Atomic strict JSON write (UTF-8, Arabic kept readable, NaN converted to null).
    By default the previous good version is kept as path.bak."""
    if indent is None:
        payload = json.dumps(to_jsonable(data), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    else:
        payload = json.dumps(to_jsonable(data), ensure_ascii=False, indent=indent, allow_nan=False)
    return atomic_write_bytes(path, (payload + "\n").encode("utf-8"), backup=backup, _validate_json_backup=True)


def load_json_safe(
    path: str,
    default: Any = None,
    expected_types: Optional[tuple] = None,
    quiet: bool = False,
) -> Any:
    """Load JSON. If the file is missing, corrupt, or of an unexpected type, try
    path.bak, then return a deep copy of default. Never raises for bad content."""
    path = os.fspath(path)
    name = os.path.basename(path)
    for candidate, is_backup in ((path, False), (path + BACKUP_SUFFIX, True)):
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            if not quiet:
                which = "النسخة الاحتياطية من" if is_backup else "الملف"
                log(f"⚠️ {which} {name} تالف أو ما ينقرأ ({short_error(exc, 90)})")
            continue
        if expected_types and not isinstance(data, expected_types):
            if not quiet:
                log(f"⚠️ محتوى {name} بصيغة غير متوقعة، تم تجاهله")
            continue
        if is_backup and not quiet:
            log(f"♻️ تم استرجاع آخر نسخة سليمة من {name}")
        return data
    return copy.deepcopy(default)


def read_text_safe(path: str, default: str = "") -> str:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return default


def safe_copy_file(src: str, dst: str) -> str:
    """Copy src to dst atomically: copy to a temp name next to dst, then rename.
    Used for /tmp to Drive transfers so Drive never holds a partial file."""
    ensure_dir(os.path.dirname(os.path.abspath(dst)))
    tmp = _tmp_path_for(dst)
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        _remove_quietly(tmp)
        raise
    return dst


def file_sha1(path: str, max_bytes: Optional[int] = None, chunk: int = 1 << 20) -> str:
    """SHA1 of a file (optionally only the first max_bytes)."""
    h = hashlib.sha1()
    read_total = 0
    with open(path, "rb") as f:
        while True:
            want = chunk if max_bytes is None else min(chunk, max_bytes - read_total)
            if want <= 0:
                break
            block = f.read(want)
            if not block:
                break
            h.update(block)
            read_total += len(block)
    return h.hexdigest()


def text_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:length]


def file_fingerprint(path: str) -> str:
    """Cheap identity of a file: size and modification time in nanoseconds."""
    try:
        st = os.stat(path)
        return f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return "missing"


# ============================================================================
# Sizes, deletion guard
# ============================================================================

def dir_size_bytes(path: str) -> int:
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path, onerror=lambda _e: None):
        for fname in files:
            try:
                total += os.path.getsize(os.path.join(root, fname))
            except OSError:
                continue
    return total


def human_size(num_bytes: float) -> str:
    size = float(max(0.0, num_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def safe_delete(path: str, drive_root: str) -> int:
    """Delete a file or folder inside drive_root/vods/<vod_id>/ only.

    Guards:
      * the real path must be inside drive_root/vods, at least one level below a VOD folder
      * a whole VOD folder or the vods folder itself is refused
      * anything inside drive_root/memory is refused
    Returns the number of bytes freed (0 if the path did not exist)."""
    if not path:
        return 0
    real = os.path.realpath(path)
    vods_root = os.path.realpath(os.path.join(drive_root, "vods"))
    memory_root = os.path.realpath(os.path.join(drive_root, "memory"))

    if real == memory_root or real.startswith(memory_root + os.sep):
        raise PermissionError("❌ ممنوع حذف أي شي داخل مجلد الذاكرة memory")
    if not real.startswith(vods_root + os.sep):
        raise PermissionError(f"❌ رفض الحذف: المسار خارج مجلد البثوث ({path})")
    rel_parts = os.path.relpath(real, vods_root).split(os.sep)
    if len(rel_parts) < 2 or any(p in ("", ".", "..") for p in rel_parts):
        raise PermissionError(f"❌ رفض الحذف: ما نحذف مجلد البث كامل، فقط ملفاته ({path})")

    if not os.path.lexists(real):
        return 0
    freed = dir_size_bytes(real)
    if os.path.isdir(real) and not os.path.islink(real):
        shutil.rmtree(real)
    else:
        os.remove(real)
    return freed


# ============================================================================
# Digits, numbers, time parsing and formatting
# ============================================================================

_DIGIT_TABLE = str.maketrans({
    **{chr(0x0660 + i): str(i) for i in range(10)},  # Arabic Indic
    **{chr(0x06F0 + i): str(i) for i in range(10)},  # Extended Arabic Indic (Persian)
    "٫": ".",  # Arabic decimal separator
    "٬": "",   # Arabic thousands separator
})


def normalize_digits(text: str) -> str:
    """Convert Arabic Indic and Persian digits to ASCII digits."""
    return str(text).translate(_DIGIT_TABLE)


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


_NUM_RE = re.compile(r"[+\-]?\d+(?:\.\d+)?")
_FRACTION_RE = re.compile(r"([+\-]?\d+(?:\.\d+)?)\s*(?:/|من)\s*(\d+(?:\.\d+)?)")


def coerce_number(value: Any, default: Optional[float] = None, fraction_scale: Optional[float] = 10.0) -> Optional[float]:
    """Tolerant number parsing for LLM output.
    8 -> 8.0 | "8.5" -> 8.5 | "٨" -> 8.0 | "9/10" -> 9.0 | "4 من 5" -> 8.0 (scaled to 10)
    Returns default when nothing usable is found."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) else default
    text = normalize_digits(str(value)).strip().replace(",", ".")
    if not text:
        return default
    frac = _FRACTION_RE.search(text)
    if frac:
        num, den = float(frac.group(1)), float(frac.group(2))
        if den > 0:
            return (num / den) * fraction_scale if fraction_scale else num
        return default
    match = _NUM_RE.search(text)
    if not match:
        return default
    try:
        val = float(match.group(0))
    except ValueError:
        return default
    return val if math.isfinite(val) else default


def coerce_int(value: Any, default: int = 0, lo: Optional[int] = None, hi: Optional[int] = None) -> int:
    num = coerce_number(value, None)
    if num is None:
        return default
    out = int(round(num))
    if lo is not None:
        out = max(lo, out)
    if hi is not None:
        out = min(hi, out)
    return out


_TIME_RE = re.compile(r"^([0-9:.]+)\s*(s|sec|secs|second|seconds|ث|ثانية|ثواني)?$")


def parse_time_to_sec(value: Any) -> Optional[float]:
    """Parse seconds from 123, 123.4, "123.4", "75s", "1:23", "01:02:03.5",
    "01:02:03,500" (SRT style), or Arabic Indic digits. Returns None if not parseable."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        val = float(value)
        return val if math.isfinite(val) and val >= 0 else None
    text = normalize_digits(str(value)).strip().lower()
    if not text:
        return None
    if re.match(r"^\d{1,2}:\d{2}:\d{2},\d{1,3}$", text):
        text = text.replace(",", ".")
    match = _TIME_RE.match(text)
    if not match:
        return None
    core = match.group(1)
    try:
        if ":" in core:
            parts = core.split(":")
            if len(parts) > 3 or any(p == "" for p in parts):
                return None
            nums = [float(p) for p in parts]
            if any(n < 0 for n in nums):
                return None
            total = 0.0
            for n in nums:
                total = total * 60.0 + n
            return total
        val = float(core)
        return val if math.isfinite(val) and val >= 0 else None
    except ValueError:
        return None


def _safe_seconds(seconds: Any) -> float:
    try:
        val = float(seconds)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(val):
        return 0.0
    return max(0.0, val)


def sec_to_hms(seconds: Any) -> str:
    """HH:MM:SS.ss (centiseconds), rounding carried correctly (59.999 -> 01:00.00)."""
    total_cs = int(round(_safe_seconds(seconds) * 100))
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:02d}:{m:02d}:{s:02d}.{cs:02d}"


def sec_to_clock(seconds: Any) -> str:
    """HH:MM:SS (whole seconds, floored). Used for orientation stamps in prompts."""
    total = int(_safe_seconds(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def sec_to_srt_time(seconds: Any) -> str:
    """HH:MM:SS,mmm for SRT files."""
    total_ms = int(round(_safe_seconds(seconds) * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_duration_ar(seconds: Any) -> str:
    total = int(round(_safe_seconds(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} ساعة و {m} دقيقة" if m else f"{h} ساعة"
    if m:
        return f"{m} دقيقة و {s} ثانية" if s else f"{m} دقيقة"
    return f"{s} ثانية"


# ============================================================================
# Dates: Riyadh display, Pacific quota day
# ============================================================================

def now_utc() -> _dt.datetime:
    return _dt.datetime.now(UTC)


def now_iso() -> str:
    """Current Riyadh time as ISO 8601 with offset, seconds precision."""
    return _dt.datetime.now(RIYADH_TZ).isoformat(timespec="seconds")


def new_run_id() -> str:
    """Unique, sortable Stage 3 run id such as r20261122T083015 (UTC)."""
    return "r" + now_utc().strftime("%Y%m%dT%H%M%S")


def _as_aware_utc(moment: Optional[_dt.datetime]) -> _dt.datetime:
    if moment is None:
        return now_utc()
    if moment.tzinfo is None:
        return moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def _pacific_zone():
    if ZoneInfo is None:
        return None
    try:
        return ZoneInfo("America/Los_Angeles")
    except Exception:
        return None


def _nth_sunday(year: int, month: int, n: int) -> _dt.date:
    first = _dt.date(year, month, 1)
    offset = (6 - first.weekday()) % 7
    return first + _dt.timedelta(days=offset + 7 * (n - 1))


def _us_pacific_offset_hours(moment_utc: _dt.datetime) -> int:
    """Fallback when tzdata is missing. US DST: second Sunday of March 02:00 PST
    (10:00 UTC) until first Sunday of November 02:00 PDT (09:00 UTC)."""
    year = moment_utc.year
    dst_start = _dt.datetime.combine(_nth_sunday(year, 3, 2), _dt.time(10, 0), tzinfo=UTC)
    dst_end = _dt.datetime.combine(_nth_sunday(year, 11, 1), _dt.time(9, 0), tzinfo=UTC)
    return -7 if dst_start <= moment_utc < dst_end else -8


def now_pacific(moment: Optional[_dt.datetime] = None) -> _dt.datetime:
    utc_moment = _as_aware_utc(moment)
    zone = _pacific_zone()
    if zone is not None:
        return utc_moment.astimezone(zone)
    offset = _us_pacific_offset_hours(utc_moment)
    return utc_moment.astimezone(_dt.timezone(_dt.timedelta(hours=offset)))


def pacific_date_str(moment: Optional[_dt.datetime] = None) -> str:
    """Quota day key (Gemini RPD resets at midnight Pacific)."""
    return now_pacific(moment).date().isoformat()


def next_pacific_midnight_utc(moment: Optional[_dt.datetime] = None) -> _dt.datetime:
    pac_now = now_pacific(moment)
    next_day = pac_now.date() + _dt.timedelta(days=1)
    zone = _pacific_zone()
    if zone is not None:
        return _dt.datetime.combine(next_day, _dt.time(0, 0), tzinfo=zone).astimezone(UTC)
    probe = _dt.datetime.combine(next_day, _dt.time(8, 0), tzinfo=UTC)
    offset = _us_pacific_offset_hours(probe)
    local_tz = _dt.timezone(_dt.timedelta(hours=offset))
    return _dt.datetime.combine(next_day, _dt.time(0, 0), tzinfo=local_tz).astimezone(UTC)


def seconds_until_pacific_midnight(moment: Optional[_dt.datetime] = None) -> float:
    return max(0.0, (next_pacific_midnight_utc(moment) - _as_aware_utc(moment)).total_seconds())


def fmt_riyadh_ar(moment: _dt.datetime) -> str:
    """Example: 11:00 صباحاً (الأحد 23 نوفمبر)"""
    local = _as_aware_utc(moment).astimezone(RIYADH_TZ)
    hour12 = local.hour % 12 or 12
    period = "صباحاً" if local.hour < 12 else "مساءً"
    weekday = _AR_WEEKDAYS[local.weekday()]
    month = _AR_MONTHS[local.month - 1]
    return f"{hour12}:{local.minute:02d} {period} ({weekday} {local.day} {month})"


def pacific_reset_riyadh(moment: Optional[_dt.datetime] = None) -> tuple:
    """Next Gemini daily quota reset as (datetime in Riyadh time, Arabic text)."""
    reset_utc = next_pacific_midnight_utc(moment)
    remaining = (reset_utc - _as_aware_utc(moment)).total_seconds()
    text = f"الساعة {fmt_riyadh_ar(reset_utc)} بتوقيت الرياض، بعد {format_duration_ar(remaining)} تقريباً"
    return reset_utc.astimezone(RIYADH_TZ), text


# ============================================================================
# Arabic text normalization
# ============================================================================

_AR_DIACRITICS_RE = re.compile("[ؐ-ًؚ-ٰٟۖ-ۭ]")
_ALEF_RE = re.compile("[آأإٱ]")  # آ أ إ ٱ
_REPEAT_RE = re.compile(r"(.)\1{2,}")
_SPACE_RE = re.compile(r"\s+")


def normalize_arabic(text: str, collapse_repeats: bool = True, keep_ta_marbuta: bool = False) -> str:
    """Normalize Arabic for matching: presentation forms to base letters, no tatweel,
    no harakat, unified alef / ya / hamza seats, optional ة to ه, elongations
    collapsed to two letters (لااااا -> لاا, هههههه -> هه), lower case Latin."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", str(text))
    out = out.replace("ـ", "")
    out = _AR_DIACRITICS_RE.sub("", out)
    out = _ALEF_RE.sub("ا", out)
    out = out.replace("ى", "ي")  # ى -> ي
    out = out.replace("ؤ", "و")  # ؤ -> و
    out = out.replace("ئ", "ي")  # ئ -> ي
    if not keep_ta_marbuta:
        out = out.replace("ة", "ه")  # ة -> ه
    if collapse_repeats:
        out = _REPEAT_RE.sub(r"\1\1", out)
    out = _SPACE_RE.sub(" ", out).strip().lower()
    return out


def arabic_char_count(text: str) -> int:
    """Visible letters and digits (no spaces, punctuation or harakat). Used for
    characters per second pacing in the SRT engine."""
    count = 0
    for ch in str(text):
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat == "Nd":
            count += 1
    return count


# ============================================================================
# File names, VOD ids, clip ids
# ============================================================================

_ILLEGAL_FS_RE = re.compile(r'[\\/*?:"<>|\x00-\x1f\x7f]+')
_DROP_CATEGORIES = {"So", "Sk", "Cs", "Co", "Cn", "Cc", "Cf", "Mn", "Me"}


def sanitize_filename(name: Any, max_len: int = 40, fallback: str = "clip", max_bytes: int = 150) -> str:
    """Safe file name: no path characters, no emoji or symbols, no bidi or zero width
    marks, no harakat, spaces become underscores, capped in characters and UTF8 bytes."""
    text = unicodedata.normalize("NFC", str(name or ""))
    text = _ILLEGAL_FS_RE.sub(" ", text)
    kept = [ch for ch in text if unicodedata.category(ch) not in _DROP_CATEGORIES]
    out = "".join(kept)
    out = re.sub(r"\s+", "_", out.strip())
    out = re.sub(r"_+", "_", out)
    out = out.strip("._ ")
    out = out[:max_len].strip("._ ")
    while out and len(out.encode("utf-8")) > max_bytes:
        out = out[:-1]
    out = out.strip("._ ")
    return out or fallback


def extract_vod_id(value: Any) -> str:
    """Accept a Twitch VOD URL or a bare numeric id."""
    text = normalize_digits(str(value or "")).strip()
    match = re.search(r"/videos/(\d+)", text)
    if match:
        return match.group(1)
    if re.fullmatch(r"\d{5,}", text):
        return text
    raise ValueError("❌ الرابط غير صحيح! لازم يكون رابط بث تويتش مثل: https://www.twitch.tv/videos/123456789 أو رقم البث فقط")


def make_clip_uid(vod_id: str, run_id: str, clip_id: int) -> str:
    """Globally unique clip id, stable across REDO runs. Example: 2934910407.r20261122T083015.07"""
    return f"{vod_id}.{run_id}.{int(clip_id):02d}"


def legacy_clip_uid(vod_id: str, clip_id: Any) -> str:
    return f"{vod_id}.legacy.{coerce_int(clip_id, 0):02d}"


def legacy_feedback_key(vod_id: str, clip_id: Any) -> str:
    """Key format used by the old Studio (vod_id + '_' + clip number)."""
    return f"{vod_id}_{coerce_int(clip_id, 0)}"


# ============================================================================
# Path layout
# ============================================================================

@dataclass(frozen=True)
class MemoryPaths:
    drive_root: str

    @property
    def dir(self) -> str:
        return os.path.join(self.drive_root, "memory")

    @property
    def rules(self) -> str:
        return os.path.join(self.dir, "my_editing_rules.md")

    @property
    def rules_history_dir(self) -> str:
        return os.path.join(self.dir, "rules_history")

    @property
    def feedback(self) -> str:
        return os.path.join(self.dir, "feedback_history.json")

    @property
    def taste(self) -> str:
        return os.path.join(self.dir, "taste_profile.json")

    @property
    def gemini_state(self) -> str:
        return os.path.join(self.dir, "gemini_state.json")

    @property
    def vocab(self) -> str:
        return os.path.join(self.dir, "vocab.txt")

    def ensure(self) -> "MemoryPaths":
        ensure_dir(self.dir)
        ensure_dir(self.rules_history_dir)
        return self


@dataclass(frozen=True)
class VodPaths:
    vod_dir: str

    @classmethod
    def from_ids(cls, drive_root: str, vod_id: str) -> "VodPaths":
        return cls(os.path.join(drive_root, "vods", str(vod_id)))

    @property
    def vod_id(self) -> str:
        return os.path.basename(os.path.normpath(self.vod_dir))

    @property
    def drive_root(self) -> str:
        return os.path.dirname(os.path.dirname(os.path.normpath(self.vod_dir)))

    @property
    def memory(self) -> MemoryPaths:
        return MemoryPaths(self.drive_root)

    # Stage 2 artifacts
    @property
    def meta(self) -> str:
        return os.path.join(self.vod_dir, "meta.json")

    @property
    def chat(self) -> str:
        return os.path.join(self.vod_dir, "chat.json")

    @property
    def chat_state(self) -> str:
        return os.path.join(self.vod_dir, "chat_state.json")

    @property
    def audio(self) -> str:
        return os.path.join(self.vod_dir, "audio.mp3")

    @property
    def features(self) -> str:
        return os.path.join(self.vod_dir, "audio_features.npz")

    @property
    def hype(self) -> str:
        return os.path.join(self.vod_dir, "hype_timeline.json")

    @property
    def transcript_parts_dir(self) -> str:
        return os.path.join(self.vod_dir, "transcript_parts")

    def part_path(self, index: int) -> str:
        return os.path.join(self.transcript_parts_dir, f"part_{int(index):03d}.json")

    @property
    def transcript(self) -> str:
        return os.path.join(self.vod_dir, "transcript.json")

    @property
    def stage2_report(self) -> str:
        return os.path.join(self.vod_dir, "stage2_report.json")

    # Stage 3 artifacts
    @property
    def stage3_dir(self) -> str:
        return os.path.join(self.vod_dir, "stage3")

    @property
    def phrases(self) -> str:
        return os.path.join(self.stage3_dir, "phrases.json")

    @property
    def local_scores(self) -> str:
        return os.path.join(self.stage3_dir, "local_scores.json")

    def run_dir(self, run_id: str) -> str:
        return os.path.join(self.stage3_dir, f"run_{run_id}")

    @property
    def clips_dir(self) -> str:
        return os.path.join(self.vod_dir, "clips")

    @property
    def clips_archive_dir(self) -> str:
        return os.path.join(self.vod_dir, "clips_archive")

    @property
    def manifest(self) -> str:
        return os.path.join(self.vod_dir, "clips_manifest.json")

    # Local scratch space (fast disk, not Drive)
    @property
    def tmp_dir(self) -> str:
        return ensure_dir(os.path.join(TMP_ROOT, self.vod_id))

    def ensure(self) -> "VodPaths":
        ensure_dir(self.vod_dir)
        ensure_dir(self.clips_dir)
        ensure_dir(self.stage3_dir)
        return self


def _latest_mtime(paths: Iterable[str], fallback: float = 0.0) -> float:
    best = fallback
    for p in paths:
        try:
            best = max(best, os.path.getmtime(p))
        except OSError:
            continue
    return best


def list_vods(drive_root: str) -> list:
    """All VOD folders under drive_root/vods, newest activity first.
    Activity uses pipeline files (transcript, manifest, meta, report), not the folder
    mtime, so writing a ZIP or temp file never changes which VOD counts as latest."""
    vods_root = os.path.join(drive_root, "vods")
    if not os.path.isdir(vods_root):
        return []
    items = []
    for name in os.listdir(vods_root):
        folder = os.path.join(vods_root, name)
        if not os.path.isdir(folder) or name.startswith("."):
            continue
        vp = VodPaths(folder)
        activity = _latest_mtime(
            (vp.transcript, vp.manifest, vp.meta, vp.stage2_report),
            fallback=0.0,
        )
        if activity == 0.0:
            try:
                activity = os.path.getmtime(folder)
            except OSError:
                activity = 0.0
        clips_count = 0
        if os.path.isfile(vp.manifest):
            raw = load_json_safe(vp.manifest, default=None, quiet=True)
            if isinstance(raw, list):
                clips_count = len(raw)
            elif isinstance(raw, dict):
                clips_count = len(raw.get("clips") or [])
        report = load_json_safe(vp.stage2_report, default={}, quiet=True) if os.path.isfile(vp.stage2_report) else {}
        duration_min = None
        if isinstance(report, dict):
            dur = coerce_number(report.get("duration_sec"), None)
            if dur:
                duration_min = round(dur / 60.0, 1)
        items.append({
            "vod_id": name,
            "vod_dir": folder,
            "activity": activity,
            "has_transcript": os.path.isfile(vp.transcript),
            "has_manifest": os.path.isfile(vp.manifest),
            "clips": clips_count,
            "duration_min": duration_min,
        })
    items.sort(key=lambda d: d["activity"], reverse=True)
    return items


def resolve_vod_dir(drive_root: str, vod_id: Any = None, require_transcript: bool = False) -> VodPaths:
    """Pick a VOD: by id or URL if given, otherwise the one with the newest pipeline activity."""
    if vod_id not in (None, "", 0):
        vid = extract_vod_id(vod_id)
        vp = VodPaths.from_ids(drive_root, vid)
        if not os.path.isdir(vp.vod_dir):
            raise FileNotFoundError(f"❌ البث رقم {vid} غير موجود في Drive ({vp.vod_dir})")
        if require_transcript and not os.path.isfile(vp.transcript):
            raise FileNotFoundError(f"❌ البث رقم {vid} ما عنده تفريغ نصي بعد. شغّل المرحلة 2 أول")
        return vp
    candidates = list_vods(drive_root)
    if require_transcript:
        candidates = [c for c in candidates if c["has_transcript"]]
    if not candidates:
        raise FileNotFoundError("❌ ما لقيت أي بث جاهز في Drive. شغّل المرحلة 2 أول")
    return VodPaths(candidates[0]["vod_dir"])


# ============================================================================
# Manifest v2
# ============================================================================

def _clean_segments(raw: Any) -> list:
    clean = []
    if not isinstance(raw, list):
        return clean
    for seg in raw:
        start = end = None
        if isinstance(seg, (list, tuple)) and len(seg) >= 2:
            start, end = parse_time_to_sec(seg[0]), parse_time_to_sec(seg[1])
        elif isinstance(seg, dict):
            start, end = parse_time_to_sec(seg.get("start")), parse_time_to_sec(seg.get("end"))
        if start is not None and end is not None and end > start:
            clean.append([round(start, 3), round(end, 3)])
    clean.sort(key=lambda x: x[0])
    return clean


def normalize_clip(clip: dict, vod_id: str) -> dict:
    """Fill every manifest v2 field with a safe default without discarding unknown keys."""
    c = dict(clip)
    segments = _clean_segments(c.get("segments"))
    if not segments:
        start, end = parse_time_to_sec(c.get("start")), parse_time_to_sec(c.get("end"))
        if start is not None and end is not None and end > start:
            segments = [[round(start, 3), round(end, 3)]]
    c["segments"] = segments
    if segments:
        c["start"] = segments[0][0]
        c["end"] = segments[-1][1]
        if coerce_number(c.get("duration"), None) is None:
            c["duration"] = round(sum(e - s for s, e in segments), 2)
    c["llm_segments"] = _clean_segments(c.get("llm_segments")) or []

    cid = coerce_number(c.get("id"), None)
    c["id"] = int(cid) if cid is not None and cid >= 0 else None

    c["title"] = str(c.get("title") or "لقطة").strip() or "لقطة"
    c["category"] = str(c.get("category") or "منوع").strip() or "منوع"
    rating = str(c.get("rating") or "pending").strip().lower()
    c["rating"] = rating if rating in VALID_RATINGS else "pending"
    score10 = coerce_number(c.get("score10"), None, fraction_scale=10.0)
    c["score10"] = clamp(round(score10, 1), 0.0, 10.0) if score10 is not None else None
    c["tags"] = [str(t) for t in c.get("tags") or [] if str(t).strip()]
    c["note"] = str(c.get("note") or "")
    c["manual_trims"] = list(c.get("manual_trims") or [])
    c["scores"] = dict(c.get("scores") or {})
    c["features"] = dict(c.get("features") or {})
    c.setdefault("reviewed", None)
    c.setdefault("models", {})
    c.setdefault("reason", "")
    c.setdefault("context_note", "")
    c.setdefault("cut_notes", "")
    c.setdefault("excerpt", "")
    c.setdefault("video_path", "")
    c.setdefault("srt_path", "")
    c.setdefault("status", "ready")
    c.setdefault("created_at", "")
    for key in ("removed_sec", "shaved_sec", "end_extended_sec"):
        val = coerce_number(c.get(key), None)
        c[key] = round(val, 2) if val is not None else 0.0
    return c


def empty_manifest(vod_id: str) -> dict:
    return {"schema": MANIFEST_SCHEMA, "vod_id": str(vod_id), "next_id": 1, "clips": []}


def load_manifest(vod_paths: VodPaths) -> dict:
    """Load clips_manifest.json as v2. Converts a v1 list automatically, repairs
    duplicate clip ids (the old len()+1 numbering bug) and guarantees unique uids."""
    vod_id = vod_paths.vod_id
    raw = load_json_safe(vod_paths.manifest, default=None)
    if raw is None:
        return empty_manifest(vod_id)

    legacy = isinstance(raw, list)
    if legacy:
        clips_raw = raw
        extra = {}
        log(f"ℹ️ تحويل ملف اللقطات للبث {vod_id} من الصيغة القديمة للصيغة الجديدة")
    elif isinstance(raw, dict):
        clips_raw = raw.get("clips") or []
        extra = {k: v for k, v in raw.items() if k not in ("schema", "vod_id", "next_id", "clips")}
    else:
        log(f"⚠️ ملف اللقطات للبث {vod_id} بصيغة غير مفهومة، بنبدأ بقائمة فاضية")
        return empty_manifest(vod_id)

    normalized = [normalize_clip(c, vod_id) for c in clips_raw if isinstance(c, dict)]
    ids_present = [c["id"] for c in normalized if c["id"] is not None]
    next_id = max(ids_present, default=0) + 1
    if not legacy:
        next_id = max(next_id, coerce_int(raw.get("next_id"), 1, lo=1))

    used_ids: set = set()
    used_uids: set = set()
    clips = []
    for c in normalized:
        original_id = c["id"]
        duplicate = original_id is not None and original_id in used_ids
        if original_id is None or duplicate:
            c["id"] = next_id
            next_id += 1
            if duplicate:
                log(f"⚠️ رقم اللقطة #{original_id:02d} مكرر في البث {vod_id}، أعطيت النسخة الثانية رقم #{c['id']:02d}")
        used_ids.add(c["id"])

        uid = str(c.get("uid") or "").strip()
        if not uid:
            uid = legacy_clip_uid(vod_id, c["id"])
            if legacy and not duplicate and original_id is not None:
                c.setdefault("legacy_key", legacy_feedback_key(vod_id, original_id))
        if uid in used_uids:
            uid = f"{uid}.dup{c['id']:02d}"
        used_uids.add(uid)
        c["uid"] = uid
        c.setdefault("run_id", "legacy" if legacy else "")
        clips.append(c)

    manifest = {"schema": MANIFEST_SCHEMA, "vod_id": vod_id, "next_id": next_id, "clips": clips}
    manifest.update(extra)
    return manifest


def save_manifest(vod_paths: VodPaths, manifest: dict) -> str:
    """Atomic save (previous good version kept as .bak). next_id never moves backwards."""
    data = dict(manifest)
    data["schema"] = MANIFEST_SCHEMA
    data["vod_id"] = vod_paths.vod_id
    clips = [c for c in data.get("clips") or [] if isinstance(c, dict)]
    clips.sort(key=lambda c: coerce_int(c.get("id"), 0))
    data["clips"] = clips
    max_id = max((coerce_int(c.get("id"), 0) for c in clips), default=0)
    data["next_id"] = max(coerce_int(data.get("next_id"), 1, lo=1), max_id + 1)
    return atomic_write_json(vod_paths.manifest, data, backup=True)


def allocate_clip_id(manifest: dict) -> int:
    """Reserve the next display number. Numbers are never reused, even after deletions."""
    clips = manifest.get("clips") or []
    max_id = max((coerce_int(c.get("id"), 0) for c in clips), default=0)
    new_id = max(coerce_int(manifest.get("next_id"), 1, lo=1), max_id + 1)
    manifest["next_id"] = new_id + 1
    return new_id


def find_clip(manifest: dict, uid: Optional[str] = None, clip_id: Any = None) -> tuple:
    """Return (index, clip) by uid first, then by display id. (None, None) if not found."""
    clips = manifest.get("clips") or []
    if uid:
        for idx, c in enumerate(clips):
            if c.get("uid") == uid:
                return idx, c
    if clip_id is not None:
        want = coerce_int(clip_id, -1)
        for idx, c in enumerate(clips):
            if coerce_int(c.get("id"), -2) == want:
                return idx, c
    return None, None


# ============================================================================
# Feedback v2
# ============================================================================

def normalize_feedback_entry(entry: dict) -> dict:
    e = dict(entry)
    if not str(e.get("uid") or "").strip():
        key = str(e.get("key") or "").strip()
        if not key:
            key = legacy_feedback_key(str(e.get("vod_id") or "unknown"), e.get("clip_id"))
        e["uid"] = f"legacy:{key}"
    rating = str(e.get("rating") or "pending").strip().lower()
    e["rating"] = rating if rating in VALID_RATINGS else "pending"
    score10 = coerce_number(e.get("score10"), None, fraction_scale=10.0)
    e["score10"] = clamp(round(score10, 1), 0.0, 10.0) if score10 is not None else None
    e["tags"] = [str(t) for t in e.get("tags") or [] if str(t).strip()]
    e["note"] = str(e.get("note") or "")
    e["manual_trims"] = list(e.get("manual_trims") or [])
    e["features"] = dict(e.get("features") or {})
    e["segments"] = _clean_segments(e.get("segments"))
    e.setdefault("schema", FEEDBACK_SCHEMA if not e["uid"].startswith("legacy:") else 1)
    e.setdefault("updated_at", "")
    return e


def load_feedback(memory_paths: MemoryPaths) -> list:
    """Load feedback_history.json (a JSON list, same container as before). Legacy
    entries keep their key and receive uid 'legacy:<key>'. Later duplicates win."""
    raw = load_json_safe(memory_paths.feedback, default=[])
    if isinstance(raw, dict):
        raw = raw.get("entries") or []
    if not isinstance(raw, list):
        return []
    by_uid: dict = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = normalize_feedback_entry(item)
        by_uid[entry["uid"]] = entry
    return list(by_uid.values())


def save_feedback(memory_paths: MemoryPaths, entries: Sequence[dict]) -> str:
    memory_paths.ensure()
    clean = [normalize_feedback_entry(e) for e in entries if isinstance(e, dict)]
    return atomic_write_json(memory_paths.feedback, clean, backup=True)


def upsert_feedback(entries: Sequence[dict], entry: dict) -> list:
    """Replace the entry with the same uid, or append. Returns a new list."""
    new_entry = normalize_feedback_entry(entry)
    out = []
    replaced = False
    for e in entries:
        if isinstance(e, dict) and e.get("uid") == new_entry["uid"]:
            out.append(new_entry)
            replaced = True
        else:
            out.append(e)
    if not replaced:
        out.append(new_entry)
    return out


def remove_feedback(entries: Sequence[dict], uid: str) -> list:
    return [e for e in entries if not (isinstance(e, dict) and e.get("uid") == uid)]


# ============================================================================
# Subprocess helper
# ============================================================================

@dataclass
class CommandResult:
    returncode: int
    stdout: Any
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def last_error_line(self) -> str:
        text = (self.stderr or "").strip()
        if self.timed_out:
            return "انتهت المهلة (timeout)"
        if not text:
            return f"exit code {self.returncode}"
        return text.splitlines()[-1][:300]


def run_command(
    cmd: Sequence[str],
    timeout: Optional[float] = None,
    text: bool = True,
    input_data: Any = None,
) -> CommandResult:
    """Run a command, capture output, never raise for timeouts or missing tools.
    With text=False stdout is bytes (stderr is always decoded text)."""
    try:
        res = subprocess.run(
            list(cmd),
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=text,
        )
        stderr = res.stderr if text else (res.stderr or b"").decode("utf-8", "replace")
        return CommandResult(res.returncode, res.stdout, stderr or "", False)
    except subprocess.TimeoutExpired as exc:
        err = exc.stderr
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return CommandResult(-1, exc.stdout if exc.stdout is not None else ("" if text else b""), err or "", True)
    except FileNotFoundError:
        return CommandResult(127, "" if text else b"", f"الأداة غير موجودة: {cmd[0] if cmd else ''}", False)


def tool_available(name: str) -> bool:
    return shutil.which(name) is not None


def chunked(seq: Sequence[Any], size: int) -> Iterator[list]:
    size = max(1, int(size))
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


# ============================================================================
# Module reload helper (Colab GPU to CPU runtime switches, git pull)
# ============================================================================

def reload_project(repo_dir: str, modules: Optional[Iterable[str]] = None) -> dict:
    """Clear stale import caches and (re)load project modules in dependency order.
    Keeps the battle tested fix: sys.path_importer_cache.pop + invalidate_caches + reload."""
    repo_dir = os.path.abspath(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    sys.path_importer_cache.pop(repo_dir, None)
    importlib.invalidate_caches()

    wanted = set(modules) if modules is not None else set(PROJECT_MODULES)
    ordered = [m for m in PROJECT_MODULES if m in wanted]
    ordered += sorted(m for m in wanted if m not in PROJECT_MODULES)

    loaded = {}
    for name in ordered:
        if not os.path.isfile(os.path.join(repo_dir, f"{name}.py")):
            log(f"⚠️ الملف {name}.py غير موجود في {repo_dir}، تم تخطيه")
            continue
        try:
            if name in sys.modules:
                loaded[name] = importlib.reload(sys.modules[name])
            else:
                loaded[name] = importlib.import_module(name)
        except Exception as exc:
            log_exception(f"تعذر تحميل {name}.py", exc)
            raise
    if loaded:
        log("🔄 تم تحميل ملفات المشروع: " + "، ".join(loaded.keys()))
    return loaded


# === END OF common.py ===
