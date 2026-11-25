"""
stage2_ingest.py
Stage 2 of TwitchClipper (Colab T4): chat, audio, VAD, Whisper transcription in
checkpointed blocks, coverage audit with targeted repair, audio features, AST audio
event tagger, and the compatibility hype timeline.

v3 change
  * Switching between the quick test mode and the full VOD resets the old mode's files on
    Drive strictly through common.safe_delete(path, drive_root), which only works inside
    TwitchClipper/vods and refuses anything under TwitchClipper/memory. v3 outputs of the old
    mode (Stage 3 previews, phrases.json) are reset too, so no stale preview or run can be
    mixed with the new transcript. Liked clips already archived in clips_archive/ are kept.
    Raw os.remove is used only for scratch files under /tmp, never on Drive.

Battle tested fixes kept
  * Pure Python Twitch GQL chat fetcher (TwitchDownloaderCLI crashes with SIGSEGV on Colab).
    Paging is now by content offset (the mode Twitch does not challenge with an integrity
    check). The old cursor path stays as a fallback with its seen cursor guard. Strict
    (5, 12) second timeouts per request.
  * Audio is decoded with FFmpeg into a 16 kHz float32 NumPy array before faster-whisper.
    No file path is ever passed to model.transcribe (PyAV breaks on Python 3.13).
  * CUDA 12 runtime preload with ctypes.CDLL(..., RTLD_GLOBAL) before the model loads.

Upgrade behaviour for an existing VOD
  * transcript.json v1 is split into checkpoint parts without transcribing anything again.
  * Only coverage buckets with VAD speech but too few words are repaired.
  * AFK stretches (long spans with almost no speech) are skipped by the primary pass, the
    repair pass and the AST tagger, so no T4 time is spent on hours of silence.
  * Chat is fetched again when chat_state.json is missing or incomplete; the old file is
    replaced only if the new fetch has at least as many messages.

Public contract
  run_stage2(vod_url, drive_root, test_minutes=0, whisper_model="large-v3-turbo",
             redo_transcript=False, run_tagger=True) -> vod_dir (str)
"""

from __future__ import annotations

import ctypes
import dataclasses
import gc
import glob
import inspect
import math
import os
import re
import shutil
import site
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np
import requests

from common import (
    TMP_ROOT,
    TRANSCRIPT_SCHEMA,
    MemoryPaths,
    VodPaths,
    atomic_write_json,
    ensure_dir,
    extract_vod_id,
    format_duration_ar,
    human_size,
    load_json_safe,
    log,
    normalize_arabic,
    now_iso,
    read_text_safe,
    safe_copy_file,
    safe_delete,
    sec_to_clock,
    short_error,
)
from media_engine import Features, afk_spans, collect_words, compute_dsp_features, save_features
from taste_engine import chat_health


# ============================================================================
# Constants
# ============================================================================

TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
CHAT_QUERY_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"

HYPE_PATTERNS = re.compile(
    r"(ه{3,}|😂|💀|😭|🔥|\?{2,}|؟{2,}|وش|اما|لاا+|كفو|موت|ضحك|lul|kekw|omegalul|lmao|clip)",
    re.IGNORECASE,
)

SR = 16000
BLOCK_S = 600.0
BLOCK_SNAP_S = 60.0
BUCKET_S = 300.0
PARAMS_ID = "ar_v2"

VAD_PRIMARY = {"threshold": 0.40, "min_speech_duration_ms": 150, "min_silence_duration_ms": 400,
               "speech_pad_ms": 250, "max_speech_duration_s": 28}
VAD_REPAIR = {"threshold": 0.25, "min_speech_duration_ms": 100, "min_silence_duration_ms": 300,
              "speech_pad_ms": 300, "max_speech_duration_s": 28}

WHISPER_PRIMARY = {
    "language": "ar",
    "task": "transcribe",
    "beam_size": 5,
    "best_of": 5,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "condition_on_previous_text": False,
    "compression_ratio_threshold": 2.4,
    "log_prob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "word_timestamps": True,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": True,
    "vad_parameters": VAD_PRIMARY,
}
WHISPER_REPAIR = dict(WHISPER_PRIMARY, no_speech_threshold=None, log_prob_threshold=None,
                      vad_parameters=VAD_REPAIR)

HALLUCINATION_PHRASES = tuple(normalize_arabic(p) for p in (
    "اشتركوا في القناة", "اشتركوا بالقناة", "اشترك في القناة", "لا تنسوا الاشتراك", "لا تنسى الاشتراك",
    "ترجمة نانسي قنقر", "نانسي قنقر", "شكرا للمشاهدة", "شكرا على المشاهدة", "شكرا لكم على المشاهدة",
    "سبحانك اللهم وبحمدك", "موسيقى", "[موسيقى]", "(موسيقى)", "thank you for watching",
    "subtitles by", "amara.org", "تفريغ نصي",
))

AST_MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_LABELS = {
    "laugh": ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"),
    "scream": ("Screaming",),
    "shout": ("Shout", "Yell", "Bellow", "Whoop", "Children shouting"),
    "music": ("Music",),
    "game_sfx": ("Gunshot, gunfire", "Machine gun", "Explosion", "Boom", "Burst, pop"),
}
AST_WINDOW_S = 4.0
AST_BATCH = 32

CHAT_FULL_BUDGET_S = 900
CHAT_TEST_BUDGET_S = 60
CHAT_SAVE_EVERY = 50


# ============================================================================
# CUDA preload (battle tested, unchanged logic)
# ============================================================================

def preload_cuda_libs() -> None:
    dirs = set()
    roots = list(site.getsitepackages()) + [p for p in sys.path if "packages" in p]
    for root in roots:
        if os.path.isdir(root):
            for d in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
                if os.path.isdir(d):
                    dirs.add(d)
    priority = [
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
    ]
    for lib_name in priority:
        for d in sorted(dirs):
            path = os.path.join(d, lib_name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def cuda_available() -> bool:
    try:
        import ctranslate2  # noqa: WPS433
        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


# ============================================================================
# Audio decode (battle tested: FFmpeg to 16 kHz float32 NumPy, never a file path to Whisper)
# ============================================================================

def load_audio_ffmpeg(audio_path: str, sr: int = SR) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", audio_path,
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0


# ============================================================================
# Checkpoint and meta
# ============================================================================

def _normalize_test_minutes(test_minutes: Any) -> Optional[int]:
    if test_minutes is None or test_minutes == "":
        return None
    try:
        val = int(float(test_minutes))
    except (TypeError, ValueError):
        return None
    return val if val > 0 else None


def _vad_path(vp: VodPaths) -> str:
    return os.path.join(vp.vod_dir, "vad_speech.json")


def _plan_path(vp: VodPaths) -> str:
    return os.path.join(vp.transcript_parts_dir, "plan.json")


def _mode_reset_targets(vp: VodPaths) -> list:
    """Drive files and folders that belong to one mode (test or full) of a VOD.
    clips_archive/ is not listed: liked clips archived by the Studio are always kept."""
    files = (vp.chat, vp.chat_state, vp.audio, vp.hype, vp.transcript, vp.manifest,
             vp.features, vp.stage2_report, _vad_path(vp), vp.phrases)
    targets = []
    for path in files:
        for candidate in (path, path + ".bak"):
            if os.path.isfile(candidate):
                targets.append(candidate)
    for folder in (vp.clips_dir, vp.transcript_parts_dir, vp.stage3_dir, vp.previews_dir):
        if os.path.isdir(folder):
            targets.append(folder)
    return targets


def _reset_mode_files(vp: VodPaths) -> int:
    """Delete the previous mode's files through safe_delete only. Returns bytes freed.
    A refused path (outside TwitchClipper/vods or inside memory) is reported and skipped."""
    freed = 0
    refused = []
    for target in _mode_reset_targets(vp):
        try:
            freed += int(safe_delete(target, vp.drive_root) or 0)
        except PermissionError as exc:
            refused.append(f"{os.path.basename(target)} ({short_error(exc, 80)})")
        except OSError as exc:
            refused.append(f"{os.path.basename(target)} ({short_error(exc, 80)})")
    if refused:
        log("⚠️ ما قدرنا نحذف هذي الملفات أثناء التصفير: " + "، ".join(refused))
    return freed


def _prepare_checkpoint(vp: VodPaths, vod_url: str, test_minutes: Optional[int], whisper_model: str) -> dict:
    old_meta = load_json_safe(vp.meta, default=None, expected_types=(dict,), quiet=True)
    if old_meta and old_meta.get("test_minutes") != test_minutes:
        old_mode = f"تجربة {old_meta.get('test_minutes')} دقيقة" if old_meta.get("test_minutes") else "البث الكامل"
        new_mode = f"تجربة {test_minutes} دقيقة" if test_minutes else "البث الكامل"
        freed = _reset_mode_files(vp)
        log(f"🔄 تحولنا من {old_mode} إلى {new_mode}، وصفّرنا ملفات الوضع السابق بأمان "
            f"(وفرنا {human_size(freed)}). الذاكرة والتقييمات والأرشيف ما انلمست")
    meta = dict(old_meta or {})
    meta.update({
        "vod_id": vp.vod_id,
        "vod_url": vod_url,
        "test_minutes": test_minutes,
        "whisper_model": whisper_model,
        "updated_at": now_iso(),
    })
    meta.setdefault("created_at", now_iso())
    atomic_write_json(vp.meta, meta)
    return meta


# ============================================================================
# Chat (pure Python GQL)
# ============================================================================

def _gql_post(payload: list, retries: int = 3) -> tuple:
    """Returns (json_or_None, error_text_or_None). Network errors are retried with backoff."""
    headers = {"Client-ID": TWITCH_CLIENT_ID, "Content-Type": "application/json"}
    last = ""
    for attempt in range(retries):
        try:
            resp = requests.post(TWITCH_GQL_URL, headers=headers, json=payload, timeout=(5, 12))
            if resp.status_code >= 500:
                last = f"HTTP {resp.status_code}"
            elif resp.status_code >= 400:
                return None, f"HTTP {resp.status_code}"
            else:
                return resp.json(), None
        except Exception as exc:
            last = short_error(exc, 120)
        time.sleep((2, 5, 10)[min(attempt, 2)])
    return None, last or "network error"


def _parse_comments(data: Any) -> tuple:
    """(edges, has_next, error_text). Twitch errors are surfaced, never swallowed."""
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return [], False, "رد غير متوقع من تويتش"
    first = data[0]
    errors = first.get("errors")
    if errors:
        msg = errors[0].get("message") if isinstance(errors, list) and errors and isinstance(errors[0], dict) else str(errors)
        return [], False, f"تويتش رفض الطلب: {msg}"
    video = (first.get("data") or {}).get("video")
    if not video:
        return [], False, "البث غير موجود أو محذوف أو خاص"
    comments = video.get("comments")
    if not comments:
        return [], False, None
    edges = comments.get("edges") or []
    has_next = bool((comments.get("pageInfo") or {}).get("hasNextPage"))
    return edges, has_next, None


def _edge_message(edge: dict) -> Optional[dict]:
    node = edge.get("node") or {}
    fragments = (node.get("message") or {}).get("fragments") or []
    text = "".join(str(f.get("text", "")) for f in fragments if isinstance(f, dict)).strip()
    if not text:
        return None
    has_emote = any(isinstance(f, dict) and f.get("emote") is not None for f in fragments)
    user = (node.get("commenter") or {}).get("displayName", "unknown")
    offset = node.get("contentOffsetSeconds", 0) or 0
    return {"t": int(offset), "u": user, "m": text, "e": has_emote, "id": node.get("id") or f"{offset}:{user}:{text[:20]}"}


def _query(variables: dict) -> list:
    return [{
        "operationName": "VideoCommentsByOffsetOrCursor",
        "variables": variables,
        "extensions": {"persistedQuery": {"version": 1, "sha256Hash": CHAT_QUERY_HASH}},
    }]


def _save_chat(vp: VodPaths, messages: list, state: dict) -> None:
    messages.sort(key=lambda m: (m.get("t", 0), str(m.get("id", ""))))
    atomic_write_json(vp.chat, messages, indent=None, backup=False)
    atomic_write_json(vp.chat_state, state, backup=False)


def _fetch_offset_mode(vod_id: str, vp: VodPaths, messages: list, seen: set, start_offset: int,
                       max_sec: Optional[int], budget_s: float) -> tuple:
    offset = int(start_offset)
    pages = 0
    stagnant = 0
    t0 = time.time()
    reason = "complete"
    while True:
        if time.time() - t0 > budget_s:
            reason = "time_budget"
            break
        data, net_err = _gql_post(_query({"videoID": str(vod_id), "contentOffsetSeconds": offset}))
        if net_err:
            reason = f"network_error: {net_err}"
            break
        edges, has_next, err = _parse_comments(data)
        if err:
            reason = f"twitch_error: {err}"
            break
        if not edges:
            break
        new = 0
        max_off = offset
        hit_limit = False
        for edge in edges:
            msg = _edge_message(edge)
            node_off = int((edge.get("node") or {}).get("contentOffsetSeconds", offset) or offset)
            max_off = max(max_off, node_off)
            if max_sec is not None and node_off > max_sec:
                hit_limit = True
                continue
            if msg and msg["id"] not in seen:
                seen.add(msg["id"])
                messages.append(msg)
                new += 1
        pages += 1
        if hit_limit:
            reason = "test_limit"
            break
        if not has_next:
            break
        if new == 0:
            stagnant += 1
            offset = max(offset + 1 + 4 * (stagnant >= 3), max_off)
            if stagnant >= 20:
                reason = "stalled"
                break
        else:
            stagnant = 0
            offset = max_off if max_off > offset else offset + 1
        if pages % CHAT_SAVE_EVERY == 0:
            _save_chat(vp, messages, {"stop_reason": "in_progress", "offset": offset, "count": len(messages),
                                      "mode": "offset", "updated_at": now_iso()})
            log(f"   ... تم سحب {len(messages)} رسالة (وصلنا للدقيقة {offset / 60:.1f})")
    return reason, offset, pages


def _fetch_cursor_mode(vod_id: str, messages: list, seen: set, max_sec: Optional[int], budget_s: float) -> tuple:
    """Original cursor loop, kept as a fallback with its seen cursor guard."""
    seen_cursors: set = set()
    cursor = None
    pages = 0
    t0 = time.time()
    reason = "complete"
    while pages < 2500:
        if time.time() - t0 > budget_s:
            reason = "time_budget"
            break
        variables = {"videoID": str(vod_id)}
        if cursor:
            variables["cursor"] = cursor
        else:
            variables["contentOffsetSeconds"] = 0
        data, net_err = _gql_post(_query(variables))
        if net_err:
            reason = f"network_error: {net_err}"
            break
        edges, has_next, err = _parse_comments(data)
        if err:
            reason = f"twitch_error: {err}"
            break
        if not edges:
            break
        hit_limit = False
        for edge in edges:
            node_off = int((edge.get("node") or {}).get("contentOffsetSeconds", 0) or 0)
            if max_sec is not None and node_off > max_sec:
                hit_limit = True
                break
            msg = _edge_message(edge)
            if msg and msg["id"] not in seen:
                seen.add(msg["id"])
                messages.append(msg)
        pages += 1
        if hit_limit:
            reason = "test_limit"
            break
        if not has_next:
            break
        next_cursor = edges[-1].get("cursor")
        if not next_cursor or next_cursor in seen_cursors:
            reason = "cursor_loop" if next_cursor else "no_cursor"
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return reason, pages


_REASON_AR = {
    "complete": "اكتمل السحب",
    "time_budget": "انتهت المهلة",
    "test_limit": "وصلنا حد وضع التجربة",
    "stalled": "توقف التقدم",
    "cursor_loop": "تكرار المؤشر",
    "no_cursor": "انتهت الصفحات",
}


def download_chat(vod_id: str, vp: VodPaths, test_minutes: Optional[int] = None) -> list:
    """Fetch chat with offset paging, resume an interrupted fetch, keep the old file if the
    new result is smaller, and record the stop reason in chat_state.json."""
    log(f"💬 جاري سحب شات البث ({vod_id})...")
    max_sec = test_minutes * 60 if test_minutes else None
    budget = CHAT_TEST_BUDGET_S if test_minutes else CHAT_FULL_BUDGET_S
    old_messages = load_json_safe(vp.chat, default=[], expected_types=(list,), quiet=True) or []
    state = load_json_safe(vp.chat_state, default={}, expected_types=(dict,), quiet=True) or {}

    messages: list = []
    seen: set = set()
    start_offset = 0
    if state.get("stop_reason") == "in_progress" and old_messages:
        messages = [m for m in old_messages if isinstance(m, dict)]
        seen = {m.get("id") for m in messages if m.get("id")}
        start_offset = int(state.get("offset", 0) or 0)
        log(f"   ♻️ نكمل سحب الشات من الدقيقة {start_offset / 60:.1f} ({len(messages)} رسالة محفوظة)")

    reason, offset, pages = _fetch_offset_mode(vod_id, vp, messages, seen, start_offset, max_sec, budget)
    mode = "offset"
    if reason.startswith("twitch_error") and not messages:
        log(f"   ⚠️ طريقة الإزاحة رجعت خطأ ({reason})، نجرب طريقة المؤشر القديمة")
        reason, pages = _fetch_cursor_mode(vod_id, messages, seen, max_sec, budget)
        mode = "cursor"

    if len(messages) < len(old_messages) and state.get("stop_reason") != "in_progress":
        log(f"   ℹ️ السحب الجديد ({len(messages)} رسالة) أقل من الملف القديم ({len(old_messages)})، نبقي القديم")
        messages = [m for m in old_messages if isinstance(m, dict)]
    final_state = {
        "stop_reason": reason, "mode": mode, "offset": offset, "pages": pages,
        "count": len(messages), "updated_at": now_iso(),
    }
    _save_chat(vp, messages, final_state)
    reason_ar = _REASON_AR.get(reason.split(":")[0], reason)
    if reason.startswith(("twitch_error", "network_error")):
        log(f"⚠️ سحب الشات توقف: {reason}")
    log(f"✅ تم حفظ الشات ({len(messages)} رسالة، {reason_ar})")
    return messages


def _chat_is_complete(vp: VodPaths) -> bool:
    state = load_json_safe(vp.chat_state, default={}, expected_types=(dict,), quiet=True) or {}
    return os.path.isfile(vp.chat) and state.get("stop_reason") in ("complete", "test_limit")


# ============================================================================
# Audio download (to /tmp first, then atomic copy to Drive)
# ============================================================================

def download_audio(vod_url: str, vp: VodPaths, test_minutes: Optional[int] = None) -> None:
    log("🎧 جاري تحميل صوت البث فقط (بدون فيديو لتوفير الوقت والمساحة)...")
    work = ensure_dir(os.path.join(TMP_ROOT, "audio_dl", vp.vod_id))
    for name in os.listdir(work):
        try:
            os.remove(os.path.join(work, name))
        except OSError:
            pass
    template = os.path.join(work, "audio.%(ext)s")
    tmp_mp3 = os.path.join(work, "audio.mp3")
    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--no-playlist",
        "--retries", "10",
        "--fragment-retries", "50",
        "--concurrent-fragments", "4",
        "-o", template,
    ]
    if test_minutes:
        end_sec = int(test_minutes * 60)
        h, rem = divmod(end_sec, 3600)
        m, s = divmod(rem, 60)
        cmd += ["--download-sections", f"*00:00:00-{h:02d}:{m:02d}:{s:02d}"]
        log(f"   🧪 وضع التجربة السريعة: أول {test_minutes} دقائق فقط")
    else:
        log("   🚀 وضع البث الكامل، جاري سحب صوت البث من بدايته لنهايته...")
    cmd.append(vod_url)

    last_err = None
    for attempt in range(2):
        try:
            subprocess.run(cmd, check=True)
            if os.path.isfile(tmp_mp3) and os.path.getsize(tmp_mp3) > 10_000:
                break
            last_err = RuntimeError("ملف الصوت ما انكتب")
        except subprocess.CalledProcessError as exc:
            last_err = exc
        if attempt == 0:
            log("   🔁 فشل التحميل، نعيد المحاولة مرة وحدة...")
            time.sleep(5)
    else:
        raise RuntimeError(f"❌ تعذر تحميل صوت البث: {short_error(last_err) if last_err else 'سبب غير معروف'}")

    safe_copy_file(tmp_mp3, vp.audio)
    size_mb = os.path.getsize(tmp_mp3) / (1024 * 1024)
    os.remove(tmp_mp3)
    log(f"✅ تم تحميل الصوت بنجاح ({size_mb:.1f} MB)")


# ============================================================================
# VAD
# ============================================================================

def _vad_options(params: dict):
    from faster_whisper.vad import VadOptions
    if dataclasses.is_dataclass(VadOptions):
        allowed = {f.name for f in dataclasses.fields(VadOptions)}
    else:
        allowed = set(getattr(VadOptions, "_fields", ()))
    return VadOptions(**{k: v for k, v in params.items() if k in allowed})


def run_vad(audio16: np.ndarray, params: dict = VAD_PRIMARY) -> list:
    """Speech spans in seconds using the Silero VAD bundled with faster-whisper (CPU)."""
    from faster_whisper.vad import get_speech_timestamps
    opts = _vad_options(params)
    try:
        stamps = get_speech_timestamps(audio16, vad_options=opts)
    except TypeError:
        stamps = get_speech_timestamps(audio16, opts)
    return [[round(st["start"] / SR, 3), round(st["end"] / SR, 3)] for st in stamps]


def load_or_run_vad(vp: VodPaths, audio16: np.ndarray, duration: float) -> list:
    cached = load_json_safe(_vad_path(vp), default=None, expected_types=(dict,), quiet=True)
    if cached and abs(float(cached.get("duration", 0)) - duration) < 1.0 and isinstance(cached.get("spans"), list):
        log(f"⏭️ كشف الكلام (VAD) موجود مسبقاً ({len(cached['spans'])} مقطع)")
        return cached["spans"]
    log("🗣️ جاري كشف مناطق الكلام في البث (VAD)...")
    t0 = time.time()
    spans = run_vad(audio16)
    speech_s = sum(e - s for s, e in spans)
    atomic_write_json(_vad_path(vp), {"duration": round(duration, 2), "params": VAD_PRIMARY, "spans": spans},
                      indent=None, backup=False)
    log(f"✅ كشف الكلام اكتمل: {speech_s / 60:.1f} دقيقة كلام من {duration / 60:.1f} دقيقة ({time.time() - t0:.0f} ثانية)")
    return spans


def _speech_seconds(spans: Sequence[Sequence[float]], t0: float, t1: float) -> float:
    total = 0.0
    for s, e in spans:
        if e <= t0:
            continue
        if s >= t1:
            break
        total += min(e, t1) - max(s, t0)
    return total


def _in_spans(spans: Sequence[Sequence[float]], t0: float, t1: float, min_frac: float = 0.9) -> bool:
    if t1 <= t0:
        return False
    covered = 0.0
    for s, e in spans:
        covered += max(0.0, min(e, t1) - max(s, t0))
    return covered / (t1 - t0) >= min_frac


# ============================================================================
# Block plan and parts
# ============================================================================

def plan_blocks(duration: float, speech_spans: Sequence[Sequence[float]]) -> list:
    """About 10 minute blocks whose edges move into the middle of the nearest gap between
    speech spans (within 60 s), so no block boundary splits a sentence."""
    if duration <= BLOCK_S * 1.5:
        return [[0.0, round(duration, 3)]]
    gaps = []
    prev_end = 0.0
    for s, e in speech_spans:
        if s > prev_end:
            gaps.append(((prev_end + s) / 2.0, s - prev_end))
        prev_end = max(prev_end, e)
    if duration > prev_end:
        gaps.append(((prev_end + duration) / 2.0, duration - prev_end))
    edges = [0.0]
    target = BLOCK_S
    while target < duration - BLOCK_S * 0.5:
        near = [g for g in gaps if abs(g[0] - target) <= BLOCK_SNAP_S and g[0] > edges[-1] + 60]
        cut = max(near, key=lambda g: g[1])[0] if near else target
        edges.append(round(cut, 3))
        target = cut + BLOCK_S
    edges.append(round(duration, 3))
    return [[edges[i], edges[i + 1]] for i in range(len(edges) - 1)]


def load_or_make_plan(vp: VodPaths, duration: float, speech_spans: list) -> list:
    plan = load_json_safe(_plan_path(vp), default=None, expected_types=(dict,), quiet=True)
    if plan and abs(float(plan.get("duration", 0)) - duration) < 1.0 and isinstance(plan.get("blocks"), list):
        return plan["blocks"]
    blocks = plan_blocks(duration, speech_spans)
    ensure_dir(vp.transcript_parts_dir)
    atomic_write_json(_plan_path(vp), {"duration": round(duration, 2), "blocks": blocks}, backup=False)
    return blocks


def _load_part(vp: VodPaths, index: int) -> Optional[dict]:
    part = load_json_safe(vp.part_path(index), default=None, expected_types=(dict,), quiet=True)
    if part and part.get("status") in ("done", "silent", "legacy") and isinstance(part.get("segments"), list):
        return part
    return None


def _write_part(vp: VodPaths, index: int, block: Sequence[float], segments: list, status: str, params: str) -> None:
    atomic_write_json(vp.part_path(index), {
        "index": index, "start": block[0], "end": block[1], "status": status, "params": params,
        "segments": segments, "updated_at": now_iso(),
    }, indent=None, backup=False)


def split_legacy_transcript(vp: VodPaths, legacy: dict, blocks: list) -> int:
    """Turn an old single transcript.json into checkpoint parts (no transcription)."""
    segs = [s for s in legacy.get("segments", []) or [] if isinstance(s, dict)]
    written = 0
    for i, (b0, b1) in enumerate(blocks):
        if _load_part(vp, i) is not None:
            continue
        inside = [s for s in segs if b0 <= float(s.get("start", 0)) < b1 or (i == len(blocks) - 1 and float(s.get("start", 0)) >= b1)]
        _write_part(vp, i, (b0, b1), inside, "legacy", "legacy")
        written += 1
    return written


# ============================================================================
# Hallucination filter
# ============================================================================

def _load_hotwords(memory: MemoryPaths) -> list:
    text = read_text_safe(memory.vocab, "")
    words = [w.strip() for w in re.split(r"[\n,،]+", text) if w.strip()]
    return words[:20]


def filter_hallucinations(segments: list, speech_spans: Sequence[Sequence[float]], hotwords: Sequence[str]) -> tuple:
    """Drop known Whisper phantom lines, the same line repeated back to back 3+ times (a decoder
    loop: each copy starts within 1.5 s of the previous one ending), high compression
    lines with no VAD speech, and lines made mostly of hotwords (prompt bleed)."""
    hot = {normalize_arabic(w) for w in hotwords}
    kept, dropped = [], 0
    history: list = []
    for seg in segments:
        norm = normalize_arabic(seg.get("text", ""))
        if not norm:
            dropped += 1
            continue
        n_words = len(norm.split())
        if n_words <= 12 and any(p and p in norm for p in HALLUCINATION_PHRASES):
            dropped += 1
            continue
        start = float(seg.get("start", 0))
        if (len(history) >= 2 and history[-1][0] == norm and history[-2][0] == norm
                and start - history[-1][1] < 1.5 and history[-1][2] - history[-2][1] < 1.5):
            dropped += 1
            continue
        speech = _speech_seconds(speech_spans, float(seg["start"]), float(seg["end"]))
        dur = max(0.01, float(seg["end"]) - float(seg["start"]))
        if float(seg.get("compression_ratio") or 0.0) > 2.4 and speech / dur < 0.2:
            dropped += 1
            continue
        if hot and n_words >= 2 and sum(1 for t in norm.split() if t in hot) / n_words >= 0.8:
            dropped += 1
            continue
        history.append((norm, float(seg["end"]), start))
        kept.append(seg)
    return kept, dropped


# ============================================================================
# Whisper
# ============================================================================

class WhisperRunner:
    """Loads faster-whisper once on demand (after the CUDA preload) and transcribes array slices."""

    def __init__(self, model_name: str, hotwords: Sequence[str]):
        self.model_name = model_name
        self.hotwords = list(hotwords)
        self.model = None
        self._allowed: Optional[set] = None

    def _load(self) -> None:
        if self.model is not None:
            return
        log(f"🎙️ جاري تجهيز مكتبات CUDA وتحميل موديل Faster-Whisper ({self.model_name})...")
        preload_cuda_libs()
        from faster_whisper import WhisperModel
        if cuda_available():
            self.model = WhisperModel(self.model_name, device="cuda", compute_type="float16")
        else:
            log("⚠️ ما فيه كرت شاشة، التفريغ على المعالج بيكون بطيء جداً. الأفضل تشغيل المرحلة 2 على T4")
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        self._allowed = set(inspect.signature(self.model.transcribe).parameters)

    def transcribe(self, audio: np.ndarray, offset: float, options: dict) -> list:
        self._load()
        opts = dict(options)
        if self.hotwords:
            opts["hotwords"] = " ".join(self.hotwords)
        opts = {k: v for k, v in opts.items() if k in (self._allowed or set())}
        segments_gen, _info = self.model.transcribe(audio, **opts)
        out = []
        for seg in segments_gen:
            text = (seg.text or "").strip()
            if not text:
                continue
            words = []
            for w in seg.words or []:
                token = (w.word or "").strip()
                if token:
                    words.append({"w": token, "s": round(float(w.start) + offset, 2),
                                  "e": round(float(w.end) + offset, 2),
                                  "p": round(float(getattr(w, "probability", 0.0) or 0.0), 3)})
            out.append({
                "start": round(float(seg.start) + offset, 2),
                "end": round(float(seg.end) + offset, 2),
                "text": text,
                "words": words,
                "avg_logprob": round(float(getattr(seg, "avg_logprob", 0.0) or 0.0), 3),
                "no_speech_prob": round(float(getattr(seg, "no_speech_prob", 0.0) or 0.0), 3),
                "compression_ratio": round(float(getattr(seg, "compression_ratio", 0.0) or 0.0), 3),
                "flags": [],
            })
        return out

    def close(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()


def transcribe_blocks(vp: VodPaths, audio16: np.ndarray, blocks: list, speech_spans: list,
                      afk: list, runner: WhisperRunner, redo: bool = False) -> dict:
    """Primary pass: one checkpoint part per block. Done parts are skipped; blocks with almost
    no speech or inside AFK spans are written as silent without touching the GPU, and AFK
    stretches inside a block are cut out before transcribing."""
    stats = {"transcribed": 0, "skipped_done": 0, "silent": 0, "segments": 0, "seconds": 0.0}
    todo = [i for i in range(len(blocks)) if redo or _load_part(vp, i) is None]
    stats["skipped_done"] = len(blocks) - len(todo)
    if stats["skipped_done"]:
        log(f"⏭️ {stats['skipped_done']} جزء من التفريغ جاهز مسبقاً، نكمل الباقي ({len(todo)} جزء)")
    for n, i in enumerate(todo, start=1):
        b0, b1 = blocks[i]
        speech = _speech_seconds(speech_spans, b0, b1)
        if speech < 0.01 * (b1 - b0) or _in_spans(afk, b0, b1, 0.95):
            _write_part(vp, i, (b0, b1), [], "silent", PARAMS_ID)
            stats["silent"] += 1
            continue
        t0 = time.time()
        segs = []
        for r0, r1 in _subtract_spans(b0, b1, afk):
            a, b = int(r0 * SR), int(min(len(audio16), r1 * SR))
            segs.extend(runner.transcribe(audio16[a:b], r0, WHISPER_PRIMARY))
        segs = [s for s in segs if s["start"] < b1 + 1.0]
        _write_part(vp, i, (b0, b1), segs, "done", PARAMS_ID)
        stats["transcribed"] += 1
        stats["segments"] += len(segs)
        stats["seconds"] += time.time() - t0
        log(f"   🎙️ جزء {n} من {len(todo)} ({sec_to_clock(b0)} إلى {sec_to_clock(b1)}): "
            f"{len(segs)} جملة خلال {time.time() - t0:.0f} ثانية")
    return stats


def _gather_segments(vp: VodPaths, blocks: list) -> list:
    segs = []
    for i in range(len(blocks)):
        part = _load_part(vp, i)
        if part:
            segs.extend(part.get("segments") or [])
    segs.sort(key=lambda s: (float(s.get("start", 0)), float(s.get("end", 0))))
    return segs


# ============================================================================
# Coverage audit and repair
# ============================================================================

def coverage_table(segments: list, speech_spans: list, afk: list, duration: float,
                   previous: Optional[list] = None) -> list:
    """Per 5 minute bucket: VAD speech seconds, words, and a status
    (ok, low, silent, afk, repaired, low_checked)."""
    prev = {(round(c.get("t0", -1)), round(c.get("t1", -1))): c.get("status") for c in previous or []}
    words_per_bucket: dict = {}
    for seg in segments:
        for w in seg.get("words") or []:
            k = int(float(w.get("s", 0)) // BUCKET_S)
            words_per_bucket[k] = words_per_bucket.get(k, 0) + 1
        if not seg.get("words"):
            k = int(float(seg.get("start", 0)) // BUCKET_S)
            words_per_bucket[k] = words_per_bucket.get(k, 0) + len(str(seg.get("text", "")).split())
    table = []
    n = int(math.ceil(duration / BUCKET_S)) if duration > 0 else 0
    for k in range(n):
        t0, t1 = k * BUCKET_S, min(duration, (k + 1) * BUCKET_S)
        speech = _speech_seconds(speech_spans, t0, t1)
        words = words_per_bucket.get(k, 0)
        if _in_spans(afk, t0, t1, 0.8):
            status = "afk"
        elif speech < 3.0:
            status = "silent"
        elif speech >= 30.0 and words / max(speech, 1e-6) < 0.5:
            status = "low"
        else:
            status = "ok"
        old = prev.get((round(t0), round(t1)))
        if status == "low" and old in ("repaired", "low_checked"):
            status = old
        elif status == "ok" and old == "repaired":
            status = "repaired"
        table.append({"t0": round(t0, 1), "t1": round(t1, 1), "speech_s": round(speech, 1),
                      "words": int(words), "status": status})
    return table


def _subtract_spans(t0: float, t1: float, spans: Sequence[Sequence[float]], min_len: float = 5.0) -> list:
    """[t0, t1] minus every span (AFK), keeping pieces of at least min_len seconds."""
    pieces = [[t0, t1]]
    for s, e in spans:
        nxt = []
        for a, b in pieces:
            if e <= a or s >= b:
                nxt.append([a, b])
                continue
            if s > a:
                nxt.append([a, s])
            if e < b:
                nxt.append([e, b])
        pieces = nxt
    return [p for p in pieces if p[1] - p[0] >= min_len]


def repair_low_buckets(vp: VodPaths, audio16: np.ndarray, blocks: list, table: list,
                       speech_spans: list, runner: WhisperRunner, hotwords: list,
                       afk: Sequence[Sequence[float]] = ()) -> int:
    """Second pass with a lower VAD threshold and no silence skip, only on buckets marked
    low (never on afk or silent buckets). A repair replaces a bucket only when it finds
    clearly more words. AFK spans inside a bucket are cut out before transcribing."""
    low = [c for c in table if c["status"] == "low"]
    if not low:
        return 0
    log(f"🩹 {len(low)} مقطع (5 دقائق) فيه كلام بس التفريغ ناقص، نعيد تفريغها بإعدادات أدق...")
    repaired = 0
    for c in low:
        t0, t1 = float(c["t0"]), float(c["t1"])
        try:
            new_segs = []
            for r0, r1 in _subtract_spans(t0, t1, afk):
                a, b = int(r0 * SR), int(min(len(audio16), r1 * SR))
                new_segs.extend(runner.transcribe(audio16[a:b], r0, WHISPER_REPAIR))
        except Exception as exc:
            log(f"   ⚠️ تعذر إصلاح {sec_to_clock(t0)}: {short_error(exc)}")
            c["status"] = "low_checked"
            continue
        new_segs, _ = filter_hallucinations(new_segs, speech_spans, hotwords)
        new_words = sum(len(s.get("words") or []) for s in new_segs)
        if new_words >= max(c["words"] * 1.3, c["words"] + 10):
            for s in new_segs:
                s["flags"] = sorted(set(s.get("flags") or []) | {"repaired"})
            for i, (b0, b1) in enumerate(blocks):
                if b1 <= t0 or b0 >= t1:
                    continue
                part = _load_part(vp, i) or {"segments": [], "status": "done", "params": PARAMS_ID}
                outside = [s for s in part["segments"] if not (t0 <= float(s.get("start", 0)) < t1)]
                inside_new = [s for s in new_segs if b0 <= float(s["start"]) < b1]
                merged = sorted(outside + inside_new, key=lambda s: float(s.get("start", 0)))
                _write_part(vp, i, (b0, b1), merged, part.get("status", "done"), part.get("params", PARAMS_ID))
            c["status"] = "repaired"
            c["words"] = new_words
            repaired += 1
            log(f"   ✅ {sec_to_clock(t0)}: من {c.get('words_before', 0)} إلى {new_words} كلمة")
        else:
            c["status"] = "low_checked"
    return repaired


def print_coverage(table: list, duration: float) -> None:
    if not table:
        return
    log("📋 تقرير التغطية (كل 30 دقيقة):")
    step = int(round(1800 / BUCKET_S))
    for i in range(0, len(table), step):
        group = table[i:i + step]
        t0, t1 = group[0]["t0"], group[-1]["t1"]
        speech = sum(c["speech_s"] for c in group) / 60.0
        words = sum(c["words"] for c in group)
        statuses = {c["status"] for c in group}
        if statuses <= {"afk", "silent"}:
            label = "😴 سكوت أو AFK"
        elif "low" in statuses or "low_checked" in statuses:
            label = "⚠️ فيه أجزاء تفريغها قليل"
        elif "repaired" in statuses:
            label = "🩹 تم إصلاح أجزاء"
        else:
            label = "✅ سليم"
        log(f"   {sec_to_clock(t0)} إلى {sec_to_clock(t1)}: كلام {speech:.1f} دقيقة | {words} كلمة | {label}")


# ============================================================================
# AST audio event tagger (GPU, optional, AFK and silence skipped)
# ============================================================================

def _active_seconds(feats: Features, afk: list, n1: int) -> np.ndarray:
    """Seconds worth tagging: not inside AFK, and either speech nearby or energy above floor."""
    active = np.zeros(n1, dtype=bool)
    per = int(round(1.0 / feats.hop10))
    total = n1 * per
    above = np.zeros(total, dtype=np.float32)
    sp = np.zeros(total, dtype=np.float32)
    m = min(total, len(feats.rms_db))
    above[:m] = feats.rms_db[:m] - feats.floor_db[:m]
    sp[:m] = feats.speech[:m]
    above1 = above.reshape(n1, per).max(axis=1)
    sp1 = sp.reshape(n1, per).max(axis=1)
    near_sp = np.convolve(sp1, np.ones(3, dtype=np.float32), mode="same") > 0
    active = near_sp | (above1 >= 4.0)
    for s, e in afk:
        active[int(max(0, s)):int(min(n1, e))] = False
    return active


def _calibrate(arr: np.ndarray, active: np.ndarray, target: float = 0.6, max_scale: float = 2.5) -> tuple:
    vals = arr[active]
    if len(vals) < 50:
        return arr, 1.0
    p = float(np.percentile(vals, 99.5))
    if p <= 0.1 or p >= target:
        return arr, 1.0
    scale = min(max_scale, target / p)
    return np.clip(arr * scale, 0.0, 1.0), scale


def tag_audio_events(audio16: np.ndarray, feats: Features, afk: list) -> bool:
    """Fill laugh / scream / shout / music / game_sfx (1 Hz) from the AST AudioSet model.
    Returns True when the tagger ran. Any failure keeps the DSP fallback events."""
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
    except Exception as exc:
        log(f"⚠️ مكتبات الـ AST غير متوفرة ({short_error(exc, 80)})، نكمل بالتقدير الصوتي البسيط")
        return False
    n1 = feats.n1
    active = _active_seconds(feats, afk, n1)
    n_active = int(active.sum())
    on_gpu = torch.cuda.is_available()
    if not on_gpu and n_active > 1800:
        log("⏭️ ما فيه كرت شاشة، تخطينا مصنف الأصوات (AST) لأنه بطيء على المعالج")
        return False
    if n_active == 0:
        log("⏭️ ما فيه ثواني نشطة للتصنيف الصوتي")
        return False
    skipped = n1 - n_active
    log(f"🏷️ تصنيف الضحك والصراخ بالموديل AST على {format_duration_ar(n_active)} نشطة "
        f"(تخطينا {format_duration_ar(skipped)} سكوت و AFK)...")
    t0 = time.time()
    try:
        extractor = ASTFeatureExtractor.from_pretrained(AST_MODEL_ID)
        model = ASTForAudioClassification.from_pretrained(AST_MODEL_ID)
        device = "cuda" if on_gpu else "cpu"
        model = model.to(device).eval()
        if on_gpu:
            model = model.half()
        label_to_idx = {str(v): int(k) for k, v in model.config.id2label.items()}
        groups = {name: [label_to_idx[l] for l in labels if l in label_to_idx] for name, labels in AST_LABELS.items()}
        out = {name: np.zeros(n1, dtype=np.float32) for name in AST_LABELS}
        seconds = np.where(active)[0]
        half = AST_WINDOW_S / 2.0
        win = int(AST_WINDOW_S * SR)
        for bstart in range(0, len(seconds), AST_BATCH):
            batch_secs = seconds[bstart:bstart + AST_BATCH]
            clips = []
            for t in batch_secs:
                a = int(max(0.0, (t + 0.5 - half)) * SR)
                clip = audio16[a:a + win]
                if len(clip) < win:
                    clip = np.pad(clip, (0, win - len(clip)))
                clips.append(clip)
            inputs = extractor(clips, sampling_rate=SR, return_tensors="pt")
            values = inputs["input_values"].to(device)
            if on_gpu:
                values = values.half()
            with torch.no_grad():
                probs = torch.sigmoid(model(input_values=values).logits.float()).cpu().numpy()
            for name, idxs in groups.items():
                if idxs:
                    out[name][batch_secs] = probs[:, idxs].max(axis=1)
            done = bstart + len(batch_secs)
            if done % (AST_BATCH * 100) < AST_BATCH:
                log(f"   ... صنفنا {done} من {len(seconds)} ثانية")
        scales = {}
        for name in ("laugh", "scream", "shout"):
            out[name], scales[name] = _calibrate(out[name], active)
        for name, arr in out.items():
            feats.set_event(name, arr)
        feats.tagger = "ast"
        del model
        if on_gpu:
            torch.cuda.empty_cache()
        gc.collect()
        scale_txt = "، ".join(f"{k} ×{v:.2f}" for k, v in scales.items() if v != 1.0)
        log(f"✅ التصنيف الصوتي اكتمل خلال {time.time() - t0:.0f} ثانية"
            + (f" (معايرة: {scale_txt})" if scale_txt else ""))
        return True
    except Exception as exc:
        log(f"⚠️ تعذر تشغيل مصنف الأصوات AST ({short_error(exc, 120)})، نكمل بالتقدير الصوتي البسيط")
        return False


# ============================================================================
# Hype timeline (compatibility file, fixed weighting)
# ============================================================================

def analyze_hype(vp: VodPaths, feats: Features, chat: list, window_sec: int = 10) -> list:
    """10 s windows: audio (normalized loudness), chat counts, and a hype score where chat only
    counts as much as the chat is healthy, so a quiet Twitch chat cannot swamp audio."""
    duration = feats.duration
    num_windows = max(1, int(math.ceil(duration / window_sec)))
    per = int(round(window_sec / feats.hop10))
    rms_lin = np.power(10.0, feats.rms_db.astype(np.float64) / 20.0)
    audio = np.zeros(num_windows, dtype=np.float64)
    for i in range(num_windows):
        chunk = rms_lin[i * per:(i + 1) * per]
        if len(chunk):
            audio[i] = float(np.sqrt(np.mean(chunk ** 2)))
    chat_count = np.zeros(num_windows, dtype=np.float64)
    chat_hype = np.zeros(num_windows, dtype=np.float64)
    for msg in chat or []:
        try:
            idx = int(float(msg["t"]) // window_sec)
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= idx < num_windows:
            chat_count[idx] += 1
            bonus = 1.0
            if HYPE_PATTERNS.search(str(msg.get("m", ""))):
                bonus += 1.5
            if msg.get("e"):
                bonus += 0.5
            chat_hype[idx] += bonus

    def norm(arr: np.ndarray) -> np.ndarray:
        mx = np.percentile(arr, 98) if len(arr) > 5 else (np.max(arr) if len(arr) else 0.0)
        return np.clip(arr / mx, 0, 1.5) if mx > 0 else np.zeros_like(arr)

    a_norm = norm(audio)
    c_norm = norm(chat_hype)
    h = chat_health(chat, duration)
    w_chat = 0.55 * h
    timeline = []
    for i in range(num_windows):
        score = w_chat * c_norm[i] + (1.0 - w_chat) * a_norm[i]
        timeline.append({
            "start": i * window_sec, "end": (i + 1) * window_sec,
            "audio": round(float(a_norm[i]), 3), "chat_msgs": int(chat_count[i]),
            "chat_score": round(float(c_norm[i]), 3), "hype": round(float(score), 3),
        })
    atomic_write_json(vp.hype, timeline, indent=None, backup=False)
    log(f"✅ رادار الحماس جاهز ({num_windows} نافذة، وزن الشات {w_chat:.2f})")
    return timeline


# ============================================================================
# Main entry
# ============================================================================

def _transcript_is_v2_complete(vp: VodPaths) -> bool:
    tr = load_json_safe(vp.transcript, default=None, expected_types=(dict,), quiet=True)
    return bool(tr) and tr.get("schema") == TRANSCRIPT_SCHEMA and isinstance(tr.get("coverage"), list) \
        and not any(c.get("status") == "low" for c in tr["coverage"])


def run_stage2(vod_url: str, drive_root: str, test_minutes: Any = 0,
               whisper_model: str = "large-v3-turbo", redo_transcript: bool = False,
               run_tagger: bool = True) -> str:
    t_start = time.time()
    test_minutes = _normalize_test_minutes(test_minutes)
    vod_id = extract_vod_id(vod_url)
    vp = VodPaths.from_ids(drive_root, vod_id).ensure()
    memory = MemoryPaths(drive_root).ensure()
    _prepare_checkpoint(vp, vod_url, test_minutes, whisper_model)
    report: dict = load_json_safe(vp.stage2_report, default={}, expected_types=(dict,), quiet=True) or {}
    timings: dict = {}

    # 1. Chat
    t0 = time.time()
    if _chat_is_complete(vp):
        chat = load_json_safe(vp.chat, default=[], expected_types=(list,), quiet=True) or []
        log(f"⏭️ الشات موجود مسبقاً ({len(chat)} رسالة)، تم التخطي")
    else:
        if os.path.isfile(vp.chat):
            log("ℹ️ ملف الشات قديم أو ناقص، نعيد سحبه بالطريقة الجديدة")
        chat = download_chat(vod_id, vp, test_minutes)
    timings["chat_s"] = round(time.time() - t0, 1)

    # 2. Audio
    t0 = time.time()
    if not os.path.isfile(vp.audio):
        download_audio(vod_url, vp, test_minutes)
    else:
        log("⏭️ الصوت موجود مسبقاً على Drive، تم التخطي")
    timings["audio_s"] = round(time.time() - t0, 1)

    # 3. Decode once (FFmpeg to NumPy)
    t0 = time.time()
    local_audio = os.path.join(ensure_dir(os.path.join(TMP_ROOT, "stage2")), f"{vod_id}.mp3")
    if not os.path.isfile(local_audio) or os.path.getsize(local_audio) != os.path.getsize(vp.audio):
        shutil.copyfile(vp.audio, local_audio)
    log("🔊 جاري فك الصوت إلى مصفوفة 16kHz...")
    audio16 = load_audio_ffmpeg(local_audio, sr=SR)
    duration = len(audio16) / float(SR)
    log(f"✅ مدة الصوت {duration / 60:.1f} دقيقة")
    timings["decode_s"] = round(time.time() - t0, 1)

    # 4. VAD and AFK
    t0 = time.time()
    speech_spans = load_or_run_vad(vp, audio16, duration)
    pre_feats = compute_dsp_features(audio16, SR, speech_spans, None)
    afk = afk_spans(pre_feats)
    if afk:
        afk_total = sum(e - s for s, e in afk)
        log(f"😴 فترات AFK أو سكوت طويل: {len(afk)} فترة مجموعها {format_duration_ar(afk_total)} (بنتخطاها)")
    timings["vad_s"] = round(time.time() - t0, 1)

    # 5. Transcription in checkpointed blocks (legacy transcripts are reused, not redone)
    t0 = time.time()
    hotwords = _load_hotwords(memory)
    runner = WhisperRunner(whisper_model, hotwords)
    blocks = load_or_make_plan(vp, duration, speech_spans)
    old_tr = load_json_safe(vp.transcript, default=None, expected_types=(dict,), quiet=True)
    previous_coverage = old_tr.get("coverage") if isinstance(old_tr, dict) else None

    if redo_transcript:
        log("♻️ طلبت إعادة التفريغ كامل، بنتجاهل الأجزاء القديمة")
    elif _transcript_is_v2_complete(vp) and all(_load_part(vp, i) is not None for i in range(len(blocks))):
        log("⏭️ التفريغ النصي جاهز مسبقاً بالصيغة الجديدة، تم التخطي")
    elif old_tr and old_tr.get("schema") != TRANSCRIPT_SCHEMA and old_tr.get("segments"):
        n = split_legacy_transcript(vp, old_tr, blocks)
        log(f"♻️ حولنا التفريغ القديم إلى {n} جزء محفوظ بدون إعادة تفريغ")

    tstats = {"transcribed": 0, "skipped_done": 0, "silent": 0, "segments": 0, "seconds": 0.0}
    try:
        tstats = transcribe_blocks(vp, audio16, blocks, speech_spans, afk, runner, redo=redo_transcript)
        segments = _gather_segments(vp, blocks)
        segments, dropped = filter_hallucinations(segments, speech_spans, hotwords)
        if dropped:
            log(f"🧹 حذفنا {dropped} سطر وهمي أو مكرر من التفريغ")

        table = coverage_table(segments, speech_spans, afk, duration, previous_coverage)
        for c in table:
            c["words_before"] = c["words"]
        repaired = repair_low_buckets(vp, audio16, blocks, table, speech_spans, runner, hotwords, afk)
        if repaired:
            segments = _gather_segments(vp, blocks)
            segments, _ = filter_hallucinations(segments, speech_spans, hotwords)
            fresh = coverage_table(segments, speech_spans, afk, duration, table)
            status_by_t0 = {c["t0"]: c["status"] for c in table}
            for c in fresh:
                if status_by_t0.get(c["t0"]) in ("repaired", "low_checked"):
                    c["status"] = status_by_t0[c["t0"]]
            table = fresh
        for c in table:
            c.pop("words_before", None)
    finally:
        runner.close()

    transcript = {
        "schema": TRANSCRIPT_SCHEMA,
        "duration": round(duration, 2),
        "model": whisper_model,
        "params_id": PARAMS_ID,
        "segments": segments,
        "coverage": table,
        "afk": afk,
        "updated_at": now_iso(),
    }
    atomic_write_json(vp.transcript, transcript, indent=None, backup=True)
    timings["transcribe_s"] = round(time.time() - t0, 1)
    print_coverage(table, duration)

    # 6. Features (DSP with words) and AST tagger
    t0 = time.time()
    words = collect_words(transcript)
    feats = compute_dsp_features(audio16, SR, speech_spans, words)
    old_feats_tagger = None
    if os.path.isfile(vp.features):
        try:
            with np.load(vp.features, allow_pickle=False) as z:
                old_feats_tagger = str(z["tagger"]) if "tagger" in z.files else None
                if old_feats_tagger == "ast" and not redo_transcript:
                    for name in ("laugh", "scream", "shout", "music", "game_sfx"):
                        key = f"ev_{name}"
                        if key in z.files:
                            feats.set_event(name, z[key].astype(np.float32))
                    feats.tagger = "ast"
        except Exception:
            old_feats_tagger = None
    if feats.tagger == "ast":
        log("⏭️ تصنيف الضحك والصراخ (AST) موجود مسبقاً، تم التخطي")
    elif run_tagger:
        tag_audio_events(audio16, feats, afk)
    save_features(vp.features, feats)
    timings["features_s"] = round(time.time() - t0, 1)

    # 7. Hype timeline (compatibility)
    analyze_hype(vp, feats, chat)

    # 8. Report
    chat_state = load_json_safe(vp.chat_state, default={}, expected_types=(dict,), quiet=True) or {}
    n_words = len(words)
    report.update({
        "vod_id": vod_id,
        "duration_sec": round(duration, 2),
        "test_minutes": test_minutes,
        "whisper_model": whisper_model,
        "chat": {"count": len(chat), "per_hour": round(len(chat) / max(duration / 3600.0, 1e-6), 1),
                 "health": round(chat_health(chat, duration), 3), "stop_reason": chat_state.get("stop_reason")},
        "speech_min": round(sum(e - s for s, e in speech_spans) / 60.0, 1),
        "afk": afk,
        "segments": len(segments),
        "words": n_words,
        "blocks": {"total": len(blocks), "transcribed_now": tstats["transcribed"],
                   "reused": tstats["skipped_done"], "silent_now": tstats["silent"]},
        "coverage_counts": {k: sum(1 for c in table if c["status"] == k) for k in {c["status"] for c in table}},
        "tagger": feats.tagger,
        "timings": timings,
        "updated_at": now_iso(),
    })
    atomic_write_json(vp.stage2_report, report)

    del audio16
    gc.collect()
    try:
        os.remove(local_audio)
    except OSError:
        pass

    log(f"\n🎉 المرحلة 2 جاهزة! مدة البث {duration / 60:.1f} دقيقة | {len(segments)} جملة | {n_words} كلمة "
        f"| الشات {len(chat)} رسالة | التصنيف الصوتي: {'AST' if feats.tagger == 'ast' else 'تقديري'}")
    log(f"⏱️ الوقت الكلي: {format_duration_ar(time.time() - t_start)}")
    log(f"📁 المسار: {vp.vod_dir}")
    return vp.vod_dir


# === END OF stage2_ingest.py ===
