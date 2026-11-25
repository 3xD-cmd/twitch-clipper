"""
media_engine.py
Audio features, phrase index, Cut Doctor, multi input cutter, dynamic SRT engine,
critic audio snippets, A/V offset guard, stream sets, window media for the Critic and
the Studio, Gemini transcodes and a self test for the TwitchClipper pipeline. Version 3.

Main contracts
  Features / compute_dsp_features / save_features / load_features / afk_spans
  collect_words(transcript) -> list[{"w","s","e","p"}]
  build_phrases(transcript_or_words, feats) -> list[phrase]       (ids are 0..N-1, sorted by time)
  ids_to_pieces(ranges, phrases) -> [[s, e], ...]
  cut_doctor(pieces, words, feats, vod_duration, fps, peak_time, prefs, auto_shave, protect) -> (pieces, report)
  trim_pieces / drop_pieces / merge_pieces / snap_to_frames / pieces_net
  StreamInfo / resolve_stream(vod_url) / pick_encoder / probe_media
  StreamSet / resolve_streams(vod_url, max_height, proxy_heights) -> StreamSet (master + proxy, shared refresh)
  cut_and_stitch(stream, pieces, out_mp4, ...) -> dict
  render_pieces(streams, pieces, out_mp4, out_srt, words, ...) -> dict      (cut + SRT in one call)
  make_window_media(stream_or_set, t0, t1, critic_out, preview_out, ...) -> dict
  transcode_for_gemini(src, out, fps, short_side, audio_kbps, max_seconds) -> dict
  waveform_peaks(feats, t0, t1, n) -> list[float]
  map_words_to_timeline / generate_srt / parse_srt / validate_srt
  audio_snippet(audio_path, start, end) -> {"data": bytes, "mime_type": "audio/mp3", "seconds": float}
  measure_av_offset(stream, audio_path, duration) -> float | None
  selftest() -> dict

Rules kept from the battle tested version
  * Video is cut locally in /tmp from direct stream URLs (yt-dlp -g), never with --download-sections.
  * h264_nvenc when it really works, automatic fallback to libx264.
  * Expired stream URLs are resolved again automatically on failure (and proactively after 3 hours).
  * No burned in text and no music: the graph has no subtitle or mixing filters, and the output
    is rejected unless it holds exactly one video and one audio stream.
  * Resolution is kept native. Only sources taller than 1080 are scaled down; nothing is upscaled.

v3 additions
  * Protected visual beats: Cut Doctor never snaps a start past a protected start (or an end
    before a protected end), never shaves inside a protected interval, keeps pieces that touch
    one, and extends a piece back when a beat ends within 4 s before it.
  * StreamSet: one yt-dlp JSON call resolves the master rendition (final renders) and a low
    proxy rendition (Critic windows and Studio previews). Both refresh together.
  * Window media: one FFmpeg pass writes the small low fps Critic video and the Studio preview,
    straight from the stream URL (no full download), proxy first and master as fallback.
  * transcode_for_gemini for Golden Clips, waveform_peaks for the Studio timeline.
"""

from __future__ import annotations

import bisect
import io
import json
import math
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

from common import (
    TMP_ROOT,
    VodPaths,
    arabic_char_count,
    atomic_write_bytes,
    atomic_write_text,
    clamp,
    ensure_dir,
    log,
    merge_spans,
    normalize_arabic,
    parse_time_to_sec,
    run_command,
    safe_copy_file,
    sec_to_srt_time,
    short_error,
)


# ============================================================================
# Constants
# ============================================================================

HOP10 = 0.1
HOP1 = 1.0
FEATURES_VERSION = 1
EVENT_NAMES = ("laugh", "scream", "shout", "speech_p", "music", "game_sfx")

DOCTOR_DEFAULTS = {
    "start_pad_s": 0.12,
    "end_pad_s": 0.30,
    "min_prev_gap_s": 0.04,
    "end_lookahead_s": 6.0,
    "laugh_thr": 0.35,
    "reaction_energy_db": 8.0,
    "quiet_stop_s": 0.4,
    "tail_s": 0.25,
    "seam_gap_s": 0.04,
    "seam_laugh_thr": 0.5,
    "seam_laugh_max_end_s": 4.0,
    "seam_laugh_max_start_s": 3.0,
    "shave_min_gap_s": 0.8,
    "shave_keep_after_s": 0.15,
    "shave_keep_before_s": 0.12,
    "shave_min_cut_s": 0.5,
    "shave_floor_db": 6.0,
    "shave_quiet_frac": 0.85,
    "shave_event_thr": 0.25,
    "shave_post_reaction_s": 1.0,
    "shave_max_speech_frac": 0.10,
    "max_shave_ratio": 0.30,
    "min_piece_s": 0.6,
    "merge_gap_s": 0.25,
    "target_s": (12.0, 35.0),
    "soft_max_s": 45.0,
    "hard_max_s": 60.0,
    "min_net_s": 8.0,
    "protect_lookback_s": 4.0,
    "protect_shave_pad_s": 0.3,
}

SRT_BANDS = {
    "fast":   {"max_words": 2, "max_chars": 14, "pause": 0.22, "min_dur": 0.40},
    "normal": {"max_words": 4, "max_chars": 24, "pause": 0.35, "min_dur": 0.70},
    "calm":   {"max_words": 7, "max_chars": 38, "pause": 0.55, "min_dur": 1.00},
}
SRT_FAST_CPS = 15.0
SRT_CALM_CPS = 9.0
SRT_BRIDGE_S = 0.40
SRT_MIN_CUE_S = 0.10

_GLUE_RAW = ("و", "ف", "ب", "ل", "ك", "في", "من", "على", "عن", "الى", "إلى", "مع", "يا", "اللي",
             "الي", "ان", "إن", "أن", "لا", "ما", "مو", "هذا", "هذي", "ذا", "ذي", "لين", "عشان")
GLUE_WORDS = frozenset(normalize_arabic(w) for w in _GLUE_RAW)
_STRONG_END = ("؟", "?", "!", ".", "…")
_SOFT_END = ("،", ",", "؛", ";")
_LAUGH_RE = re.compile(r"ه{7,}")
_PUNCT_STRIP = "؟?!.،,؛;:…\"'«»()[]{}"

STREAM_MAX_AGE_S = 3 * 3600
RW_TIMEOUT_US = "30000000"

PROXY_HEIGHTS_DEFAULT = (360, 480, 240, 160)
WINDOW_PREVIEW_CRF = 30
WINDOW_CRITIC_CRF = 32
WINDOW_PREVIEW_GOP_S = 1.0


# ============================================================================
# Small helpers
# ============================================================================

def merge_pieces(pieces: Sequence[Sequence[float]], gap: float = 0.25) -> list:
    """Sort, drop invalid pieces, merge pieces that overlap or sit closer than gap seconds."""
    clean = []
    for p in pieces or []:
        try:
            s, e = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(s) and math.isfinite(e) and e > s:
            clean.append([s, e])
    clean.sort(key=lambda x: x[0])
    merged: list = []
    for s, e in clean:
        if merged and s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def pieces_net(pieces: Sequence[Sequence[float]]) -> float:
    return float(sum(max(0.0, float(e) - float(s)) for s, e in pieces or []))


def snap_to_frames(pieces: Sequence[Sequence[float]], fps: Optional[float]) -> list:
    """Round every boundary to the video frame grid so subtitle offsets equal real cut points."""
    if not fps or fps <= 0:
        return [[round(float(s), 3), round(float(e), 3)] for s, e in pieces]
    frame = 1.0 / fps
    out = []
    for s, e in pieces:
        s2 = round(float(s) * fps) / fps
        e2 = round(float(e) * fps) / fps
        if e2 - s2 < 2 * frame:
            e2 = s2 + 2 * frame
        if out and s2 < out[-1][1]:
            s2 = out[-1][1]
            if e2 - s2 < 2 * frame:
                continue
        out.append([round(s2, 6), round(e2, 6)])
    return out


def _proc_hwm_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status", "r") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


@dataclass
class _RunResult:
    returncode: int
    stderr: str
    timed_out: bool
    peak_mb: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def last_line(self) -> str:
        if self.timed_out:
            return "انتهت المهلة (timeout)"
        text = (self.stderr or "").strip()
        return text.splitlines()[-1][:300] if text else f"exit code {self.returncode}"


def _run_measured(cmd: Sequence[str], timeout: Optional[float]) -> _RunResult:
    """Run a command, track peak memory (VmHWM) and enforce a timeout. stderr goes to a
    temp file so a chatty process can never block on a full pipe."""
    with tempfile.TemporaryFile() as err_f:
        try:
            proc = subprocess.Popen(list(cmd), stdout=subprocess.DEVNULL, stderr=err_f, stdin=subprocess.DEVNULL)
        except FileNotFoundError:
            return _RunResult(127, f"الأداة غير موجودة: {cmd[0]}", False, 0.0)
        peak_kb = 0
        t0 = time.time()
        timed_out = False
        while True:
            rc = proc.poll()
            peak_kb = max(peak_kb, _proc_hwm_kb(proc.pid))
            if rc is not None:
                break
            if timeout and time.time() - t0 > timeout:
                proc.kill()
                proc.wait()
                timed_out = True
                break
            time.sleep(0.2)
        err_f.seek(0)
        stderr = err_f.read().decode("utf-8", "replace")[-6000:]
    return _RunResult(proc.returncode if proc.returncode is not None else -1, stderr, timed_out, peak_kb / 1024.0)


def decode_audio(path: str, sr: int = 16000, start: Optional[float] = None,
                 duration: Optional[float] = None, as_int16: bool = False,
                 timeout: Optional[float] = None) -> np.ndarray:
    """Decode any audio/video input to mono PCM with FFmpeg (never PyAV)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.3f}"]
    if duration is not None:
        cmd += ["-t", f"{max(0.05, duration):.3f}"]
    cmd += ["-i", path, "-vn", "-ac", "1", "-ar", str(int(sr)), "-f", "s16le", "-"]
    res = run_command(cmd, timeout=timeout, text=False)
    if not res.ok or not res.stdout:
        raise RuntimeError(f"تعذر فك الصوت ({res.last_error_line()})")
    pcm = np.frombuffer(res.stdout, dtype=np.int16)
    if as_int16:
        return pcm.copy()
    return pcm.astype(np.float32) / 32768.0


# ============================================================================
# Audio features
# ============================================================================

@dataclass
class Features:
    rms_db: np.ndarray
    floor_db: np.ndarray
    speech: np.ndarray
    events: dict = field(default_factory=dict)
    tagger: str = "dsp"
    hop10: float = HOP10
    hop1: float = HOP1

    @property
    def duration(self) -> float:
        return float(len(self.rms_db)) * self.hop10

    @property
    def n1(self) -> int:
        return int(math.ceil(self.duration / self.hop1)) if len(self.rms_db) else 0

    def i10(self, t: float) -> int:
        if not len(self.rms_db):
            return 0
        return int(clamp(math.floor(float(t) / self.hop10), 0, len(self.rms_db) - 1))

    def range10(self, t0: float, t1: float) -> tuple:
        n = len(self.rms_db)
        a = int(clamp(math.floor(float(t0) / self.hop10), 0, max(0, n - 1)))
        b = int(clamp(math.ceil(float(t1) / self.hop10), a + 1, n))
        return a, b

    def above_floor(self, t0: float, t1: float) -> np.ndarray:
        a, b = self.range10(t0, t1)
        return (self.rms_db[a:b].astype(np.float32) - self.floor_db[a:b].astype(np.float32))

    def above_at(self, t: float) -> float:
        i = self.i10(t)
        return float(self.rms_db[i]) - float(self.floor_db[i]) if len(self.rms_db) else 0.0

    def speech_at(self, t: float) -> bool:
        return bool(self.speech[self.i10(t)]) if len(self.speech) else False

    def speech_frac(self, t0: float, t1: float) -> float:
        a, b = self.range10(t0, t1)
        seg = self.speech[a:b]
        return float(seg.mean()) if len(seg) else 0.0

    def event(self, name: str) -> Optional[np.ndarray]:
        arr = self.events.get(name)
        return arr if isinstance(arr, np.ndarray) and len(arr) else None

    def event_at(self, name: str, t: float) -> float:
        arr = self.event(name)
        if arr is None:
            return 0.0
        idx = int(clamp(math.floor(float(t) / self.hop1), 0, len(arr) - 1))
        return float(arr[idx])

    def event_max(self, name: str, t0: float, t1: float) -> float:
        arr = self.event(name)
        if arr is None:
            return 0.0
        a = int(clamp(math.floor(float(t0) / self.hop1), 0, len(arr) - 1))
        b = int(clamp(math.ceil(float(t1) / self.hop1), a + 1, len(arr)))
        return float(arr[a:b].max())

    def set_event(self, name: str, values: np.ndarray) -> None:
        """Install a 1 Hz event track (for example from the AST tagger), resized to n1."""
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        n = self.n1
        if len(arr) < n:
            arr = np.concatenate([arr, np.zeros(n - len(arr), dtype=np.float32)])
        self.events[name] = np.clip(arr[:n], 0.0, 1.0)


def _frame_rms_db(samples: np.ndarray, sr: int, hop: float) -> np.ndarray:
    n = max(1, int(round(sr * hop)))
    frames = len(samples) // n
    out = np.empty(frames, dtype=np.float32)
    is_int = np.issubdtype(samples.dtype, np.integer)
    step = 100_000
    for a in range(0, frames, step):
        b = min(frames, a + step)
        blk = samples[a * n:b * n].astype(np.float32).reshape(b - a, n)
        if is_int:
            blk /= 32768.0
        out[a:b] = np.sqrt(np.mean(blk * blk, axis=1))
    return (20.0 * np.log10(out + 1e-5)).astype(np.float32)


def _rolling_floor_db(rms_db: np.ndarray, block: int = 100) -> np.ndarray:
    """Background level: 10th percentile per 10 s block, median over neighbouring blocks."""
    n = len(rms_db)
    if n == 0:
        return rms_db.copy()
    nb = int(math.ceil(n / block))
    p10 = np.empty(nb, dtype=np.float32)
    for i in range(nb):
        p10[i] = np.percentile(rms_db[i * block:(i + 1) * block], 10)
    smooth = np.empty(nb, dtype=np.float32)
    for i in range(nb):
        smooth[i] = np.median(p10[max(0, i - 1):i + 2])
    return np.repeat(smooth, block)[:n].astype(np.float32)


def _mask_from_spans(spans: Sequence[Sequence[float]], n: int, hop: float, pad: float = 0.0) -> np.ndarray:
    mask = np.zeros(n, dtype=np.uint8)
    for sp in spans or []:
        try:
            s, e = float(sp[0]), float(sp[1])
        except (TypeError, ValueError, IndexError, KeyError):
            continue
        a = int(clamp(math.floor((s - pad) / hop), 0, n))
        b = int(clamp(math.ceil((e + pad) / hop), 0, n))
        if b > a:
            mask[a:b] = 1
    return mask


def _per_second_mean(arr10: np.ndarray, n1: int, per: int = 10) -> np.ndarray:
    total = n1 * per
    buf = np.zeros(total, dtype=np.float32)
    m = min(total, len(arr10))
    buf[:m] = arr10[:m]
    return buf.reshape(n1, per).mean(axis=1)


def compute_dsp_features(
    samples: np.ndarray,
    sr: int,
    speech_segments: Optional[Sequence[Sequence[float]]] = None,
    words: Optional[Sequence[dict]] = None,
) -> Features:
    """Loudness track (10 Hz), background floor, speech mask, and heuristic 1 Hz events.
    speech_segments: VAD spans in seconds (Stage 2). Without them the mask comes from words.
    Heuristic events are a fallback; Stage 2 replaces them with AST tagger output."""
    rms_db = _frame_rms_db(samples, sr, HOP10)
    floor_db = _rolling_floor_db(rms_db)
    n10 = len(rms_db)
    word_spans = [(w["s"], w["e"]) for w in words or []]
    cover = _mask_from_spans(word_spans, n10, HOP10, pad=0.05)
    if speech_segments:
        speech = _mask_from_spans(speech_segments, n10, HOP10)
    else:
        speech = _mask_from_spans(word_spans, n10, HOP10, pad=0.15)

    feats = Features(rms_db=rms_db, floor_db=floor_db, speech=speech, events={}, tagger="dsp")
    n1 = feats.n1
    above = rms_db - floor_db
    idx = np.arange(n10)
    last_cover = np.maximum.accumulate(np.where(cover > 0, idx, -10_000))
    recent_speech = (idx - last_cover) * HOP10 <= 2.5
    no_words = cover == 0

    laugh10 = ((above >= 12.0) & no_words & recent_speech).astype(np.float32)
    scream10 = ((above >= 18.0) & (cover > 0)).astype(np.float32)
    game10 = ((above >= 15.0) & no_words & ~recent_speech).astype(np.float32)

    feats.events = {
        "laugh": np.clip(_per_second_mean(laugh10, n1) * 0.8, 0.0, 1.0),
        "scream": np.clip(_per_second_mean(scream10, n1) * 0.7, 0.0, 1.0),
        "shout": np.clip(_per_second_mean(scream10, n1) * 0.7, 0.0, 1.0),
        "speech_p": _per_second_mean(speech.astype(np.float32), n1),
        "music": np.zeros(n1, dtype=np.float32),
        "game_sfx": np.clip(_per_second_mean(game10, n1) * 0.8, 0.0, 1.0),
    }
    return feats


def save_features(path: str, feats: Features) -> str:
    buf = io.BytesIO()
    payload = {
        "version": np.array(FEATURES_VERSION, dtype=np.int32),
        "hop10": np.array(feats.hop10, dtype=np.float32),
        "hop1": np.array(feats.hop1, dtype=np.float32),
        "rms_db": feats.rms_db.astype(np.float16),
        "floor_db": feats.floor_db.astype(np.float16),
        "speech": feats.speech.astype(np.uint8),
        "tagger": np.array(str(feats.tagger)),
    }
    for name in EVENT_NAMES:
        arr = feats.event(name)
        payload[f"ev_{name}"] = (arr if arr is not None else np.zeros(feats.n1, dtype=np.float32)).astype(np.float16)
    np.savez_compressed(buf, **payload)
    return atomic_write_bytes(path, buf.getvalue())


def _load_npz(path: str) -> Optional[Features]:
    try:
        with np.load(path, allow_pickle=False) as z:
            rms = z["rms_db"].astype(np.float32)
            floor = z["floor_db"].astype(np.float32) if "floor_db" in z.files else _rolling_floor_db(rms)
            speech = z["speech"].astype(np.uint8) if "speech" in z.files else np.zeros(len(rms), np.uint8)
            hop10 = round(float(z["hop10"]), 6) if "hop10" in z.files else HOP10
            hop1 = round(float(z["hop1"]), 6) if "hop1" in z.files else HOP1
            tagger = str(z["tagger"]) if "tagger" in z.files else "dsp"
            events = {}
            for name in EVENT_NAMES:
                key = f"ev_{name}"
                if key in z.files:
                    events[name] = z[key].astype(np.float32)
        feats = Features(rms_db=rms, floor_db=floor, speech=speech, events=events, tagger=tagger, hop10=hop10, hop1=hop1)
        for name in EVENT_NAMES:
            if name not in feats.events:
                feats.events[name] = np.zeros(feats.n1, dtype=np.float32)
        return feats
    except Exception as exc:
        log(f"⚠️ ملف الخصائص الصوتية تالف ({short_error(exc, 80)})، بنعيد بناءه")
        return None


def load_features(vod_paths: VodPaths, words: Optional[Sequence[dict]] = None,
                  allow_build: bool = True, save_built: bool = True) -> Optional[Features]:
    """Load audio_features.npz. Older VODs without it get a loudness only version built from
    audio.mp3 (and saved to Drive so it is built once)."""
    if os.path.isfile(vod_paths.features):
        feats = _load_npz(vod_paths.features)
        if feats is not None:
            return feats
    if not allow_build:
        return None
    if not os.path.isfile(vod_paths.audio):
        log("⚠️ ما فيه ملف صوت للبث، نكمل بدون خصائص صوتية")
        return None
    log("📊 بناء الخصائص الصوتية من ملف الصوت (مرة وحدة لهذا البث)...")
    t0 = time.time()
    samples = decode_audio(vod_paths.audio, sr=8000, as_int16=True)
    feats = compute_dsp_features(samples, 8000, None, words)
    del samples
    if save_built:
        try:
            save_features(vod_paths.features, feats)
        except Exception as exc:
            log(f"⚠️ تعذر حفظ الخصائص الصوتية في Drive: {short_error(exc)}")
    log(f"✅ الخصائص الصوتية جاهزة ({feats.duration / 60:.1f} دقيقة) خلال {time.time() - t0:.0f} ثانية")
    return feats


def afk_spans(feats: Features, min_len_s: float = 180.0, max_speech_frac: float = 0.02,
              window_s: float = 30.0) -> list:
    """Long stretches with almost no speech (AFK, BRB screen, stream left running).
    Stage 2 skips them in the Whisper repair pass and in the AST tagger."""
    if feats is None or not len(feats.speech):
        return []
    n1 = feats.n1
    sp = _per_second_mean(feats.speech.astype(np.float32), n1)
    w = max(1, int(window_s))
    kernel = np.ones(w, dtype=np.float32) / w
    local = np.convolve(sp, kernel, mode="same")
    quiet = local <= max_speech_frac
    spans = []
    start = None
    for i, q in enumerate(quiet):
        if q and start is None:
            start = i
        elif not q and start is not None:
            if i - start >= min_len_s:
                spans.append([float(start), float(i)])
            start = None
    if start is not None and n1 - start >= min_len_s:
        spans.append([float(start), float(n1)])
    return spans


# ============================================================================
# Words and phrases
# ============================================================================

def collect_words(transcript: dict) -> list:
    """All words sorted by start time. Segments without word timings are split evenly."""
    words: list = []
    for seg in (transcript or {}).get("segments", []) or []:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                txt = str(w.get("w", "")).strip()
                s = parse_time_to_sec(w.get("s"))
                e = parse_time_to_sec(w.get("e"))
                if not txt or s is None:
                    continue
                if e is None or e <= s:
                    e = s + 0.05
                words.append({"w": txt, "s": float(s), "e": float(e), "p": w.get("p")})
        else:
            tokens = str(seg.get("text", "")).split()
            s0 = parse_time_to_sec(seg.get("start"))
            e0 = parse_time_to_sec(seg.get("end"))
            if not tokens or s0 is None or e0 is None or e0 <= s0:
                continue
            step = (e0 - s0) / len(tokens)
            for i, tok in enumerate(tokens):
                words.append({"w": tok, "s": s0 + i * step, "e": s0 + (i + 1) * step, "p": None})
    words.sort(key=lambda w: (w["s"], w["e"]))
    return words


def _ends_sentence(token: str) -> bool:
    return bool(token) and token.rstrip().endswith(_STRONG_END)


def _gap_label(feats: Optional[Features], t0: float, t1: float) -> str:
    if feats is None:
        return "silence" if t1 - t0 >= 1.2 else "other"
    if feats.event_max("laugh", t0, t1) >= 0.35:
        return "laugh"
    scream = max(feats.event_max("scream", t0, t1), feats.event_max("shout", t0, t1))
    if scream >= 0.35 and feats.speech_frac(t0, t1) > 0.3:
        return "scream"
    above = feats.above_floor(t0, t1)
    sp = feats.speech_frac(t0, t1)
    if len(above) and float(above.mean()) >= 12.0 and sp < 0.2:
        return "game_loud"
    if len(above) and float((above <= 6.0).mean()) >= 0.85 and sp < 0.1:
        return "silence"
    return "other"


_EVENT_MIN_LEN = {"laugh": 0.6, "scream": 0.6, "game_loud": 0.8, "silence": 1.2}


def _event_text(kind: str, dur: float) -> str:
    if kind == "silence" and dur >= 60:
        return f"(صمت طويل {dur / 60:.0f} دقيقة)"
    label = {"laugh": "ضحك", "scream": "صراخ", "game_loud": "صوت لعبة عالي", "silence": "صمت"}[kind]
    return f"({label} {dur:.1f}ث)"


def build_phrases(transcript_or_words: Any, feats: Optional[Features] = None,
                  gap_split_s: float = 0.35, max_phrase_s: float = 4.0, max_words: int = 10,
                  vod_duration: Optional[float] = None) -> list:
    """Phrase index for prompts. Speech phrases come from word timings; gaps between them
    become event lines (laugh, scream, game_loud, silence) from the audio features.
    ids run 0..N-1 in time order; every line has s, e, kind, text."""
    words = collect_words(transcript_or_words) if isinstance(transcript_or_words, dict) else list(transcript_or_words or [])
    total = float(vod_duration or (feats.duration if feats is not None else (words[-1]["e"] if words else 0.0)))
    speech: list = []
    cur: list = []

    def flush() -> None:
        if not cur:
            return
        first, last = words[cur[0]], words[cur[-1]]
        text = " ".join(words[i]["w"] for i in cur).strip()
        ph = {"s": round(first["s"], 3), "e": round(last["e"], 3), "kind": "speech", "text": text,
              "w0": cur[0], "w1": cur[-1]}
        if feats is not None:
            ph["laugh"] = round(feats.event_max("laugh", first["s"], last["e"]), 3)
            ph["scream"] = round(max(feats.event_max("scream", first["s"], last["e"]),
                                     feats.event_max("shout", first["s"], last["e"])), 3)
            above = feats.above_floor(first["s"], last["e"])
            ph["loud"] = round(float(above.max()) if len(above) else 0.0, 2)
            tags = []
            if ph["laugh"] >= 0.5:
                tags.append("ضحك")
            if ph["scream"] >= 0.5:
                tags.append("صراخ")
            ph["tags"] = tags
        else:
            ph["tags"] = []
        speech.append(ph)
        cur.clear()

    for i, w in enumerate(words):
        if cur:
            prev = words[cur[-1]]
            gap = w["s"] - prev["e"]
            dur = w["e"] - words[cur[0]]["s"]
            if gap >= gap_split_s or dur > max_phrase_s or len(cur) >= max_words or _ends_sentence(prev["w"]):
                flush()
        cur.append(i)
    flush()

    events: list = []
    bounds = [0.0] + [x for ph in speech for x in (ph["s"], ph["e"])] + [max(total, speech[-1]["e"] if speech else 0.0)]
    for k in range(0, len(bounds) - 1, 2):
        g0, g1 = bounds[k], bounds[k + 1]
        if g1 - g0 < 0.6:
            continue
        cells = []
        t = g0
        while t < g1 - 1e-6:
            t_end = min(g1, t + 0.5)
            cells.append([t, t_end, _gap_label(feats, t, t_end)])
            t = t_end
        runs: list = []
        for c in cells:
            if runs and runs[-1][2] == c[2]:
                runs[-1][1] = c[1]
            else:
                runs.append(list(c))
        for s, e, kind in runs:
            if kind in _EVENT_MIN_LEN and e - s >= _EVENT_MIN_LEN[kind]:
                events.append({"s": round(s, 3), "e": round(e, 3), "kind": kind,
                               "text": _event_text(kind, e - s), "w0": None, "w1": None, "tags": []})

    phrases = sorted(speech + events, key=lambda p: (p["s"], 0 if p["kind"] == "speech" else 1))
    for i, ph in enumerate(phrases):
        ph["id"] = i
    return phrases


def _range_ids(r: Any) -> Optional[tuple]:
    if isinstance(r, dict):
        a, b = r.get("from_id"), r.get("to_id")
    elif isinstance(r, (list, tuple)) and len(r) >= 2:
        a, b = r[0], r[1]
    else:
        a, b = getattr(r, "from_id", None), getattr(r, "to_id", None)
    try:
        a, b = int(a), int(b)
    except (TypeError, ValueError):
        return None
    return (a, b) if a <= b else (b, a)


def ids_to_pieces(ranges: Sequence[Any], phrases: Sequence[dict], merge_gap: float = 0.25) -> list:
    """Convert phrase id ranges into [[start, end], ...] seconds (sorted, merged)."""
    by_id = {int(p["id"]): p for p in phrases}
    pieces = []
    for r in ranges or []:
        ids = _range_ids(r)
        if ids is None:
            continue
        pa, pb = by_id.get(ids[0]), by_id.get(ids[1])
        if pa is None or pb is None:
            continue
        pieces.append([float(pa["s"]), float(pb["e"])])
    return merge_pieces(pieces, merge_gap)


# ============================================================================
# Cut Doctor
# ============================================================================

def _word_starts(words: Sequence[dict]) -> list:
    return [w["s"] for w in words]


def _word_at(words: Sequence[dict], starts: list, t: float) -> Optional[int]:
    """Index of the word that contains time t (s < t < e), if any."""
    j = bisect.bisect_right(starts, t) - 1
    for k in (j, j - 1):
        if 0 <= k < len(words) and words[k]["s"] < t < words[k]["e"]:
            return k
    return None


def _first_word_in(words, starts, s, e) -> Optional[int]:
    j = max(0, bisect.bisect_left(starts, s - 2.0))
    while j < len(words) and words[j]["s"] < e:
        mid = (words[j]["s"] + words[j]["e"]) / 2.0
        if s - 0.05 <= mid < e:
            return j
        j += 1
    return None


def _last_word_in(words, starts, s, e) -> Optional[int]:
    j = min(len(words) - 1, bisect.bisect_right(starts, e))
    while j >= 0:
        w = words[j]
        mid = (w["s"] + w["e"]) / 2.0
        if mid <= e + 0.05 and mid > s:
            return j
        if w["e"] < s:
            break
        j -= 1
    return None


def _has_reaction(feats: Optional[Features], t0: float, t1: float, thr: float) -> bool:
    if feats is None or t1 <= t0:
        return False
    return (feats.event_max("laugh", t0, t1) >= thr
            or feats.event_max("scream", t0, t1) >= thr
            or feats.event_max("shout", t0, t1) >= thr)


def _clean_protect(protect: Optional[Sequence[Any]], vod_duration: float) -> list:
    """Valid protected intervals [a, b] inside the VOD, sorted and merged."""
    if not protect:
        return []
    out = []
    for a, b in merge_spans(protect, gap=0.0):
        a, b = max(0.0, a), min(float(vod_duration), b)
        if b > a:
            out.append([a, b])
    return out


def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and a1 > b0


def _apply_protect_coverage(pcs: list, prot: list, P) -> list:
    """Make every protected beat that touches a piece fully covered, and pull a piece back
    over a beat that ends at most protect_lookback_s before it (the silent visual setup)."""
    out = [list(p) for p in pcs]
    for a, b in prot:
        if any(_overlaps(p[0], p[1], a, b) for p in out):
            out.append([a, b])
            continue
        after = [p for p in out if 0.0 <= p[0] - b <= P["protect_lookback_s"]]
        if after:
            target = min(after, key=lambda p: p[0])
            target[0] = min(target[0], a)
    return merge_pieces(out, P["merge_gap_s"])


def _snap_piece(s: float, e: float, words, starts, feats, P, protect: Optional[list] = None) -> tuple:
    new_s, new_e = s, e
    j = _first_word_in(words, starts, s, e)
    if j is not None:
        w = words[j]
        cand = w["s"] - P["start_pad_s"]
        if j > 0 and words[j - 1]["e"] < w["s"]:
            cand = max(cand, words[j - 1]["e"] + P["min_prev_gap_s"])
        elif j > 0:
            cand = max(cand, w["s"] - 0.02)
        if s < cand - 0.05 and _has_reaction(feats, s, w["s"], P["laugh_thr"]):
            new_s = s
        else:
            new_s = max(0.0, cand)
    k = _last_word_in(words, starts, s, e)
    if k is not None:
        w = words[k]
        cand = w["e"] + P["end_pad_s"]
        if k + 1 < len(words) and words[k + 1]["s"] > w["e"]:
            cand = min(cand, words[k + 1]["s"] - P["min_prev_gap_s"])
        if e > cand + 0.05 and _has_reaction(feats, w["e"], e, P["laugh_thr"]):
            new_e = e
        else:
            new_e = max(cand, w["e"] + 0.02)
    if protect:
        inside = [(a, b) for a, b in protect if _overlaps(s, e, a, b)]
        if inside:
            new_s = min(new_s, min(a for a, _ in inside))
            new_e = max(new_e, max(b for _, b in inside))
    if new_e - new_s < 0.2:
        return s, e
    return new_s, new_e


def _extend_ending(e: float, words, starts, feats: Optional[Features], vod_duration: float, P) -> float:
    """Save the ending: keep laughter or wordless vocal reaction after the last word."""
    if feats is None:
        return e
    limit = min(vod_duration, e + P["end_lookahead_s"])
    j = bisect.bisect_left(starts, e - 0.01)
    next_ws = words[j]["s"] if j < len(words) else None
    t = e
    quiet = 0.0
    last_active = e
    while t < limit:
        if next_ws is not None and t >= next_ws - 0.05:
            break
        laugh = feats.event_at("laugh", t) >= P["laugh_thr"]
        vocal = feats.speech_at(t) and feats.above_at(t) >= P["reaction_energy_db"]
        if laugh or vocal:
            last_active = t + HOP10
            quiet = 0.0
        else:
            quiet += HOP10
            if quiet >= P["quiet_stop_s"]:
                break
        t += HOP10
    if last_active <= e + 1e-6:
        return e
    new_e = last_active + P["tail_s"]
    if next_ws is not None:
        new_e = min(new_e, next_ws - 0.05)
    return max(e, min(new_e, limit))


def _fix_seams(pcs: list, words, starts, feats, P, report: dict) -> list:
    for i in range(len(pcs) - 1):
        end_i, start_n = pcs[i][1], pcs[i + 1][0]
        k = _word_at(words, starts, end_i)
        if k is not None:
            end_i = words[k]["e"] + P["seam_gap_s"]
            report["seams_fixed"] += 1
        k = _word_at(words, starts, start_n)
        if k is not None:
            start_n = words[k]["s"] - P["seam_gap_s"]
            report["seams_fixed"] += 1
        if feats is not None:
            thr = P["seam_laugh_thr"]
            if feats.event_at("laugh", end_i - 0.05) >= thr:
                t = end_i
                stop = end_i + P["seam_laugh_max_end_s"]
                j = bisect.bisect_left(starts, end_i)
                if j < len(words):
                    stop = min(stop, max(end_i, words[j]["s"] - P["seam_gap_s"]))
                while t < stop and feats.event_at("laugh", t) >= thr:
                    t += HOP10
                if t > end_i:
                    end_i = t
                    report["seams_fixed"] += 1
            if feats.event_at("laugh", start_n + 0.05) >= thr:
                t = start_n
                stop = start_n - P["seam_laugh_max_start_s"]
                j = bisect.bisect_right(starts, start_n) - 1
                if j >= 0:
                    stop = max(stop, min(start_n, words[j]["e"] + P["seam_gap_s"]))
                while t > stop and feats.event_at("laugh", t - HOP10) >= thr:
                    t -= HOP10
                if t < start_n:
                    start_n = t
                    report["seams_fixed"] += 1
        pcs[i][1], pcs[i + 1][0] = end_i, start_n
    return merge_pieces(pcs, P["merge_gap_s"])


def _shave_piece(s: float, e: float, words, starts, feats: Features, P, protect: Optional[list] = None) -> tuple:
    j0 = bisect.bisect_left(starts, s - 0.01)
    inside = []
    j = j0
    while j < len(words) and words[j]["s"] < e:
        if words[j]["e"] <= e + 0.01:
            inside.append(words[j])
        j += 1
    budget = (e - s) * P["max_shave_ratio"]
    cuts = []
    used = 0.0
    thr = P["shave_event_thr"]
    for k in range(len(inside) - 1):
        a, b = inside[k]["e"], inside[k + 1]["s"]
        if b - a < P["shave_min_gap_s"]:
            continue
        c0, c1 = a + P["shave_keep_after_s"], b - P["shave_keep_before_s"]
        if c1 - c0 < P["shave_min_cut_s"]:
            continue
        if protect:
            pad = P["protect_shave_pad_s"]
            if any(_overlaps(c0, c1, pa - pad, pb + pad) for pa, pb in protect):
                continue
        if (feats.event_max("laugh", c0, c1) >= thr or feats.event_max("scream", c0, c1) >= thr
                or feats.event_max("shout", c0, c1) >= thr):
            continue
        if _has_reaction(feats, c0 - P["shave_post_reaction_s"], c0, P["laugh_thr"]):
            continue
        if feats.speech_frac(c0, c1) > P["shave_max_speech_frac"]:
            continue
        above = feats.above_floor(c0, c1)
        if not len(above) or float((above <= P["shave_floor_db"]).mean()) < P["shave_quiet_frac"]:
            continue
        allowed = budget - used
        if allowed < P["shave_min_cut_s"]:
            break
        if c1 - c0 > allowed:
            mid = (c0 + c1) / 2.0
            c0, c1 = mid - allowed / 2.0, mid + allowed / 2.0
        cuts.append((c0, c1))
        used += c1 - c0
    if not cuts:
        return [[s, e]], 0.0
    out = []
    cur = s
    for c0, c1 in cuts:
        if c0 > cur:
            out.append([cur, c0])
        cur = c1
    if e > cur:
        out.append([cur, e])
    return out, used


def cut_doctor(
    pieces: Sequence[Sequence[float]],
    words: Sequence[dict],
    feats: Optional[Features],
    vod_duration: float,
    fps: Optional[float] = None,
    peak_time: Optional[float] = None,
    prefs: Optional[dict] = None,
    auto_shave: bool = True,
    protect: Optional[Sequence[Sequence[float]]] = None,
) -> tuple:
    """Deterministic clean up of LLM chosen pieces. Returns (pieces, report).
    Order: cover protected beats, snap to words, save the ending, fix seams, safe shave,
    drop tiny pieces, clamp, snap to frames.

    protect: visual beats [[a, b], ...] in VOD seconds (already padded by the caller). A beat
    that touches a piece stays fully inside the result; a beat that ends within
    protect_lookback_s before a piece pulls that piece back over it. Snapping never moves a
    start past a protected start or an end before a protected end, shaving never cuts inside
    one, and pieces that hold one are never dropped as tiny."""
    P = dict(DOCTOR_DEFAULTS)
    if prefs:
        P.update({k: v for k, v in prefs.items() if v is not None})
    vod_duration = float(vod_duration or (feats.duration if feats is not None else 0.0) or 1e9)
    starts = _word_starts(words)
    pcs = merge_pieces(pieces, P["merge_gap_s"])
    report = {
        "llm_net": round(pieces_net(pcs), 2), "final_net": 0.0, "shaved_sec": 0.0,
        "end_extended_sec": 0.0, "seams_fixed": 0, "n_pieces": 0, "duration_status": "empty",
        "warnings": [], "protected": 0, "protect_extended_sec": 0.0,
    }
    if not pcs:
        return [], report

    prot = _clean_protect(protect, vod_duration)
    if prot:
        before = pieces_net(pcs)
        pcs = _apply_protect_coverage(pcs, prot, P)
        report["protect_extended_sec"] = round(max(0.0, pieces_net(pcs) - before), 2)
        prot = [p for p in prot if any(_overlaps(q[0], q[1], p[0], p[1]) for q in pcs)]

    pcs = [list(_snap_piece(s, e, words, starts, feats, P, prot)) for s, e in pcs]
    pcs = merge_pieces(pcs, P["merge_gap_s"])

    old_end = pcs[-1][1]
    pcs[-1][1] = _extend_ending(old_end, words, starts, feats, vod_duration, P)
    report["end_extended_sec"] = round(max(0.0, pcs[-1][1] - old_end), 2)

    pcs = _fix_seams(pcs, words, starts, feats, P, report)

    if auto_shave and feats is not None:
        shaved = []
        total_cut = 0.0
        for s, e in pcs:
            parts, cut = _shave_piece(s, e, words, starts, feats, P, prot)
            shaved.extend(parts)
            total_cut += cut
        pcs = shaved
        report["shaved_sec"] = round(total_cut, 2)

    kept = []
    for s, e in pcs:
        has_peak = peak_time is not None and s <= peak_time <= e
        has_beat = any(_overlaps(s, e, a, b) for a, b in prot)
        if e - s >= P["min_piece_s"] or has_peak or has_beat:
            kept.append([s, e])
    if not kept and pcs:
        kept = [max(pcs, key=lambda p: p[1] - p[0])]
    pcs = merge_pieces(kept, P["merge_gap_s"])
    pcs = [[max(0.0, s), min(vod_duration, e)] for s, e in pcs if min(vod_duration, e) > max(0.0, s)]
    pcs = snap_to_frames(pcs, fps)

    if prot:
        tol = (1.0 / fps) if fps else 0.05
        report["protected"] = sum(1 for a, b in prot
                                  if any(p[0] <= a + tol and p[1] >= b - tol for p in pcs))
        if report["protected"] < len(prot):
            report["warnings"].append("بعض اللحظات البصرية المحمية ما انحفظت كاملة")
    net = pieces_net(pcs)
    report["final_net"] = round(net, 2)
    report["n_pieces"] = len(pcs)
    if net < P["min_net_s"]:
        report["duration_status"] = "short"
    elif net > P["hard_max_s"]:
        report["duration_status"] = "hard_long"
    elif net > P["soft_max_s"]:
        report["duration_status"] = "soft_long"
    elif net > P["target_s"][1]:
        report["duration_status"] = "above_target"
    else:
        report["duration_status"] = "ok"
    return pcs, report


def trim_pieces(pieces: Sequence[Sequence[float]], start_delta: float = 0.0, end_delta: float = 0.0,
                words: Optional[Sequence[dict]] = None, snap_to_words: bool = True,
                vod_duration: Optional[float] = None, fps: Optional[float] = None) -> list:
    """Studio nudges. start_delta is added to the first start (negative starts earlier),
    end_delta is added to the last end (positive ends later). Optional snap to word edges."""
    pcs = merge_pieces(pieces, 0.0)
    if not pcs:
        return []
    new_start = pcs[0][0] + float(start_delta or 0.0)
    new_end = pcs[-1][1] + float(end_delta or 0.0)
    if words and snap_to_words:
        starts = _word_starts(words)
        k = _word_at(words, starts, new_start)
        if k is not None:
            new_start = words[k]["s"] - 0.05
        k = _word_at(words, starts, new_end)
        if k is not None:
            new_end = words[k]["e"] + 0.05
    new_start = max(0.0, new_start)
    if vod_duration:
        new_end = min(float(vod_duration), new_end)
    popped_front = False
    while pcs and new_start >= pcs[0][1]:
        pcs.pop(0)
        popped_front = True
    if not pcs:
        return []
    pcs[0][0] = max(new_start, pcs[0][0]) if popped_front else new_start
    popped_back = False
    while pcs and new_end <= pcs[-1][0]:
        pcs.pop()
        popped_back = True
    if not pcs:
        return []
    pcs[-1][1] = min(new_end, pcs[-1][1]) if popped_back else new_end
    pcs = [p for p in pcs if p[1] - p[0] > 0.05]
    return snap_to_frames(pcs, fps)


def drop_pieces(pieces: Sequence[Sequence[float]], keep_indexes: Sequence[int]) -> list:
    keep = set(int(i) for i in keep_indexes)
    return [list(p) for i, p in enumerate(pieces) if i in keep]


# ============================================================================
# Streams, probing, encoder
# ============================================================================

@dataclass
class StreamInfo:
    vod_url: str
    urls: list
    fps: Optional[float] = None
    fps_str: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    resolved_at: float = field(default_factory=time.time)
    refresher: Optional[Callable[[], list]] = None
    label: str = ""
    format_id: str = ""

    def age_s(self) -> float:
        return time.time() - self.resolved_at

    def refresh(self) -> None:
        log("🔗 تحديث رابط البث (الرابط القديم ممكن انتهت صلاحيته)...")
        if self.refresher is not None:
            urls = list(self.refresher())
        else:
            urls = _resolve_urls(self.vod_url)
        if not urls:
            raise RuntimeError("تعذر تحديث رابط البث")
        self.urls = urls
        self.resolved_at = time.time()


def _parse_rate(rate: Any) -> tuple:
    text = str(rate or "")
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            n, d = float(num), float(den)
            if n > 0 and d > 0:
                return n / d, text
        except ValueError:
            pass
    try:
        val = float(text)
        if val > 0:
            return val, text
    except ValueError:
        pass
    return None, None


def probe_media(target: str, timeout: float = 60.0) -> dict:
    """ffprobe summary: counts of streams per type, video width/height/fps/duration, audio duration."""
    res = run_command(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", target], timeout=timeout)
    info: dict = {"ok": False, "video": {}, "audio": {}, "counts": {}, "format_duration": None}
    if not res.ok:
        info["error"] = res.last_error_line()
        return info
    try:
        data = json.loads(res.stdout or "{}")
    except ValueError:
        info["error"] = "ffprobe json"
        return info
    counts: dict = {}
    for st in data.get("streams", []) or []:
        ctype = st.get("codec_type", "other")
        counts[ctype] = counts.get(ctype, 0) + 1
        if ctype == "video" and not info["video"]:
            fps, fps_str = _parse_rate(st.get("r_frame_rate"))
            if fps is None or fps > 240:
                fps, fps_str = _parse_rate(st.get("avg_frame_rate"))
            info["video"] = {
                "width": st.get("width"), "height": st.get("height"), "fps": fps, "fps_str": fps_str,
                "duration": parse_time_to_sec(st.get("duration")),
                "nb_frames": int(st["nb_frames"]) if str(st.get("nb_frames", "")).isdigit() else None,
            }
        elif ctype == "audio" and not info["audio"]:
            info["audio"] = {"duration": parse_time_to_sec(st.get("duration")),
                             "sample_rate": st.get("sample_rate"), "channels": st.get("channels")}
    info["counts"] = counts
    info["format_duration"] = parse_time_to_sec((data.get("format") or {}).get("duration"))
    info["ok"] = True
    return info


def _resolve_urls(vod_url: str, max_height: int = 1080, timeout: float = 90.0) -> list:
    fmt = f"best[height<={max_height}]/bestvideo[height<={max_height}]+bestaudio/best"
    res = run_command(["yt-dlp", "-f", fmt, "-g", "--no-playlist", "--no-warnings", vod_url], timeout=timeout)
    if not res.ok:
        raise RuntimeError(f"تعذر استخراج رابط البث من تويتش ({res.last_error_line()})")
    urls = [line.strip() for line in (res.stdout or "").splitlines() if line.strip().startswith("http")]
    if not urls:
        raise RuntimeError("تعذر استخراج رابط البث من تويتش (ما رجع yt-dlp أي رابط)")
    return urls


def resolve_stream(vod_url: str, max_height: int = 1080) -> StreamInfo:
    """Resolve direct stream URLs with yt-dlp -g and probe fps and resolution once."""
    log("🔗 جاري استخراج رابط البث المباشر للقص السريع...")
    urls = _resolve_urls(vod_url, max_height=max_height)
    probe = probe_media(urls[0], timeout=60)
    video = probe.get("video") or {}
    fps = video.get("fps")
    info = StreamInfo(vod_url=vod_url, urls=urls, fps=fps if fps else 30.0,
                      fps_str=video.get("fps_str") if fps else "30",
                      width=video.get("width"), height=video.get("height"))
    res_txt = f"{info.width}x{info.height}" if info.height else "غير معروفة"
    log(f"✅ الرابط جاهز (الدقة {res_txt} | {info.fps:.2f} إطار/ث)")
    return info


# ----------------------------------------------------------------------------
# StreamSet: master (final renders) + proxy (Critic windows, Studio previews)
# ----------------------------------------------------------------------------

def _fmt_num(value: Any) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if math.isfinite(val) and val > 0 else None


def _usable_formats(info: dict) -> list:
    """Muxed video formats with a direct URL from a yt-dlp JSON dump."""
    out = []
    for f in info.get("formats") or []:
        if not isinstance(f, dict) or not str(f.get("url") or "").startswith("http"):
            continue
        vcodec = str(f.get("vcodec") or "")
        acodec = str(f.get("acodec") or "")
        if vcodec == "none" or acodec == "none":
            continue
        height = _fmt_num(f.get("height"))
        if height is None:
            match = re.search(r"(\d{3,4})p", str(f.get("format_id") or "") + " " + str(f.get("format_note") or ""))
            height = float(match.group(1)) if match else None
        if height is None:
            continue
        out.append({
            "format_id": str(f.get("format_id") or ""),
            "url": str(f["url"]),
            "height": int(height),
            "width": int(_fmt_num(f.get("width")) or 0) or None,
            "fps": _fmt_num(f.get("fps")),
            "tbr": _fmt_num(f.get("tbr")) or 0.0,
        })
    return out


def _pick_renditions(info: dict, max_height: int, proxy_heights: Sequence[int]) -> dict:
    """{"master": fmt, "proxy": fmt or None} chosen from a yt-dlp JSON dump."""
    formats = _usable_formats(info)
    if not formats:
        return {"master": None, "proxy": None}
    allowed = [f for f in formats if f["height"] <= int(max_height)] or sorted(formats, key=lambda f: f["height"])[:1]
    master = max(allowed, key=lambda f: (f["height"], f["fps"] or 0.0, f["tbr"]))
    proxy = None
    for h in proxy_heights:
        near = [f for f in formats if abs(f["height"] - int(h)) <= 24 and f["height"] < master["height"]]
        if near:
            proxy = min(near, key=lambda f: (f["tbr"] or 1e9, f["fps"] or 0.0))
            break
    return {"master": master, "proxy": proxy}


def _dump_info(vod_url: str, timeout: float = 120.0) -> dict:
    res = run_command(["yt-dlp", "-J", "--no-playlist", "--no-warnings", vod_url], timeout=timeout)
    if not res.ok:
        raise RuntimeError(f"تعذر قراءة معلومات البث من تويتش ({res.last_error_line()})")
    try:
        data = json.loads(res.stdout or "{}")
    except ValueError as exc:
        raise RuntimeError("رد yt-dlp مو JSON صالح") from exc
    if not isinstance(data, dict):
        raise RuntimeError("رد yt-dlp بصيغة غير متوقعة")
    return data


def _resolve_picks(vod_url: str, max_height: int, proxy_heights: Sequence[int]) -> dict:
    """{"master": {"urls", "height", "width", "fps", "format_id"}, "proxy": {...} | None}.
    One yt-dlp JSON call; falls back to yt-dlp -g for the master when JSON has no muxed formats."""
    picks: dict = {"master": None, "proxy": None}
    try:
        chosen = _pick_renditions(_dump_info(vod_url), max_height, proxy_heights)
        for key in ("master", "proxy"):
            f = chosen.get(key)
            if f:
                picks[key] = {"urls": [f["url"]], "height": f["height"], "width": f["width"],
                              "fps": f["fps"], "format_id": f["format_id"]}
    except Exception as exc:
        log(f"⚠️ قراءة الجودات من تويتش ما نجحت ({short_error(exc, 100)})، نستخدم الطريقة القديمة")
    if picks["master"] is None:
        picks["master"] = {"urls": _resolve_urls(vod_url, max_height=max_height), "height": None,
                           "width": None, "fps": None, "format_id": ""}
        picks["proxy"] = None
    return picks


@dataclass
class StreamSet:
    """Two renditions of one VOD that refresh together.
    master: best height up to max_height, used for final 1080p renders and the A/V offset.
    proxy:  a low rendition (360p first) for Critic windows and Studio previews. When Twitch
            exposes no low rendition, proxy points at the master URLs (proxy_is_master=True)."""
    vod_url: str
    master: StreamInfo
    proxy: StreamInfo
    resolved_at: float = field(default_factory=time.time)
    max_height: int = 1080
    proxy_heights: tuple = PROXY_HEIGHTS_DEFAULT
    proxy_is_master: bool = False
    resolver: Optional[Callable[[], dict]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.master.label = self.master.label or "master"
        self.proxy.label = self.proxy.label or "proxy"
        self.master.refresher = lambda: self._refresh_for("master")
        self.proxy.refresher = lambda: self._refresh_for("proxy")

    def age_s(self) -> float:
        return time.time() - self.resolved_at

    def _refresh_for(self, which: str) -> list:
        self.refresh(quiet=True)
        return list(getattr(self, which).urls)

    def refresh(self, quiet: bool = False) -> None:
        """Resolve both renditions again (one yt-dlp call) and update their URLs in place."""
        if not quiet:
            log("🔗 تحديث روابط البث (الجودة الأساسية وجودة المعاينة)...")
        if self.resolver is not None:
            picks = self.resolver()
        else:
            picks = _resolve_picks(self.vod_url, self.max_height, self.proxy_heights)
        master = (picks or {}).get("master")
        if not master or not master.get("urls"):
            raise RuntimeError("تعذر تحديث رابط البث")
        now = time.time()
        self.master.urls = list(master["urls"])
        self.master.resolved_at = now
        proxy = (picks or {}).get("proxy")
        if proxy and proxy.get("urls"):
            self.proxy.urls = list(proxy["urls"])
            self.proxy_is_master = False
        else:
            self.proxy.urls = list(master["urls"])
            self.proxy_is_master = True
        self.proxy.resolved_at = now
        self.resolved_at = now

    def ensure_fresh(self) -> bool:
        """Refresh proactively when the URLs are older than STREAM_MAX_AGE_S."""
        if self.age_s() > STREAM_MAX_AGE_S:
            self.refresh()
            return True
        return False


def _stream_from_pick(vod_url: str, pick: dict, label: str, probe: bool) -> StreamInfo:
    fps = pick.get("fps")
    fps_str = None
    width, height = pick.get("width"), pick.get("height")
    if probe:
        video = (probe_media(pick["urls"][0], timeout=60).get("video") or {})
        if video.get("fps"):
            fps, fps_str = video.get("fps"), video.get("fps_str")
        width = video.get("width") or width
        height = video.get("height") or height
    if not fps:
        fps, fps_str = 30.0, "30"
    return StreamInfo(vod_url=vod_url, urls=list(pick["urls"]), fps=float(fps),
                      fps_str=fps_str or f"{float(fps):.6g}", width=width, height=height,
                      label=label, format_id=str(pick.get("format_id") or ""))


def resolve_streams(vod_url: str, max_height: int = 1080,
                    proxy_heights: Sequence[int] = PROXY_HEIGHTS_DEFAULT) -> StreamSet:
    """Resolve the master and proxy renditions with one yt-dlp JSON call. The master is
    probed for an exact frame rate (frame snapping and SRT timing depend on it)."""
    log("🔗 جاري استخراج روابط البث (جودة أساسية للقص وجودة منخفضة للناقد والمعاينة)...")
    heights = tuple(int(h) for h in proxy_heights)
    picks = _resolve_picks(vod_url, max_height, heights)
    master = _stream_from_pick(vod_url, picks["master"], "master", probe=True)
    proxy_pick = picks.get("proxy")
    if proxy_pick:
        proxy = _stream_from_pick(vod_url, proxy_pick, "proxy", probe=False)
        proxy_is_master = False
    else:
        proxy = StreamInfo(vod_url=vod_url, urls=list(master.urls), fps=master.fps, fps_str=master.fps_str,
                           width=master.width, height=master.height, label="proxy", format_id=master.format_id)
        proxy_is_master = True
    streams = StreamSet(vod_url=vod_url, master=master, proxy=proxy, max_height=int(max_height),
                        proxy_heights=heights, proxy_is_master=proxy_is_master)
    m_txt = f"{master.width}x{master.height}" if master.height else "غير معروفة"
    p_txt = "نفس الأساسية (ما فيه جودة منخفضة)" if proxy_is_master else f"{proxy.height}p"
    log(f"✅ الروابط جاهزة: الأساسية {m_txt} | {master.fps:.2f} إطار/ث، المعاينة {p_txt}")
    return streams


_ENCODER_CACHE: dict = {}


def pick_encoder(force_cpu: bool = False) -> str:
    """'h264_nvenc' if a real one frame NVENC encode works, else 'libx264'. Cached per session."""
    if force_cpu:
        return "libx264"
    if "best" in _ENCODER_CACHE:
        return _ENCODER_CACHE["best"]
    res = run_command(["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.3",
                       "-c:v", "h264_nvenc", "-f", "null", "-"], timeout=30)
    best = "h264_nvenc" if res.ok else "libx264"
    _ENCODER_CACHE["best"] = best
    log(f"🎛️ المرمّز المستخدم: {'h264_nvenc (كرت الشاشة)' if best == 'h264_nvenc' else 'libx264 (المعالج)'}")
    return best


def _mark_nvenc_broken() -> None:
    _ENCODER_CACHE["best"] = "libx264"


def encoder_args(name: str, fps: Optional[float]) -> list:
    gop = str(int(round((fps or 30.0) * 2)))
    if name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0",
                "-profile:v", "high", "-g", gop, "-bf", "2"]
    return ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-profile:v", "high",
            "-g", gop, "-bf", "2"]


# ============================================================================
# Cutter
# ============================================================================

def build_cut_command(urls: Sequence[str], pieces: Sequence[Sequence[float]], out_path: str,
                      encoder: str, fps: Optional[float], fps_str: Optional[str],
                      src_height: Optional[int], max_height: int = 1080,
                      time_offset: float = 0.0) -> list:
    """One input per piece (one per piece per URL when video and audio are separate),
    timestamps reset per piece, exact length audio with 10 ms fades, concat, encode.
    Memory stays flat because FFmpeg only reads the input concat is consuming."""
    n_in = 2 if len(urls) >= 2 else 1
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error"]
    for s, e in pieces:
        d = float(e) - float(s)
        ss = max(0.0, float(s) + float(time_offset))
        for u in list(urls)[:n_in]:
            cmd += ["-rw_timeout", RW_TIMEOUT_US, "-ss", f"{ss:.6f}", "-t", f"{d:.6f}", "-i", u]
    filters = []
    concat_in = ""
    for i, (s, e) in enumerate(pieces):
        d = float(e) - float(s)
        vi = i * n_in
        ai = i * n_in + (n_in - 1)
        fade = min(0.010, d / 4.0)
        filters.append(f"[{vi}:v:0]setpts=PTS-STARTPTS[v{i}]")
        filters.append(
            f"[{ai}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,"
            f"aresample=48000:async=1:first_pts=0,atrim=duration={d:.6f},apad=whole_dur={d:.6f},"
            f"afade=t=in:st=0:d={fade:.4f},afade=t=out:st={max(0.0, d - fade):.6f}:d={fade:.4f}[a{i}]"
        )
        concat_in += f"[v{i}][a{i}]"
    filters.append(f"{concat_in}concat=n={len(pieces)}:v=1:a=1[vc][ac]")
    last = "vc"
    if src_height and int(src_height) > max_height:
        filters.append(f"[vc]scale=-2:{max_height}:flags=lanczos[vs]")
        last = "vs"
    filters.append(f"[{last}]format=yuv420p[vout]")
    cmd += ["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[ac]",
            "-sn", "-dn", "-map_metadata", "-1"]
    if fps:
        cmd += ["-r", fps_str or f"{fps:.6f}", "-fps_mode", "cfr"]
    cmd += encoder_args(encoder, fps)
    cmd += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart", out_path]
    return cmd


def _verify_output(path: str, expected: float, fps: Optional[float], max_height: int) -> tuple:
    if not os.path.isfile(path) or os.path.getsize(path) < 10_000:
        return False, {}, "الملف الناتج فاضي أو صغير جداً"
    info = probe_media(path, timeout=60)
    if not info.get("ok"):
        return False, info, f"ffprobe فشل ({info.get('error', '')})"
    counts = info.get("counts", {})
    if counts.get("video", 0) != 1 or counts.get("audio", 0) != 1 or sum(counts.values()) != 2:
        return False, info, f"عدد المسارات غير صحيح {counts}"
    frame = 1.0 / fps if fps else 0.04
    vdur = (info.get("video") or {}).get("duration") or info.get("format_duration") or 0.0
    adur = (info.get("audio") or {}).get("duration") or info.get("format_duration") or 0.0
    if abs(vdur - expected) > frame + 0.06:
        return False, info, f"مدة الفيديو {vdur:.3f}ث بدل {expected:.3f}ث"
    if abs(adur - expected) > 0.10:
        return False, info, f"مدة الصوت {adur:.3f}ث بدل {expected:.3f}ث"
    height = (info.get("video") or {}).get("height") or 0
    if height and height > max_height:
        return False, info, f"الدقة {height} أعلى من {max_height}"
    return True, info, ""


_HTTP_FAIL_RE = re.compile(r"(40[0-9]|410|Forbidden|Not Found|HTTP error|Server returned|expired|Connection refused|timed out|I/O error)", re.IGNORECASE)


def cut_and_stitch(
    stream: StreamInfo,
    pieces: Sequence[Sequence[float]],
    out_mp4: str,
    time_offset: float = 0.0,
    max_height: int = 1080,
    timeout_s: Optional[float] = None,
    allow_refresh: bool = True,
    force_cpu: bool = False,
) -> dict:
    """Cut and stitch pieces from the live stream into a clean MP4 on Drive.
    Renders in /tmp, verifies with ffprobe, then copies atomically to out_mp4.
    Returns {"duration", "size_mb", "encoder", "refreshed", "peak_mb", "height", "fps"}."""
    pcs = merge_pieces(pieces, 0.0)
    if not pcs:
        raise RuntimeError("ما فيه مقاطع للقص")
    expected = pieces_net(pcs)
    render_dir = ensure_dir(os.path.join(TMP_ROOT, "render"))
    tmp_out = os.path.join(render_dir, os.path.basename(out_mp4))
    timeout = float(timeout_s) if timeout_s else 120.0 + 8.0 * expected + 3.0 * len(pcs)

    refreshed = False
    if allow_refresh and stream.refresher is None and stream.vod_url and stream.age_s() > STREAM_MAX_AGE_S:
        stream.refresh()
        refreshed = True

    def attempt(encoder: str) -> tuple:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        cmd = build_cut_command(stream.urls, pcs, tmp_out, encoder, stream.fps, stream.fps_str,
                                stream.height, max_height=max_height, time_offset=time_offset)
        run = _run_measured(cmd, timeout)
        if not run.ok:
            return False, run, None, run.last_line()
        ok, info, reason = _verify_output(tmp_out, expected, stream.fps, max_height)
        return ok, run, info, reason

    last_reason = ""
    peak = 0.0
    info: Optional[dict] = None
    success = False
    used_encoder = ""
    for round_no in range(2):
        encoders = [pick_encoder(force_cpu)]
        if encoders[0] != "libx264":
            encoders.append("libx264")
        http_like = False
        for enc in encoders:
            ok, run, info, reason = attempt(enc)
            peak = max(peak, run.peak_mb)
            if ok:
                success = True
                used_encoder = enc
                break
            last_reason = reason
            if run.stderr and _HTTP_FAIL_RE.search(run.stderr):
                http_like = True
            if enc == "h264_nvenc":
                _mark_nvenc_broken()
            if http_like:
                break
        if success:
            break
        if round_no == 0 and allow_refresh and not refreshed:
            try:
                stream.refresh()
                refreshed = True
                continue
            except Exception as exc:
                last_reason = f"{last_reason} | تحديث الرابط فشل: {short_error(exc)}"
        break

    if not success:
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
        raise RuntimeError(f"فشل FFmpeg أثناء القص والدمج: {last_reason}")

    safe_copy_file(tmp_out, out_mp4)
    size_mb = os.path.getsize(tmp_out) / (1024 * 1024)
    os.remove(tmp_out)
    video = (info or {}).get("video") or {}
    return {
        "duration": round(expected, 3),
        "size_mb": round(size_mb, 2),
        "encoder": used_encoder,
        "refreshed": refreshed,
        "peak_mb": round(peak, 1),
        "height": video.get("height"),
        "fps": stream.fps,
    }


def render_pieces(
    streams: Union[StreamSet, StreamInfo],
    pieces: Sequence[Sequence[float]],
    out_mp4: str,
    out_srt: str,
    words: Sequence[dict],
    time_offset: float = 0.0,
    max_height: int = 1080,
    srt_bom: bool = False,
) -> dict:
    """Final CapCut deliverables for one clip: frame snapped pieces cut from the master
    rendition (cut_and_stitch) and the matching SRT (generate_srt). Used by Stage 3 and
    the Studio so both follow exactly the same path.
    Returns the cut_and_stitch info plus "segments" (the frame snapped pieces) and "srt"."""
    master = streams.master if isinstance(streams, StreamSet) else streams
    refreshed = False
    if isinstance(streams, StreamSet):
        refreshed = streams.ensure_fresh()
    pcs = snap_to_frames(merge_pieces(pieces, 0.0), master.fps)
    if not pcs:
        raise RuntimeError("ما فيه مقاطع للقص")
    info = cut_and_stitch(master, pcs, out_mp4, time_offset=time_offset, max_height=max_height)
    srt = generate_srt(words, pcs, out_srt, fps=master.fps, bom=srt_bom)
    out = dict(info)
    out["refreshed"] = bool(info.get("refreshed") or refreshed)
    out["segments"] = pcs
    out["srt"] = {k: srt[k] for k in ("cues", "min_cue_s", "avg_cps", "duration", "errors")}
    return out


# ============================================================================
# SRT engine
# ============================================================================

def map_words_to_timeline(words: Sequence[dict], pieces: Sequence[Sequence[float]]) -> tuple:
    """Words whose midpoint falls inside a piece, mapped to the stitched timeline.
    Returns (mapped_words, total_duration). Each mapped word carries piece index,
    seam flag (first word of a later piece) and the piece end on the timeline."""
    starts = _word_starts(words)
    out = []
    offset = 0.0
    for pi, (s, e) in enumerate(pieces):
        s, e = float(s), float(e)
        d = e - s
        j = max(0, bisect.bisect_left(starts, s - 3.0))
        first = True
        while j < len(words) and words[j]["s"] < e:
            w = words[j]
            mid = (w["s"] + w["e"]) / 2.0
            if s <= mid < e:
                ms = offset + max(0.0, w["s"] - s)
                me = offset + min(d, w["e"] - s)
                if me <= ms:
                    me = min(offset + d, ms + 0.05)
                out.append({"w": w["w"], "s": ms, "e": me, "seam": first and pi > 0,
                            "piece": pi, "piece_end": offset + d})
                first = False
            j += 1
        offset += d
    return out, offset


def _clean_token(tok: str) -> str:
    t = _LAUGH_RE.sub("هههههه", str(tok).strip())
    return t


def _bare(tok: str) -> str:
    return normalize_arabic(tok.strip(_PUNCT_STRIP))


def _band_of(cps: float) -> str:
    if cps >= SRT_FAST_CPS:
        return "fast"
    if cps < SRT_CALM_CPS:
        return "calm"
    return "normal"


def _local_cps(mapped: list, i: int, window: float = 1.5) -> float:
    center = (mapped[i]["s"] + mapped[i]["e"]) / 2.0
    lo, hi = center - window, center + window
    chars = 0
    first_s, last_e = None, None
    for w in mapped:
        if w["e"] < lo:
            continue
        if w["s"] > hi:
            break
        chars += arabic_char_count(w["w"])
        first_s = w["s"] if first_s is None else first_s
        last_e = w["e"]
    if first_s is None:
        return 10.0
    span = max(0.6, last_e - first_s)
    return chars / span


def _group_cues(mapped: list) -> list:
    cues: list = []
    buf: list = []
    band: Optional[str] = None
    pending: Optional[str] = None
    pending_count = 0

    def flush(cur_band: str) -> None:
        if buf:
            cues.append({"words": list(buf), "band": cur_band, "piece": buf[0]["piece"],
                         "piece_end": buf[0]["piece_end"]})
            buf.clear()

    for i, w in enumerate(mapped):
        raw = _band_of(_local_cps(mapped, i))
        if band is None:
            band = raw
        elif raw != band:
            if raw == pending:
                pending_count += 1
            else:
                pending, pending_count = raw, 1
            if pending_count >= 2:
                band, pending, pending_count = raw, None, 0
        else:
            pending, pending_count = None, 0
        cfg = SRT_BANDS[band]
        if buf:
            prev = buf[-1]
            gap = w["s"] - prev["e"]
            projected = len(" ".join(x["w"] for x in buf)) + 1 + len(w["w"])
            must = w["seam"] or w["piece"] != prev["piece"] or gap > cfg["pause"]
            soft = len(buf) >= cfg["max_words"] or projected > cfg["max_chars"]
            if must:
                flush(band)
            elif soft:
                glue = _bare(prev["w"]) in GLUE_WORDS
                if not (glue and len(buf) <= cfg["max_words"] and projected <= cfg["max_chars"] + 10):
                    flush(band)
        buf.append(w)
        token = w["w"].rstrip()
        if token.endswith(_STRONG_END):
            flush(band)
        elif token.endswith(_SOFT_END) and len(buf) >= 2:
            flush(band)
    flush(band or "normal")
    return cues


def _merge_flashes(cues: list) -> list:
    """Fold a cue that would flash (next cue starts before half its minimum time)
    into the following cue of the same piece."""
    out: list = []
    i = 0
    while i < len(cues):
        c = cues[i]
        if i + 1 < len(cues):
            n = cues[i + 1]
            cfg = SRT_BANDS[c["band"]]
            span_to_next = n["words"][0]["s"] - c["words"][0]["s"]
            n_words = len(c["words"]) + len(n["words"])
            chars = len(" ".join(x["w"] for x in c["words"] + n["words"]))
            if (n["piece"] == c["piece"] and span_to_next < 0.5 * cfg["min_dur"]
                    and n_words <= cfg["max_words"] + 2 and chars <= cfg["max_chars"] + 10):
                n["words"] = c["words"] + n["words"]
                i += 1
                continue
        out.append(c)
        i += 1
    return out


def _time_cues(cues: list, total: float, fps: Optional[float]) -> list:
    frame = 1.0 / fps if fps else 0.02
    timed: list = []
    for i, c in enumerate(cues):
        text = " ".join(_clean_token(x["w"]) for x in c["words"]).strip()
        if not text:
            continue
        start = c["words"][0]["s"]
        end = c["words"][-1]["e"]
        if timed:
            start = max(start, timed[-1]["e"])
        cap = min(c["piece_end"], total)
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        if nxt is not None:
            cap = min(cap, nxt["words"][0]["s"] - frame)
        end = max(end, start + SRT_BANDS[c["band"]]["min_dur"])
        if nxt is not None and nxt["piece"] == c["piece"] and nxt["words"][0]["s"] - end < SRT_BRIDGE_S:
            end = nxt["words"][0]["s"] - frame
        end = min(end, cap)
        if end - start < SRT_MIN_CUE_S:
            if timed and timed[-1]["piece"] == c["piece"]:
                timed[-1]["text"] = f"{timed[-1]['text']} {text}"
                timed[-1]["e"] = max(timed[-1]["e"], min(max(c["words"][-1]["e"], timed[-1]["e"]), cap))
                continue
            end = min(start + SRT_MIN_CUE_S, total)
            if end - start < SRT_MIN_CUE_S * 0.5:
                continue
        timed.append({"s": start, "e": end, "text": text, "piece": c["piece"], "band": c["band"]})
    return timed


def cues_to_srt(cues: Sequence[dict]) -> str:
    blocks = []
    for idx, c in enumerate(cues, start=1):
        blocks.append(f"{idx}\n{sec_to_srt_time(c['s'])} --> {sec_to_srt_time(c['e'])}\n{c['text']}")
    return ("\n\n".join(blocks) + "\n") if blocks else ""


def generate_srt(words: Sequence[dict], pieces: Sequence[Sequence[float]], out_srt: Optional[str] = None,
                 fps: Optional[float] = None, bom: bool = False) -> dict:
    """Dynamic pacing SRT synced to the stitched clip. Writes out_srt atomically when given.
    Returns stats: cues, min_cue_s, avg_cps, duration, bands, errors, text."""
    mapped, total = map_words_to_timeline(words, pieces)
    cues = _merge_flashes(_group_cues(mapped)) if mapped else []
    timed = _time_cues(cues, total, fps)
    text = cues_to_srt(timed)
    errors = validate_srt(text, clip_duration=total)
    if out_srt:
        atomic_write_text(out_srt, text, encoding="utf-8-sig" if bom else "utf-8")
    durations = [c["e"] - c["s"] for c in timed]
    speech_time = sum(w["e"] - w["s"] for w in mapped) or 1.0
    chars = sum(arabic_char_count(w["w"]) for w in mapped)
    bands: dict = {}
    for c in timed:
        bands[c["band"]] = bands.get(c["band"], 0) + 1
    return {
        "cues": len(timed),
        "min_cue_s": round(min(durations), 3) if durations else 0.0,
        "avg_cps": round(chars / speech_time, 2) if mapped else 0.0,
        "avg_wps": round(len(mapped) / speech_time, 2) if mapped else 0.0,
        "duration": round(total, 3),
        "bands": bands,
        "errors": errors,
        "text": text,
    }


_SRT_TIME_RE = re.compile(
    r"^\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*$"
)


def _hmsms(h: str, m: str, s: str, ms: str) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def parse_srt(text: str) -> list:
    """[(index_or_None, start, end, text, block_number)] for every block with a valid time line."""
    out = []
    norm = str(text or "").replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in re.split(r"\n\s*\n", norm.strip()) if b.strip()]
    for bno, block in enumerate(blocks, start=1):
        lines = block.split("\n")
        idx = None
        if lines and lines[0].strip().isdigit():
            idx = int(lines[0].strip())
            lines = lines[1:]
        if not lines:
            out.append((idx, None, None, "", bno))
            continue
        m = _SRT_TIME_RE.match(lines[0])
        if not m:
            out.append((idx, None, None, "\n".join(lines).strip(), bno))
            continue
        st = _hmsms(*m.groups()[:4])
        en = _hmsms(*m.groups()[4:])
        out.append((idx, st, en, "\n".join(lines[1:]).strip(), bno))
    return out


def validate_srt(text: str, clip_duration: Optional[float] = None) -> list:
    """Arabic error list (empty means valid): time format, order, overlaps, empty text,
    cues shorter than 0.1 s, cues past the clip end."""
    errors = []
    blocks = parse_srt(text)
    if not str(text or "").strip():
        return errors
    prev_end = None
    for idx, st, en, body, bno in blocks:
        label = f"السطر رقم {idx if idx is not None else bno}"
        if st is None or en is None:
            errors.append(f"{label}: صيغة التوقيت غير صحيحة")
            continue
        if en <= st:
            errors.append(f"{label}: النهاية قبل البداية أو تساويها")
        elif en - st < SRT_MIN_CUE_S - 1e-6:
            errors.append(f"{label}: مدة السطر أقل من 0.1 ثانية")
        if not body:
            errors.append(f"{label}: السطر فاضي بدون كلام")
        if prev_end is not None and st < prev_end - 0.0005:
            errors.append(f"{label}: يتداخل مع السطر اللي قبله")
        if clip_duration is not None and en > clip_duration + 0.05:
            errors.append(f"{label}: ينتهي بعد نهاية اللقطة ({sec_to_srt_time(en)})")
        prev_end = en if prev_end is None else max(prev_end, en)
    return errors


# ============================================================================
# Window media (Critic video + Studio preview), Gemini transcodes, waveform
# ============================================================================

def _scale_no_upscale(height: int) -> str:
    """Scale to at most `height` lines, keeping aspect, never upscaling."""
    h = int(max(16, height))
    return f"scale=w=-2:h='min({h}\\,ih)':flags=bicubic"


def _probe_dur(info: dict) -> float:
    return float((info.get("video") or {}).get("duration") or info.get("format_duration") or 0.0)


def _window_command(urls: Sequence[str], p0: float, dur: float, crit: Optional[tuple], prev_path: Optional[str],
                    crit_path: Optional[str], critic_fps: float, critic_height: int, preview_height: int,
                    src_fps: Optional[float]) -> list:
    """One input (or video + audio inputs), one decode, up to two encodes."""
    n_in = 2 if len(urls) >= 2 else 1
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error"]
    for u in list(urls)[:n_in]:
        cmd += ["-rw_timeout", RW_TIMEOUT_US, "-ss", f"{max(0.0, p0):.3f}", "-t", f"{dur:.3f}", "-i", u]
    ai = n_in - 1
    outs = []
    if prev_path:
        outs.append("p")
    if crit_path:
        outs.append("c")
    filters = [f"[0:v:0]setpts=PTS-STARTPTS" + (f",split={len(outs)}" if len(outs) > 1 else "")
               + "".join(f"[v{o}]" for o in outs),
               f"[{ai}:a:0]aformat=sample_rates=48000:channel_layouts=stereo,aresample=48000:async=1:first_pts=0"
               + (f",asplit={len(outs)}" if len(outs) > 1 else "") + "".join(f"[a{o}]" for o in outs)]
    if prev_path:
        filters.append(f"[vp]{_scale_no_upscale(preview_height)},format=yuv420p[pv]")
        filters.append("[ap]anull[pa]")
    if crit_path and crit is not None:
        c0, cd = crit
        filters.append(f"[vc]trim=start={c0:.3f}:duration={cd:.3f},setpts=PTS-STARTPTS,"
                       f"fps={critic_fps:g},{_scale_no_upscale(critic_height)},format=yuv420p[cv]")
        filters.append(f"[ac]atrim=start={c0:.3f}:duration={cd:.3f},asetpts=PTS-STARTPTS[ca]")
    cmd += ["-filter_complex", ";".join(filters)]
    if prev_path:
        gop = str(max(1, int(round((src_fps or 30.0) * WINDOW_PREVIEW_GOP_S))))
        cmd += ["-map", "[pv]", "-map", "[pa]", "-sn", "-dn", "-map_metadata", "-1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(WINDOW_PREVIEW_CRF), "-g", gop,
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "48000",
                "-movflags", "+faststart", prev_path]
    if crit_path:
        cmd += ["-map", "[cv]", "-map", "[ca]", "-sn", "-dn", "-map_metadata", "-1",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", str(WINDOW_CRITIC_CRF),
                "-g", str(max(1, int(round(critic_fps * 2)))), "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "32k", "-ac", "1", "-ar", "16000",
                "-movflags", "+faststart", crit_path]
    return cmd


def _check_window_file(path: str, expected: float, max_height: int) -> tuple:
    if not os.path.isfile(path) or os.path.getsize(path) < 2_000:
        return False, {}, "الملف الناتج فاضي"
    info = probe_media(path, timeout=60)
    if not info.get("ok"):
        return False, info, f"ffprobe فشل ({info.get('error', '')})"
    counts = info.get("counts", {})
    if counts.get("video", 0) != 1 or counts.get("audio", 0) != 1:
        return False, info, f"عدد المسارات غير صحيح {counts}"
    dur = _probe_dur(info)
    if dur < min(0.5, expected * 0.5):
        return False, info, f"المدة {dur:.2f}ث قصيرة جداً"
    height = (info.get("video") or {}).get("height") or 0
    if height and height > max_height:
        return False, info, f"الدقة {height} أعلى من {max_height}"
    return True, info, ""


def make_window_media(
    stream: Union[StreamInfo, StreamSet],
    t0: float,
    t1: float,
    critic_out: Optional[str] = None,
    preview_out: Optional[str] = None,
    preview_pad_s: float = 15.0,
    time_offset: float = 0.0,
    critic_fps: float = 4.0,
    critic_height: int = 360,
    preview_height: int = 360,
    allow_refresh: bool = True,
    timeout_s: Optional[float] = None,
) -> dict:
    """Write the Critic video for [t0, t1] and the Studio preview for [t0 - pad, t1 + pad]
    in ONE FFmpeg pass straight from the stream URL (no full download).

      critic  : low fps (critic_fps), at most critic_height lines, mono 32 kbps audio, small
      preview : native frame rate, at most preview_height lines, AAC 96 kbps, 1 s keyframes
                for fast seeking in the browser

    stream may be a StreamSet (proxy first, then the master scaled down) or a single
    StreamInfo. Expired URLs are refreshed once. Times are transcript times; time_offset is
    added for the stream (stream_time = transcript_time + time_offset).
    Returns {"critic", "preview", "t0", "t1", "p0", "p1", "seconds", "preview_seconds",
    "source", "refreshed"}. Raises RuntimeError when every source fails (the caller then
    falls back to an audio snippet)."""
    t0, t1 = float(t0), float(t1)
    if t1 <= t0:
        raise ValueError("نافذة الفيديو فاضية")
    if not critic_out and not preview_out:
        raise ValueError("لازم نحدد ملف واحد على الأقل (الناقد أو المعاينة)")
    pad = max(0.0, float(preview_pad_s)) if preview_out else 0.0
    p0 = max(0.0, t0 - pad)
    p1 = t1 + pad
    dur = p1 - p0
    crit = (t0 - p0, t1 - t0) if critic_out else None
    timeout = float(timeout_s) if timeout_s else 90.0 + 1.5 * dur

    refreshed = False
    if isinstance(stream, StreamSet):
        if allow_refresh and stream.ensure_fresh():
            refreshed = True
        sources = [stream.proxy] if stream.proxy_is_master else [stream.proxy, stream.master]
    else:
        if allow_refresh and stream.refresher is None and stream.vod_url and stream.age_s() > STREAM_MAX_AGE_S:
            stream.refresh()
            refreshed = True
        sources = [stream]

    work = ensure_dir(os.path.join(TMP_ROOT, "windows"))
    tag = f"{os.getpid()}_{int(time.time() * 1000)}"
    tmp_prev = os.path.join(work, f"prev_{tag}.mp4") if preview_out else None
    tmp_crit = os.path.join(work, f"crit_{tag}.mp4") if critic_out else None

    def cleanup() -> None:
        for path in (tmp_prev, tmp_crit):
            if path and os.path.exists(path):
                os.remove(path)

    last_reason = ""
    for src in sources:
        for round_no in range(2):
            cleanup()
            cmd = _window_command(src.urls, p0 + float(time_offset), dur, crit, tmp_prev, tmp_crit,
                                  float(critic_fps), int(critic_height), int(preview_height), src.fps)
            run = _run_measured(cmd, timeout)
            ok = run.ok
            reason = run.last_line()
            infos = {}
            if ok:
                for key, path, expect, max_h in (("preview", tmp_prev, dur, preview_height),
                                                 ("critic", tmp_crit, t1 - t0, critic_height)):
                    if not path:
                        continue
                    good, info, why = _check_window_file(path, expect, max_h)
                    if not good:
                        ok, reason = False, f"{key}: {why}"
                        break
                    infos[key] = info
            if ok:
                result = {"critic": None, "preview": None, "t0": round(t0, 3), "t1": round(t1, 3),
                          "p0": round(p0, 3), "p1": round(p1, 3), "seconds": 0.0, "preview_seconds": 0.0,
                          "source": src.label or "stream", "refreshed": refreshed}
                if tmp_prev and preview_out:
                    safe_copy_file(tmp_prev, preview_out)
                    result["preview"] = preview_out
                    result["preview_seconds"] = round(_probe_dur(infos["preview"]), 3)
                    result["p1"] = round(p0 + result["preview_seconds"], 3)
                if tmp_crit and critic_out:
                    safe_copy_file(tmp_crit, critic_out)
                    result["critic"] = critic_out
                    result["seconds"] = round(_probe_dur(infos["critic"]), 3)
                else:
                    result["seconds"] = result["preview_seconds"]
                cleanup()
                return result
            last_reason = reason
            if round_no == 0 and allow_refresh:
                try:
                    src.refresh()
                    refreshed = True
                    continue
                except Exception as exc:
                    last_reason = f"{last_reason} | تحديث الرابط فشل: {short_error(exc)}"
            break
        if len(sources) > 1 and src is sources[0]:
            log(f"   ↪️ جودة المعاينة ما اشتغلت ({last_reason[:80]})، نجرب الجودة الأساسية")
    cleanup()
    raise RuntimeError(f"تعذر تجهيز فيديو النافذة: {last_reason}")


def transcode_for_gemini(src: str, out: str, fps: float = 3.0, short_side: int = 480,
                         audio_kbps: int = 64, max_seconds: float = 240.0) -> dict:
    """Compact MP4 for Gemini video understanding (Golden Clips): H.264 at `fps`, short side
    at most `short_side` (portrait or landscape, never upscaled), CRF 32, mono AAC 16 kHz,
    capped at max_seconds. Written to a temp file next to `out`, then renamed.
    Returns {"path", "seconds", "bytes", "width", "height", "truncated"}."""
    if not os.path.isfile(src):
        raise FileNotFoundError(f"الملف غير موجود: {src}")
    src_info = probe_media(src, timeout=60)
    src_dur = _probe_dur(src_info) if src_info.get("ok") else 0.0
    if src_info.get("ok") and not (src_info.get("counts") or {}).get("video"):
        raise RuntimeError(f"الملف ما فيه فيديو: {os.path.basename(src)}")
    cap = max(1.0, float(max_seconds))
    truncated = bool(src_dur and src_dur > cap + 0.5)
    side = int(max(64, short_side))
    vf = (f"scale=w='if(gte(iw\\,ih)\\,-2\\,min(iw\\,{side}))':h='if(gte(iw\\,ih)\\,min(ih\\,{side})\\,-2)'"
          f":flags=bicubic,fps={float(fps):g},format=yuv420p")
    ensure_dir(os.path.dirname(os.path.abspath(out)))
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), f".{os.path.basename(out)}.{os.getpid()}.tmp.mp4")
    cmd = ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-v", "error", "-i", src, "-t", f"{cap:.3f}",
           "-map", "0:v:0", "-map", "0:a:0?", "-vf", vf, "-sn", "-dn", "-map_metadata", "-1",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "32", "-g", str(max(1, int(round(float(fps) * 2)))),
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{int(audio_kbps)}k", "-ac", "1", "-ar", "16000",
           "-movflags", "+faststart", tmp]
    run = _run_measured(cmd, 120.0 + 2.0 * min(cap, src_dur or cap))
    if not run.ok:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(f"تعذر تحويل المقطع لصيغة Gemini ({run.last_line()})")
    info = probe_media(tmp, timeout=60)
    if not info.get("ok") or not (info.get("counts") or {}).get("video"):
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError("الملف المحول غير صالح")
    os.replace(tmp, out)
    video = info.get("video") or {}
    if truncated:
        log(f"✂️ المقطع {os.path.basename(src)} أطول من {cap:.0f} ثانية، أخذنا أول {cap:.0f} ثانية فقط")
    return {
        "path": out,
        "seconds": round(_probe_dur(info), 3),
        "bytes": os.path.getsize(out),
        "width": video.get("width"),
        "height": video.get("height"),
        "truncated": truncated,
        "has_audio": bool((info.get("counts") or {}).get("audio")),
    }


def waveform_peaks(feats: Optional[Features], t0: float, t1: float, n: int = 1200) -> list:
    """Loudness envelope of [t0, t1] for the Studio timeline: n values in 0..1 (bucket max of
    the 10 Hz RMS track, scaled between the window's noise floor and its loud peaks).
    All zeros when features are missing. No audio is decoded."""
    n = int(clamp(int(n or 1200), 10, 5000))
    t0, t1 = float(t0), float(t1)
    if feats is None or not len(feats.rms_db) or t1 <= t0:
        return [0.0] * n
    a, b = feats.range10(max(0.0, t0), max(t0 + feats.hop10, t1))
    seg = feats.rms_db[a:b].astype(np.float32)
    if not len(seg):
        return [0.0] * n
    floor = feats.floor_db[a:b].astype(np.float32)
    lo = float(min(np.percentile(seg, 5), np.median(floor) if len(floor) else np.percentile(seg, 5)))
    hi = float(max(np.percentile(seg, 99.5), lo + 20.0))
    scaled = np.clip((seg - lo) / (hi - lo), 0.0, 1.0)
    span = t1 - t0
    out = []
    first_t = a * feats.hop10
    for i in range(n):
        b0 = t0 + span * i / n
        b1 = t0 + span * (i + 1) / n
        ia = int(clamp(math.floor((b0 - first_t) / feats.hop10), 0, len(scaled) - 1))
        ib = int(clamp(math.ceil((b1 - first_t) / feats.hop10), ia + 1, len(scaled)))
        out.append(round(float(scaled[ia:ib].max()), 3))
    return out


# ============================================================================
# Critic audio snippets and A/V offset guard
# ============================================================================

def audio_snippet(audio_path: str, start: float, end: float, kbps: int = 32, sr: int = 16000) -> dict:
    """Mono MP3 bytes of [start, end] for the multimodal critic (gateway content format)."""
    start = max(0.0, float(start))
    dur = max(0.5, float(end) - start)
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
           "-i", audio_path, "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "libmp3lame", "-b:a", f"{kbps}k",
           "-f", "mp3", "-"]
    res = run_command(cmd, timeout=60, text=False)
    if not res.ok or not res.stdout:
        raise RuntimeError(f"تعذر قص الصوت للناقد ({res.last_error_line()})")
    return {"data": bytes(res.stdout), "mime_type": "audio/mp3", "seconds": round(dur, 2)}


def _xcorr_lag(a: np.ndarray, b: np.ndarray, sr: int) -> tuple:
    """Lag k (samples) where a best matches b[k:k+len(a)], plus a quality score 0..1."""
    a = a.astype(np.float64) - float(np.mean(a))
    b = b.astype(np.float64) - float(np.mean(b))
    n = 1 << int(math.ceil(math.log2(len(a) + len(b))))
    corr = np.fft.irfft(np.fft.rfft(b, n) * np.conj(np.fft.rfft(a, n)), n)
    max_lag = len(b) - len(a)
    if max_lag <= 0:
        return 0, 0.0
    window = corr[:max_lag + 1]
    k = int(np.argmax(window))
    denom = math.sqrt(float(np.sum(a * a)) * float(np.sum(b[k:k + len(a)] ** 2))) or 1.0
    return k, float(window[k] / denom)


def measure_av_offset(stream: StreamInfo, audio_path: str, duration: float,
                      points: Sequence[float] = (0.25, 0.5, 0.75), window_s: float = 20.0,
                      search_s: float = 1.0, sr: int = 8000) -> Optional[float]:
    """Seconds to add to transcript times to land on the same moment in the stream
    (stream_time = transcript_time + offset). None when unmeasurable or inconsistent."""
    offsets = []
    for frac in points:
        t = float(duration) * float(frac)
        if t < search_s + 5:
            continue
        try:
            a = decode_audio(stream.urls[-1], sr=sr, start=t, duration=window_s, timeout=90)
            b = decode_audio(audio_path, sr=sr, start=t - search_s, duration=window_s + 2 * search_s, timeout=90)
        except Exception:
            continue
        if len(a) < sr * window_s * 0.8 or len(b) < len(a) + sr:
            continue
        k, quality = _xcorr_lag(a, b, sr)
        if quality < 0.3:
            continue
        offsets.append(search_s - k / sr)
    if len(offsets) < 2:
        return None
    if max(offsets) - min(offsets) > 0.06:
        return None
    return float(np.median(offsets))


# ============================================================================
# Self test (no network)
# ============================================================================

def _make_sync_hls(folder: str, seconds: int, height: int, fps: int) -> str:
    m3u8 = os.path.join(folder, "index.m3u8")
    if os.path.isfile(m3u8):
        return m3u8
    ensure_dir(folder)
    width = int(round(height * 16 / 9 / 2) * 2)
    vsrc = (f"color=black:s={width}x{height}:r={fps},"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='between(mod(t\\,1)\\,0.5\\,{0.5 + 0.6 / fps:.4f})'")
    asrc = "aevalsrc='if(between(mod(t\\,1)\\,0.5\\,0.52)\\,0.8*sin(2*PI*1000*t)\\,0)':s=48000"
    res = run_command(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", vsrc, "-f", "lavfi", "-i", asrc,
                       "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-g", str(fps * 2),
                       "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-f", "hls", "-hls_time", "6",
                       "-hls_playlist_type", "vod", m3u8], timeout=900)
    if not res.ok:
        raise RuntimeError(f"تعذر تجهيز بث الاختبار ({res.last_error_line()})")
    return m3u8


def _flash_and_beep_times(path: str, fps: float) -> tuple:
    v = run_command(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=16:9,format=gray", "-f", "rawvideo", "-"],
                    timeout=300, text=False)
    fr = np.frombuffer(v.stdout or b"", np.uint8)
    flashes = np.array([])
    if len(fr) >= 144:
        fr = fr[:len(fr) // 144 * 144].reshape(-1, 144).mean(axis=1)
        flashes = (np.where((fr[1:] > 128) & (fr[:-1] <= 128))[0] + 1) / fps
    a = run_command(["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", "48000", "-f", "s16le", "-"],
                    timeout=300, text=False)
    s = np.abs(np.frombuffer(a.stdout or b"", np.int16).astype(np.float32) / 32768.0)
    beeps = np.array([])
    if len(s):
        env = np.convolve(s, np.ones(48) / 48.0, mode="same") > 0.2
        beeps = np.where(env[1:] & ~env[:-1])[0] / 48000.0
    return flashes, beeps


def _selftest_features(duration: float) -> Features:
    n10 = int(duration / HOP10)
    rms = np.full(n10, -50.0, dtype=np.float32)
    floor = np.full(n10, -50.0, dtype=np.float32)
    speech = np.zeros(n10, dtype=np.uint8)
    feats = Features(rms_db=rms, floor_db=floor, speech=speech, events={}, tagger="test")
    for name in EVENT_NAMES:
        feats.events[name] = np.zeros(feats.n1, dtype=np.float32)

    def speak(s: float, e: float) -> None:
        a, b = int(s / HOP10), int(e / HOP10)
        rms[a:b] = -30.0
        speech[a:b] = 1

    for s, e in ((2.0, 6.0), (8.0, 12.0), (14.0, 20.0), (30.0, 34.0), (38.0, 40.0)):
        speak(s, e)
    feats.events["laugh"][20:23] = 0.8          # laugh right after the last word of piece 1
    rms[200:230] = -28.0
    feats.events["laugh"][12:14] = 0.0
    rms[int(34.5 / HOP10):int(37.0 / HOP10)] = -48.0   # dead air 34.5 to 37.0
    rms[int(12.3 / HOP10):int(13.7 / HOP10)] = -46.0   # soft giggle 12.3 to 13.7
    feats.events["laugh"][12:14] = 0.3
    return feats


def _selftest_words() -> list:
    words = []
    for s, e in ((2.0, 6.0), (8.0, 12.0), (14.0, 20.0), (30.0, 34.0), (38.0, 40.0)):
        t = s
        while t + 0.4 <= e + 1e-6:
            words.append({"w": "كلمة", "s": round(t, 2), "e": round(t + 0.35, 2), "p": 0.9})
            t += 0.5
    return words


def selftest(workdir: Optional[str] = None, height: int = 1080, fps: int = 60, keep_files: bool = False) -> dict:
    """Offline checks with synthetic sources (no Twitch, no Gemini):
      1. cutter memory on the clip 4 layout (three pieces over a 70 s span), under 1200 MB
      2. output duration equals frame snapped pieces within one frame
      3. A/V offset under one frame at every seam, and flashes land where the SRT mapper expects
      4. exactly one video and one audio stream
      5. URL refresh after a dead stream URL
      6. SRT: valid, no overlaps, glue words never end a cue unless forced
      7. Cut Doctor: ending saved, giggle kept, dead air shaved, no seam inside a word
      8. v3 protected beats: a silent setup is never snapped away or shaved, and a beat that
         ends just before a piece pulls the piece back over it
      9. v3 window media: Critic and preview files from one pass with the right durations,
         frame rate and height; the proxy falls back to the master when it keeps failing
     10. v3 StreamSet: dead master and proxy URLs recover with one shared refresh, and
         render_pieces writes the clip and its SRT
     11. v3 Gemini transcode (portrait, short side capped, low fps) and waveform peaks
    Returns a dict of results with an "all_ok" flag."""
    work = ensure_dir(workdir or os.path.join(TMP_ROOT, "selftest"))
    results: dict = {}
    log("🧪 فحص ذاتي لمحرك القص والسبتايتل (مصادر صناعية بدون إنترنت)...")
    t0 = time.time()

    def mark(name: str, ok: bool, detail: str) -> None:
        results[name] = {"ok": bool(ok), "detail": detail}
        log(f"   {'✅' if ok else '❌'} {name}: {detail}")

    frame = 1.0 / fps
    try:
        log(f"   ⏳ تجهيز بث اختبار {height}p{fps} مدته 80 ثانية...")
        m3u8 = _make_sync_hls(os.path.join(work, f"hls_{height}p{fps}"), 80, height, fps)
        stream = StreamInfo(vod_url="selftest", urls=[m3u8], fps=float(fps), fps_str=str(fps),
                            width=int(round(height * 16 / 9 / 2) * 2), height=height,
                            refresher=lambda: [m3u8])
        pieces = snap_to_frames([[5.53, 12.21], [21.93, 24.38], [63.71, 75.02]], fps)
        out = os.path.join(work, "selftest_clip.mp4")
        info = cut_and_stitch(stream, pieces, out, allow_refresh=False)
        mark("memory", info["peak_mb"] < 1200, f"أعلى استهلاك ذاكرة {info['peak_mb']:.0f} MB (الحد 1200، الطريقة القديمة كانت تتجاوز 6000)")

        probe = probe_media(out)
        vdur = (probe.get("video") or {}).get("duration") or 0.0
        expected = pieces_net(pieces)
        mark("duration", abs(vdur - expected) <= frame + 1e-3,
             f"مدة الفيديو {vdur:.3f}ث والمتوقع {expected:.3f}ث")
        counts = probe.get("counts", {})
        mark("streams", counts.get("video") == 1 and counts.get("audio") == 1 and sum(counts.values()) == 2,
             f"المسارات {counts}")

        flashes, beeps = _flash_and_beep_times(out, fps)
        offsets = []
        for t in flashes:
            if len(beeps):
                j = int(np.argmin(np.abs(beeps - t)))
                offsets.append((beeps[j] - t) * 1000.0)
        expected_flashes = []
        acc = 0.0
        for s, e in pieces:
            k = math.ceil(s - 0.5)
            while k + 0.5 < e:
                if k + 0.5 >= s:
                    expected_flashes.append(acc + (k + 0.5 - s))
                k += 1
            acc += e - s
        pos_err = [abs(a - b) * 1000.0 for a, b in zip(flashes, expected_flashes)]
        av_ok = bool(offsets) and max(abs(x) for x in offsets) < frame * 1000.0 + 1.0
        pos_ok = len(flashes) == len(expected_flashes) and bool(pos_err) and max(pos_err) <= 2 * frame * 1000.0 + 1.0
        mark("av_sync", av_ok, f"فرق الصوت عن الصورة {max((abs(x) for x in offsets), default=0):.1f} ms في {len(offsets)} نقطة")
        mark("timeline", pos_ok, f"{len(flashes)} من {len(expected_flashes)} ومضة في مكانها، أكبر انحراف {max(pos_err, default=0):.1f} ms")

        dead = StreamInfo(vod_url="selftest", urls=[os.path.join(work, "missing", "index.m3u8")], fps=float(fps),
                          fps_str=str(fps), height=height, refresher=lambda: [m3u8])
        out2 = os.path.join(work, "selftest_refresh.mp4")
        info2 = cut_and_stitch(dead, [[10.0, 12.0]], out2, allow_refresh=True)
        mark("url_refresh", info2["refreshed"], "الرابط الميت تحدث تلقائياً ونجح القص")
        mark("encoder", info["encoder"] in ("h264_nvenc", "libx264"), f"المرمّز {info['encoder']}")
    except Exception as exc:
        mark("cutter", False, f"خطأ: {short_error(exc)}")

    try:
        W = lambda w, s, e: {"w": w, "s": s, "e": e, "p": 0.9}  # noqa: E731
        words = [W("لا!", 10.00, 10.06), W("لا", 10.12, 10.30), W("والله", 10.32, 10.60), W("ما", 10.62, 10.70),
                 W("أبغى", 10.72, 11.00), W("في", 11.02, 11.08), W("الحارة", 11.10, 11.50), W("يا", 11.52, 11.58),
                 W("جماعة", 11.60, 12.00), W("هههههههههه", 14.00, 15.50),
                 W("تخيلوا", 30.00, 30.60), W("مرة", 30.70, 31.10), W("رحت", 31.20, 31.60),
                 W("السوق", 31.70, 32.30), W("وشفت", 32.40, 33.00), W("واحد", 33.10, 33.60)]
        srt_pieces = [[9.9, 15.6], [29.9, 33.8]]
        stats = generate_srt(words, srt_pieces, os.path.join(work, "selftest.srt"), fps=fps)
        cues = parse_srt(stats["text"])
        glue_end = [c[3] for c in cues if _bare(c[3].split()[-1]) in GLUE_WORDS]
        seam_ok = all(not (c[1] < 5.7 < c[2]) for c in cues)
        mark("srt_valid", not stats["errors"], f"{stats['cues']} سطر، أخطاء: {stats['errors'] or 'لا يوجد'}")
        mark("srt_flash", stats["min_cue_s"] >= SRT_MIN_CUE_S and seam_ok,
             f"أقصر سطر {stats['min_cue_s']:.2f}ث، ولا سطر يعبر نقطة القص")
        mark("srt_glue", not glue_end, f"أسطر تنتهي بحرف ربط: {glue_end or 'لا يوجد'}")
    except Exception as exc:
        mark("srt", False, f"خطأ: {short_error(exc)}")

    try:
        feats = _selftest_features(60.0)
        dwords = _selftest_words()
        pcs, rep = cut_doctor([[1.9, 19.9]], dwords, feats, 60.0, fps=fps)
        mark("doctor_ending", pcs and pcs[-1][1] >= 22.9, f"النهاية امتدت إلى {pcs[-1][1]:.2f}ث لتشمل الضحكة")
        pcs2, rep2 = cut_doctor([[7.9, 20.0]], dwords, feats, 60.0, fps=fps)
        giggle_kept = any(s <= 12.6 and e >= 13.4 for s, e in pcs2)
        mark("doctor_giggle", giggle_kept, "الضحكة الخفيفة بين الجمل ما انحذفت")
        dwords2 = [w for w in dwords if not (34.0 < w["s"] < 38.0)]
        pcs3, rep3 = cut_doctor([[29.9, 40.1]], dwords2, feats, 60.0, fps=fps)
        dead_cut = rep3["shaved_sec"] > 1.0 and not any(s < 35.5 < e for s, e in pcs3)
        mark("doctor_dead_air", dead_cut, f"حذف {rep3['shaved_sec']:.2f}ث صمت ميت")
        pcs4, rep4 = cut_doctor([[1.9, 4.17], [8.62, 11.0]], dwords, None, 60.0, fps=fps, auto_shave=False)
        starts = _word_starts(dwords)
        inside = [b for p in pcs4 for b in p if _word_at(dwords, starts, b) is not None]
        mark("doctor_seams", not inside, f"نقاط القص داخل كلمة: {len(inside)}")

        tol = 1.0 / fps + 1e-3
        base, _ = cut_doctor([[24.0, 34.0]], dwords, feats, 60.0, fps=fps)
        prot, prep = cut_doctor([[24.0, 34.0]], dwords, feats, 60.0, fps=fps, protect=[[25.0, 28.0]])
        snap_ok = base[0][0] > 29.0 and prot[0][0] <= 25.0 + tol and prep["protected"] == 1
        mark("doctor_protect_snap", snap_ok,
             f"بدون حماية البداية {base[0][0]:.2f}ث (تنقص اللقطة البصرية)، مع الحماية {prot[0][0]:.2f}ث")
        pcs5, rep5 = cut_doctor([[29.9, 40.1]], dwords2, feats, 60.0, fps=fps, protect=[[35.0, 36.5]])
        shave_ok = any(s <= 35.0 + tol and e >= 36.5 - tol for s, e in pcs5) and rep5["protected"] == 1
        mark("doctor_protect_shave", shave_ok,
             f"الصمت اللي فيه حركة محمية ما انحذف (المحذوف {rep5['shaved_sec']:.2f}ث بدل {rep3['shaved_sec']:.2f}ث)")
        pcs6, rep6 = cut_doctor([[30.0, 34.0]], dwords, feats, 60.0, fps=fps, protect=[[26.5, 28.0], [5.0, 6.0]])
        back_ok = pcs6[0][0] <= 26.5 + tol and rep6["protect_extended_sec"] > 1.0 and rep6["protected"] == 1
        mark("doctor_protect_back", back_ok,
             f"اللقطة رجعت للخلف إلى {pcs6[0][0]:.2f}ث (+{rep6['protect_extended_sec']:.2f}ث) والبعيدة تجاهلناها")
    except Exception as exc:
        mark("doctor", False, f"خطأ: {short_error(exc)}")

    try:
        m3u8 = _make_sync_hls(os.path.join(work, f"hls_{height}p{fps}"), 80, height, fps)
        width = int(round(height * 16 / 9 / 2) * 2)
        dead = os.path.join(work, "missing", "index.m3u8")
        mk = lambda urls: StreamInfo(vod_url="selftest", urls=list(urls), fps=float(fps), fps_str=str(fps),  # noqa: E731
                                     width=width, height=height)
        good_pick = {"urls": [m3u8], "height": height, "width": width, "fps": float(fps), "format_id": "selftest"}
        crit_path = os.path.join(work, "selftest_critic.mp4")
        prev_path = os.path.join(work, "selftest_preview.mp4")
        ss = StreamSet(vod_url="selftest", master=mk([m3u8]), proxy=mk([dead]),
                       resolver=lambda: {"master": good_pick, "proxy": good_pick})
        win = make_window_media(ss, 20.0, 35.0, critic_out=crit_path, preview_out=prev_path, preview_pad_s=5.0)
        cinfo = probe_media(crit_path)
        pinfo = probe_media(prev_path)
        cv, pv = cinfo.get("video") or {}, pinfo.get("video") or {}
        cdur, pdur = _probe_dur(cinfo), _probe_dur(pinfo)
        win_ok = (abs(cdur - 15.0) <= 0.35 and abs(pdur - 25.0) <= 0.35 and abs((cv.get("fps") or 0) - 4.0) < 0.01
                  and (cv.get("height") or 0) == min(360, height) and (pv.get("height") or 0) == min(360, height)
                  and cinfo.get("counts", {}).get("audio") == 1 and pinfo.get("counts", {}).get("audio") == 1
                  and win["refreshed"] and win["source"] == "proxy")
        mark("window_media", win_ok,
             f"الناقد {cdur:.2f}ث بـ {cv.get('fps') or 0:.0f} إطار/ث و{cv.get('height')}p، المعاينة {pdur:.2f}ث بـ {pv.get('height')}p"
             f" ({os.path.getsize(crit_path) / 1024:.0f} KB و {os.path.getsize(prev_path) / 1024:.0f} KB)")

        ss2 = StreamSet(vod_url="selftest", master=mk([m3u8]), proxy=mk([dead]),
                        resolver=lambda: {"master": good_pick, "proxy": dict(good_pick, urls=[dead])})
        win2 = make_window_media(ss2, 40.0, 46.0, critic_out=crit_path, preview_out=None)
        mark("window_fallback", win2["source"] == "master" and abs(win2["seconds"] - 6.0) <= 0.35,
             f"المعاينة المنخفضة ميتة، استخدمنا الجودة الأساسية ({win2['seconds']:.2f}ث)")

        ss3 = StreamSet(vod_url="selftest", master=mk([dead]), proxy=mk([dead]),
                        resolver=lambda: {"master": good_pick, "proxy": good_pick})
        out3 = os.path.join(work, "selftest_render.mp4")
        srt3 = os.path.join(work, "selftest_render.srt")
        rwords = [{"w": "تجربة", "s": 10.2, "e": 10.6, "p": 0.9}, {"w": "القص", "s": 10.8, "e": 11.4, "p": 0.9}]
        rinfo = render_pieces(ss3, [[10.0, 12.0]], out3, srt3, rwords, max_height=1080)
        rs_ok = (rinfo["refreshed"] and ss3.proxy.urls == [m3u8] and ss3.master.urls == [m3u8]
                 and os.path.isfile(out3) and os.path.isfile(srt3) and not rinfo["srt"]["errors"])
        mark("streamset_refresh", rs_ok, "الروابط الميتة (الأساسية والمعاينة) تحدثت مع بعض، والقص والسبتايتل نجحوا")
    except Exception as exc:
        mark("window", False, f"خطأ: {short_error(exc)}")

    try:
        portrait = os.path.join(work, "selftest_portrait_src.mp4")
        res = run_command(["ffmpeg", "-y", "-hide_banner", "-v", "error", "-f", "lavfi", "-i",
                           "testsrc2=s=720x1280:r=30", "-f", "lavfi", "-i", "sine=f=440:r=48000",
                           "-t", "12", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", portrait], timeout=300)
        if not res.ok:
            raise RuntimeError(res.last_error_line())
        tinfo = transcode_for_gemini(portrait, os.path.join(work, "selftest_gemini.mp4"), fps=3.0,
                                     short_side=480, max_seconds=10.0)
        tprobe = probe_media(tinfo["path"])
        tfps = (tprobe.get("video") or {}).get("fps") or 0.0
        t_ok = (tinfo["width"] == 480 and tinfo["height"] == 854 and abs(tfps - 3.0) < 0.01
                and tinfo["truncated"] and abs(tinfo["seconds"] - 10.0) <= 0.4 and tinfo["has_audio"])
        mark("gemini_transcode", t_ok,
             f"{tinfo['width']}x{tinfo['height']} بـ {tfps:.0f} إطار/ث، {tinfo['seconds']:.1f}ث، {tinfo['bytes'] / 1024:.0f} KB")
        peaks = waveform_peaks(_selftest_features(60.0), 0.0, 60.0, n=120)
        w_ok = len(peaks) == 120 and max(peaks) > 0.8 and peaks[100] < 0.2 and min(peaks) >= 0.0
        mark("waveform", w_ok, f"{len(peaks)} قيمة، أعلى {max(peaks):.2f} ووقت الصمت {peaks[100]:.2f}")
    except Exception as exc:
        mark("gemini_transcode", False, f"خطأ: {short_error(exc)}")

    all_ok = all(v["ok"] for v in results.values())
    results["all_ok"] = all_ok
    log(f"{'🎉 كل الفحوصات ناجحة' if all_ok else '⚠️ فيه فحوصات فشلت'} (خلال {time.time() - t0:.0f} ثانية)")
    if not keep_files:
        for name in ("selftest_clip.mp4", "selftest_refresh.mp4", "selftest.srt", "selftest_critic.mp4",
                     "selftest_preview.mp4", "selftest_render.mp4", "selftest_render.srt",
                     "selftest_portrait_src.mp4", "selftest_gemini.mp4"):
            path = os.path.join(work, name)
            if os.path.exists(path):
                os.remove(path)
    return results


# === END OF media_engine.py ===
