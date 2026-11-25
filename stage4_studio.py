"""
stage4_studio.py
Stage 4 of TwitchClipper (v3): the Gradio Review, Edit and Self Learning Studio (Colab CPU).

Tabs
  1. ✂️ المراجعة والمحرر
     * VOD picker and ◀ ▶ clip navigation with status badges
       (👍 / 👎 / ⏳ rating, 📝 draft or ✅ ready, 👀 visual moment, ⚠️ no Critic, 🗑️ deleted)
     * CapCut style timeline editor (studio_timeline) on the clip preview: waveform from the
       stored loudness, laugh track, word chips, protected visual beats, virtual jump cut
       playback. Falls back to the v2 controls (piece checkboxes + 0.5 s nudges) when the
       installed Gradio has no custom HTML components or the clip has no preview.
     * Approving an edit updates the segments, rewrites the SRT and CapCut notes at once,
       records manual_trims for taste learning and marks the clip as a draft (no render yet)
     * 🎬 render this clip at 1080p, and 🎬 render every liked clip in one click (live
       progress). Renders use render_pieces with versioned file names (no stale browser cache)
     * SRT editor (validated before saving) and regeneration, rating, 0 to 10 score, quick
       tags, note; the numeric taste profile retrains instantly
  2. 🗂️ المرشحون: rejected and unselected candidates of the latest Stage 3 run with Critic
     scores, reasons, visual event and preview; ⬆️ promote one into a draft clip
  3. 🧠 الذوق والقواعد: taste summary (with the Golden priors, fade_n=35), retrain, rules
     editor, Gemini proposal with diff, 📌 locked lines guard, timestamped backup and restore
  4. 🌟 اللقطات الذهبية: summary, include or exclude analyzed clips, train or redistill
  5. 🩺 صحة Gemini: usage, model health, refresh, reset
  6. 📦 التصدير والتنظيف: ZIP (MP4 stored, SRT and notes deflated) with a clear warning
     when liked clips are still drafts; guarded Drive cleanup through common.safe_delete
     that never touches TwitchClipper/memory

Public contract
  launch_studio(vod_dir=None, drive_root=DRIVE_ROOT_DEFAULT, api_key=None, share=True,
                server_port=None) -> gradio Blocks
"""

from __future__ import annotations

import difflib
import glob
import inspect
import math
import os
import threading
import time
import zipfile
from typing import Any, Optional

import gradio as gr

from common import (
    DRIVE_ROOT_DEFAULT,
    RIYADH_TZ,
    TMP_ROOT,
    MemoryPaths,
    VodPaths,
    allocate_clip_id,
    atomic_write_json,
    atomic_write_text,
    clip_needs_render,
    coerce_number,
    dir_size_bytes,
    ensure_dir,
    find_clip,
    format_duration_ar,
    human_size,
    list_vods,
    load_feedback,
    load_json_safe,
    load_manifest,
    log,
    make_clip_uid,
    normalize_clip,
    now_iso,
    now_utc,
    read_text_safe,
    safe_delete,
    sanitize_filename,
    save_feedback,
    save_manifest,
    sec_to_hms,
    short_error,
    upsert_feedback,
)
from gemini_gateway import GatewayError, GeminiGateway, ParseFailure, QuotaPause
from golden_engine import golden_stats, golden_summary_ar, load_golden, set_excluded, train_golden
from media_engine import (
    build_phrases,
    collect_words,
    cut_doctor,
    drop_pieces,
    generate_srt,
    load_features,
    make_window_media,
    pieces_net,
    render_pieces,
    resolve_streams,
    trim_pieces,
    validate_srt,
    waveform_peaks,
)
from stage3_clipper import list_run_candidates, load_av_offset, write_clip_notes
from studio_timeline import (
    COMMIT_EVENT,
    build_bridge,
    build_timeline,
    empty_timeline_value,
    file_url,
    parse_timeline_edit,
    timeline_supported,
    timeline_value,
)
from taste_engine import (
    apply_golden_priors,
    backfill_legacy,
    build_signals,
    compute_clip_features,
    doctor_prefs,
    load_profile,
    save_profile,
    taste_summary_ar,
    train_profile,
)


# ============================================================================
# Constants
# ============================================================================

POSITIVE_TAGS = [
    "ضحك هستيري",
    "ردة فعل قوية",
    "التوتر والخوف مضحك",
    "طقطقة رهيبة مع الشات",
    "القص الذكي (Jump Cuts) مضبوط",
    "البداية قوية",
    "النهاية مضبوطة",
    "اللحظة البصرية واضحة",
]
NEGATIVE_TAGS = [
    "كلام عادي ما يستاهل نشر",
    "طويلة ومملة / فيها سكوت",
    "القصّ بدأ متأخر أو انقطع بدري",
    "قصّ الضحكة أو الذروة",
    "سياق ناقص",
    "تشتتت وضاعت السالفة",
    "الصوت العالي من اللعبة مو منها",
    "ما صار شي",
    "الحدث اللي على الشاشة انقص",
]
RATING_CHOICES = [("⏳ بدون تقييم", "pending"), ("👍 عجبني", "liked"), ("👎 ما عجبني", "disliked")]
RATING_BADGE = {"liked": "👍", "disliked": "👎", "pending": "⏳"}
STATUS_BADGE = {"draft": "📝", "ready": "✅", "rendering": "⏳", "failed": "❌", "media_deleted": "🗑️"}
STATUS_AR = {"draft": "مسودة (ما انرندرت 1080p)", "ready": "جاهزة", "rendering": "جاري الرندر",
             "failed": "فشل الرندر", "media_deleted": "ملفاتها انحذفت"}
CANDIDATE_FILTERS = [("الكل", "all"), ("المرفوضة ❌", "rejected"), ("بدون ناقد ⏳", "unreviewed"),
                     ("مقبولة وما اختيرت ✅", "kept")]
CANDIDATE_STATES = {"rejected": "❌ مرفوضة", "unreviewed": "⏳ ما وصلت للناقد", "new": "⏳ ما انحكم عليها",
                    "kept": "✅ قبلها الناقد وما اختيرت", "selected": "🎞️ صارت لقطة", "promoted": "⬆️ مترقية"}
RUBRIC_AR = {"hook": "البداية", "payoff": "القفلة", "reaction": "ردة الفعل", "context": "السياق",
             "pacing": "الإيقاع", "visual": "الشاشة"}
RENDER_HEIGHTS = [("1080p (الأصل)", 1080), ("720p (أسرع)", 720)]
PIN = "📌"
NUDGE = 0.5
GOLDEN_FADE_N = 35
PREVIEW_PAD_S = 15.0
PREVIEW_MAX_SPAN_S = 150.0
WAVE_POINTS = 1200
MIN_PIECE_S = 0.2
TIMELINE_ELEM = "tc_timeline"


# ============================================================================
# Gradio compatibility helpers (Gradio 5 and 6)
# ============================================================================

def _mk(cls: Any, **kwargs: Any) -> Any:
    """Create a component with only the keyword arguments the installed Gradio accepts."""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return cls(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return cls(**accepted)


def _on(listener: Any, fn: Any, **kwargs: Any) -> Any:
    """Attach an event listener with only the keyword arguments the installed Gradio accepts
    (concurrency_id and concurrency_limit are Gradio 4+)."""
    try:
        params = inspect.signature(listener).parameters
        if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            kwargs = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        pass
    try:
        return listener(fn, **kwargs)
    except TypeError:
        for key in ("concurrency_id", "concurrency_limit"):
            kwargs.pop(key, None)
        return listener(fn, **kwargs)


def _blocks(title: str) -> Any:
    try:
        return gr.Blocks(title=title)
    except TypeError:
        return gr.Blocks()


def _launch(demo: Any, **kwargs: Any) -> None:
    try:
        params = inspect.signature(demo.launch).parameters
        accepted = {k: v for k, v in kwargs.items() if k in params}
    except (TypeError, ValueError):
        accepted = kwargs
    demo.launch(**accepted)


def _stamp() -> str:
    return now_utc().astimezone(RIYADH_TZ).strftime("%Y%m%d_%H%M%S")


# ============================================================================
# Per VOD cache
# ============================================================================

class _VodData:
    """Lazy per VOD data: words, audio features, signals, shared streams and A/V offset."""

    def __init__(self, vp: VodPaths):
        self.vp = vp
        self._words: Optional[list] = None
        self._duration = 0.0
        self._feats = None
        self._feats_loaded = False
        self._signals = None
        self._streams = None
        self._offset: Optional[float] = None
        self.lock = threading.RLock()
        self.meta = load_json_safe(vp.meta, default={}, expected_types=(dict,), quiet=True) or {}

    @property
    def vod_id(self) -> str:
        return self.vp.vod_id

    @property
    def vod_url(self) -> str:
        return self.meta.get("vod_url") or f"https://www.twitch.tv/videos/{self.vp.vod_id}"

    @property
    def words(self) -> list:
        if self._words is None:
            tr = load_json_safe(self.vp.transcript, default={}, expected_types=(dict,), quiet=True) or {}
            self._words = collect_words(tr)
            self._duration = float(coerce_number(tr.get("duration"), 0.0) or 0.0)
        return self._words

    @property
    def feats(self):
        if not self._feats_loaded:
            try:
                self._feats = load_features(self.vp, words=self.words, allow_build=False)
            except Exception as exc:
                log(f"⚠️ تعذر تحميل الخصائص الصوتية: {short_error(exc, 100)}")
                self._feats = None
            self._feats_loaded = True
        return self._feats

    @property
    def vod_duration(self) -> float:
        _ = self.words
        dur = self._duration or 0.0
        if self.feats is not None:
            dur = max(dur, self.feats.duration)
        return dur or (self.words[-1]["e"] if self.words else 0.0)

    @property
    def signals(self):
        if self._signals is None:
            chat = load_json_safe(self.vp.chat, default=[], expected_types=(list,), quiet=True) or []
            phrases = build_phrases(self.words, self.feats, vod_duration=self.vod_duration)
            self._signals = build_signals(self.feats, self.words, phrases, chat, self.vod_duration)
        return self._signals

    def streams(self, max_height: int = 1080):
        with self.lock:
            if self._streams is None or int(getattr(self._streams, "max_height", max_height)) != int(max_height):
                self._streams = resolve_streams(self.vod_url, max_height=int(max_height))
            return self._streams

    def av_offset(self, streams=None) -> float:
        """Cached offset from Stage 3; measured once on the master stream when missing."""
        if self._offset is not None:
            return self._offset
        value = float(load_av_offset(self.vp, streams, self.vod_duration) or 0.0)
        if os.path.isfile(os.path.join(self.vp.stage3_dir, "av_offset.json")):
            self._offset = value
        return value

    def laugh_spans(self, t0: float, t1: float) -> list:
        feats = self.feats
        if feats is None:
            return []
        try:
            arr = feats.event("laugh")
        except Exception:
            arr = None
        if arr is None or not len(arr):
            return []
        hop = float(getattr(feats, "hop1", 1.0) or 1.0)
        a = max(0, int(math.floor(t0 / hop)))
        b = min(len(arr), int(math.ceil(t1 / hop)))
        out = []
        for i in range(a, b):
            v = float(arr[i])
            if v >= 0.15:
                out.append([i * hop, (i + 1) * hop, round(v, 3)])
        return out

    def words_in(self, t0: float, t1: float) -> list:
        return [{"w": w["w"], "s": w["s"], "e": w["e"]} for w in self.words if w["e"] >= t0 and w["s"] <= t1]


# ============================================================================
# Studio
# ============================================================================

class Studio:
    def __init__(self, drive_root: str, api_key: Optional[str]):
        self.drive_root = drive_root
        self.api_key = api_key
        self.memory = MemoryPaths(drive_root).ensure()
        self._vods: dict = {}
        self._gw: Optional[GeminiGateway] = None
        self.pending_rules: Optional[str] = None
        self.lock = threading.RLock()
        self.timeline_ok = timeline_supported()

    # ------------------------------------------------------------ basics

    def gateway(self) -> Optional[GeminiGateway]:
        if self._gw is None and self.api_key and len(str(self.api_key)) >= 10:
            try:
                self._gw = GeminiGateway(self.api_key, self.memory.gemini_state, verbose=True)
            except GatewayError as exc:
                log(str(exc))
                self._gw = None
        return self._gw

    def vod(self, vod_id: str) -> Optional[_VodData]:
        if not vod_id:
            return None
        if vod_id not in self._vods:
            vp = VodPaths.from_ids(self.drive_root, vod_id)
            if not os.path.isdir(vp.vod_dir):
                return None
            self._vods[vod_id] = _VodData(vp)
        return self._vods[vod_id]

    def vod_choices(self) -> list:
        out = []
        for v in list_vods(self.drive_root):
            dur = f" | {v['duration_min']:.0f} دقيقة" if v.get("duration_min") else ""
            drafts = f" | 📝 {v['drafts']} مسودة" if v.get("drafts") else ""
            out.append((f"🎥 {v['vod_id']} | {v['clips']} لقطة{drafts}{dur}", v["vod_id"]))
        return out

    @staticmethod
    def clip_label(c: dict) -> str:
        badge = RATING_BADGE.get(c.get("rating", "pending"), "⏳")
        status = c.get("status", "ready")
        state = STATUS_BADGE.get(status, "")
        if status == "ready" and clip_needs_render(c):
            state = "📝"
        eye = " 👀" if c.get("visual_event") else ""
        warn = " ⚠️" if c.get("reviewed") is False else ""
        promoted = " ⬆️" if c.get("source") == "promoted" else ""
        return (f"{badge}{state} #{int(c.get('id') or 0):02d} {c.get('title', 'لقطة')} "
                f"({float(c.get('duration') or 0):.0f}ث){eye}{warn}{promoted}")

    def clip_choices(self, vod_id: str) -> list:
        vd = self.vod(vod_id)
        if vd is None:
            return []
        manifest = load_manifest(vd.vp)
        clips = sorted(manifest.get("clips") or [], key=lambda c: float(c.get("start") or 0))
        return [(self.clip_label(c), c["uid"]) for c in clips]

    def get_clip(self, vod_id: str, uid: str) -> tuple:
        vd = self.vod(vod_id)
        if vd is None or not uid:
            return None, None, None
        manifest = load_manifest(vd.vp)
        _, clip = find_clip(manifest, uid=uid)
        return vd, manifest, clip

    def update_clip(self, vd: _VodData, uid: str, fn) -> Optional[dict]:
        """Load the manifest, change one clip with fn(clip) and save, under a lock, so a long
        render never overwrites a rating saved meanwhile."""
        with vd.lock:
            manifest = load_manifest(vd.vp)
            idx, clip = find_clip(manifest, uid=uid)
            if clip is None:
                return None
            clip = dict(clip)
            fn(clip)
            manifest["clips"][idx] = normalize_clip(clip, vd.vod_id)
            save_manifest(vd.vp, manifest)
            return manifest["clips"][idx]

    @staticmethod
    def feedback_for(entries: list, clip: dict) -> Optional[dict]:
        for e in entries:
            if e.get("uid") == clip.get("uid"):
                return e
        key = clip.get("legacy_key")
        if key:
            for e in entries:
                if e.get("uid") == f"legacy:{key}":
                    return e
        return None

    def _clip_features(self, vd: _VodData, clip: dict, pieces: list) -> dict:
        try:
            return compute_clip_features(pieces, vd.signals, critic=(clip.get("scores") or {}).get("critic"),
                                         category=clip.get("category"))
        except Exception:
            return dict(clip.get("features") or {})

    @staticmethod
    def _base_name(clip: dict, version: Optional[int] = None) -> str:
        base = f"clip_{int(clip.get('id') or 0):02d}_{sanitize_filename(clip.get('title'))}"
        return f"{base}_v{version}" if version else base

    def _paths_for(self, vd: _VodData, clip: dict) -> tuple:
        base = self._base_name(clip)
        srt = clip.get("srt_path") or os.path.join(vd.vp.clips_dir, base + ".srt")
        notes = clip.get("notes_path") or os.path.join(vd.vp.clips_dir, base + "_notes.txt")
        return srt, notes

    # ------------------------------------------------------------ timeline data

    def preview_window(self, clip: dict) -> Optional[list]:
        win = clip.get("preview_window") or []
        path = clip.get("preview_path") or ""
        if len(win) != 2 or not path or not os.path.isfile(path):
            return None
        try:
            w0, w1 = float(win[0]), float(win[1])
        except (TypeError, ValueError):
            return None
        return [w0, w1] if w1 - w0 >= 1.0 else None

    def timeline_for(self, vd: _VodData, clip: dict) -> dict:
        if clip is None:
            return empty_timeline_value("اختر لقطة عشان يظهر محرر القص")
        win = self.preview_window(clip)
        if win is None:
            return empty_timeline_value("ما فيه معاينة لهذي اللقطة. اضغط «🎞️ جهّز معاينة للمحرر» أو استخدم التحكم العادي")
        w0, w1 = win
        segs = clip.get("segments") or []
        beats = [list(b) for b in clip.get("visual_beats") or [] if isinstance(b, (list, tuple)) and len(b) >= 2]
        if not beats:
            beats = [[s, e, "لحظة محمية"] for s, e in clip.get("protect") or []]
        fps = (clip.get("render") or {}).get("fps")
        return timeline_value(file_url(clip["preview_path"]), [w0, w1], segs, vd.words_in(w0, w1),
                              waveform_peaks(vd.feats, w0, w1, n=WAVE_POINTS), vd.laugh_spans(w0, w1),
                              beats, fps, int(coerce_number(clip.get("edit_rev"), 0) or 0))

    # ------------------------------------------------------------ views

    def _empty_view(self, message: str) -> tuple:
        return (None, message, gr.update(choices=[], value=[]), "", "pending", None, [], "", 0.0, 0.0,
                empty_timeline_value(message), gr.update(visible=self.timeline_ok), gr.update(visible=not self.timeline_ok))

    def view(self, vod_id: str, uid: str) -> tuple:
        """Outputs: video, info, pieces, srt, rating, score, tags, note, start delta, end delta,
        timeline value, timeline column visibility, fallback column visibility."""
        vd, _manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return self._empty_view("ℹ️ اختر بث ولقطة من القائمة.")
        has_video = bool(clip.get("video_path")) and os.path.isfile(clip["video_path"])
        stale = clip.get("status") in ("draft", "rendering", "failed")
        win = self.preview_window(clip)
        use_timeline = self.timeline_ok and win is not None
        if has_video and not stale:
            video = clip["video_path"]
        elif not use_timeline and win is not None:
            video = clip["preview_path"]
        elif has_video:
            video = clip["video_path"]
        else:
            video = None
        srt_text = read_text_safe(clip.get("srt_path") or "", "") if clip.get("srt_path") else ""
        segs = clip.get("segments") or []
        choices = [(f"قطعة {i + 1}: {sec_to_hms(s)} إلى {sec_to_hms(e)} ({e - s:.1f}ث)", str(i)) for i, (s, e) in enumerate(segs)]
        pieces = gr.update(choices=choices, value=[str(i) for i in range(len(segs))])
        tags = [t for t in clip.get("tags") or [] if t in POSITIVE_TAGS + NEGATIVE_TAGS]
        tl_value = self.timeline_for(vd, clip) if self.timeline_ok else empty_timeline_value("")
        return (video, self.info_md(clip, use_timeline, video), pieces, srt_text, clip.get("rating", "pending"),
                clip.get("score10"), tags, clip.get("note", ""), 0.0, 0.0, tl_value,
                gr.update(visible=use_timeline), gr.update(visible=not use_timeline))

    @staticmethod
    def info_md(clip: dict, use_timeline: bool = False, video: Optional[str] = None) -> str:
        scores = clip.get("scores") or {}
        critic = scores.get("critic") or {}
        status = clip.get("status", "ready")
        if status == "ready" and clip_needs_render(clip):
            status = "draft"
        lines = [f"### {Studio.clip_label(clip)}",
                 f"🏷️ **{clip.get('category', '')}** | {STATUS_BADGE.get(status, '')} {STATUS_AR.get(status, status)} "
                 f"| ⏱️ {float(clip.get('duration') or 0):.1f} ثانية | يبدأ من {sec_to_hms(clip.get('start') or 0)[:8]} "
                 f"| {len(clip.get('segments') or [])} قطعة | محذوف {float(clip.get('removed_sec') or 0):.1f} ثانية"]
        if status in ("draft", "failed", "rendering"):
            lines.append("📝 هذي اللقطة تحتاج رندر 1080p. راجعها في المحرر، وإذا عجبتك اضغط «🎬 رندر هذي اللقطة (1080p)» "
                         "أو قيّمها 👍 واضغط «🎬 رندر كل اللقطات المقبولة (👍)».")
            if video and video == clip.get("video_path"):
                lines.append("ℹ️ الفيديو المعروض نسخة قديمة قبل آخر تعديل.")
        if clip.get("render_error"):
            lines.append(f"❌ آخر محاولة رندر: {clip['render_error']}")
        if clip.get("reviewed") is False:
            lines.append("⚠️ هذي اللقطة ما راجعها الناقد")
        if clip.get("source") == "promoted":
            lines.append("⬆️ رقيتها بنفسك من قائمة المرشحين")
        if clip.get("hook_line"):
            lines.append(f"🪝 جملة الهوك: {clip['hook_line']}")
        if clip.get("visual_event"):
            lines.append(f"👀 على الشاشة: {clip['visual_event']}")
        if critic:
            lines.append("🎯 الناقد: " + " | ".join(f"{RUBRIC_AR.get(k, k)} {v}" for k, v in critic.items()))
        extra = []
        for key, label in (("final", "النهائي"), ("taste", "الذوق"), ("scout", "الكشاف"), ("local", "المؤشرات")):
            val = scores.get(key)
            if val is not None:
                extra.append(f"{label} {val}")
        if extra:
            lines.append("📊 " + " | ".join(str(x) for x in extra))
        for key, label in (("reason", "💡 السبب"), ("context_note", "🧭 السياق"), ("cut_notes", "✂️ ملاحظات القص")):
            if clip.get(key):
                lines.append(f"{label}: {clip[key]}")
        if clip.get("status") == "media_deleted":
            lines.append("🗑️ ملفات هذي اللقطة انحذفت من Drive بعد التنظيف")
        if clip.get("manual_trims"):
            lines.append(f"🔧 عدد التعديلات اليدوية: {len(clip['manual_trims'])}")
        if not use_timeline:
            lines.append("ℹ️ المحرر الزمني غير متاح لهذي اللقطة، استخدم القطع والأزرار تحت.")
        return "\n\n".join(lines)

    # ------------------------------------------------------------ applying edits (no render)

    def _apply_segments(self, vd: _VodData, uid: str, pieces: list, trim_entry: dict) -> Optional[dict]:
        """Save new segments as a draft, rewrite SRT and CapCut notes, record the manual trim
        and update the feedback entry (retrains the taste when the clip is rated)."""
        _m, clip = find_clip(load_manifest(vd.vp), uid=uid)
        if clip is None:
            return None
        net = pieces_net(pieces)
        feats = self._clip_features(vd, clip, pieces)
        srt_path, notes_path = self._paths_for(vd, clip)
        ensure_dir(vd.vp.clips_dir)
        fps = (clip.get("render") or {}).get("fps")
        srt = generate_srt(vd.words, pieces, srt_path, fps=fps)

        def change(c: dict) -> None:
            c.update({
                "segments": pieces, "start": pieces[0][0], "end": pieces[-1][1], "duration": round(net, 2),
                "removed_sec": round(pieces[-1][1] - pieces[0][0] - net, 2), "features": feats,
                "srt_path": srt_path, "notes_path": notes_path,
                "srt_stats": {k: srt.get(k) for k in ("cues", "min_cue_s", "avg_cps", "duration")},
                "edit_rev": int(coerce_number(c.get("edit_rev"), 0) or 0) + 1, "edited_at": now_iso(),
            })
            if c.get("status") != "media_deleted":
                c["status"] = "draft"
            c["manual_trims"] = list(c.get("manual_trims") or []) + [trim_entry]
            c["vod_url"] = c.get("vod_url") or vd.vod_url

        updated = self.update_clip(vd, uid, change)
        if updated is None:
            return None
        try:
            write_clip_notes(updated, updated["notes_path"])
        except Exception as exc:
            log(f"⚠️ تعذر كتابة ملاحظات CapCut: {short_error(exc, 100)}")
        with self.lock:
            entries = load_feedback(self.memory)
            fb = self.feedback_for(entries, updated)
            updated["_retrain_msg"] = ""
            if fb is not None:
                fb = dict(fb)
                fb.update({"segments": pieces, "features": feats, "duration": round(net, 2),
                           "manual_trims": list(fb.get("manual_trims") or []) + [trim_entry], "updated_at": now_iso()})
                entries = upsert_feedback(entries, fb)
                save_feedback(self.memory, entries)
                updated["_retrain_msg"] = self._retrain(entries)
        updated["_srt_cues"] = srt.get("cues", 0)
        return updated

    @staticmethod
    def _trim_entry(old: list, new: list, source: str, extra: Optional[dict] = None) -> dict:
        entry = {"at": now_iso(), "source": source,
                 "start_delta": round(float(new[0][0]) - float(old[0][0]), 3) if old and new else 0.0,
                 "end_delta": round(float(new[-1][1]) - float(old[-1][1]), 3) if old and new else 0.0,
                 "pieces_before": len(old), "pieces_after": len(new),
                 "net_before": round(pieces_net(old), 2) if old else 0.0, "net_after": round(pieces_net(new), 2)}
        if extra:
            entry.update(extra)
        return entry

    def commit_timeline(self, vod_id: str, uid: str, tl_payload: Any, bridge_payload: Any = None) -> tuple:
        """Approve button of the timeline editor."""
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return ("❌ اختر لقطة أول",) + self.view(vod_id, uid)
        win = self.preview_window(clip)
        if win is None:
            return ("❌ هذي اللقطة ما لها معاينة للمحرر",) + self.view(vod_id, uid)
        edit = parse_timeline_edit(tl_payload, win, vd.vod_duration, min_piece_s=MIN_PIECE_S)
        if edit is None or not edit.get("committed"):
            alt = parse_timeline_edit(bridge_payload, win, vd.vod_duration, min_piece_s=MIN_PIECE_S)
            if alt is not None and (edit is None or alt.get("committed")):
                edit = alt
        if edit is None:
            return ("❌ ما وصلنا تعديل صالح من المحرر (يمكن المحرر يعرض لقطة ثانية). اضغط اعتماد مرة ثانية",) + self.view(vod_id, uid)
        current_rev = int(coerce_number(clip.get("edit_rev"), 0) or 0)
        if str(edit.get("rev")) != str(current_rev):
            return ("⚠️ التعديل قديم: اللقطة تغيرت بعد ما فتحت المحرر. رجعنا آخر نسخة، عدّل واعتمد من جديد",) + self.view(vod_id, uid)
        old = [list(p) for p in clip.get("segments") or []]
        new = edit["pieces"]
        if [[round(a, 3), round(b, 3)] for a, b in old] == new:
            return ("ℹ️ ما فيه تغيير على القص",) + self.view(vod_id, uid)
        updated = self._apply_segments(vd, uid, new, self._trim_entry(old, new, "timeline"))
        if updated is None:
            return ("❌ تعذر حفظ التعديل",) + self.view(vod_id, uid)
        msg = (f"✅ اعتمدنا القص ({edit['count']} قطعة، {edit['net']:.1f} ثانية) وكتبنا السبتايتل "
               f"({updated.get('_srt_cues', 0)} سطر) وملاحظات CapCut. اللقطة الحين مسودة تحتاج رندر 1080p")
        if updated.get("_retrain_msg"):
            msg += " | " + updated["_retrain_msg"]
        return (msg,) + self.view(vod_id, uid)

    def apply_fallback(self, vod_id: str, uid: str, keep: list, start_delta: Any, end_delta: Any, snap: bool) -> tuple:
        """v2 style controls: keep ticked pieces, nudge the first start and the last end."""
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return ("❌ اختر لقطة أول",) + self.view(vod_id, uid)
        segs = clip.get("segments") or []
        keep_idx = sorted(int(k) for k in keep or [] if str(k).isdigit() and int(k) < len(segs))
        if not keep_idx:
            return ("❌ لازم تبقي قطعة وحدة على الأقل",) + self.view(vod_id, uid)
        sd = float(coerce_number(start_delta, 0.0) or 0.0)
        ed = float(coerce_number(end_delta, 0.0) or 0.0)
        pieces = drop_pieces(segs, keep_idx)
        pieces = trim_pieces(pieces, sd, ed, words=vd.words, snap_to_words=bool(snap), vod_duration=vd.vod_duration)
        if not pieces:
            return ("❌ التعديل حذف اللقطة كاملة، خفف القص",) + self.view(vod_id, uid)
        if [[round(a, 3), round(b, 3)] for a, b in segs] == [[round(a, 3), round(b, 3)] for a, b in pieces]:
            return ("ℹ️ ما فيه تغيير على القص",) + self.view(vod_id, uid)
        extra = {"removed_pieces": [i for i in range(len(segs)) if i not in keep_idx], "snap": bool(snap),
                 "start_nudge": sd, "end_nudge": ed}
        updated = self._apply_segments(vd, uid, pieces, self._trim_entry(segs, pieces, "controls", extra))
        if updated is None:
            return ("❌ تعذر حفظ التعديل",) + self.view(vod_id, uid)
        msg = (f"✅ طبقنا القص ({len(pieces)} قطعة، {pieces_net(pieces):.1f} ثانية) وكتبنا السبتايتل وملاحظات CapCut. "
               "اللقطة الحين مسودة تحتاج رندر 1080p")
        if updated.get("_retrain_msg"):
            msg += " | " + updated["_retrain_msg"]
        return (msg,) + self.view(vod_id, uid)

    def make_preview(self, vod_id: str, uid: str):
        """Cut a preview window for a clip that has none (legacy or v2 clips)."""
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None:
            yield ("❌ اختر لقطة أول",) + self.view(vod_id, uid)
            return
        segs = clip.get("segments") or []
        if not segs:
            yield ("❌ اللقطة ما فيها قطع",) + self.view(vod_id, uid)
            return
        s, e = float(segs[0][0]), float(segs[-1][1])
        if e - s > PREVIEW_MAX_SPAN_S:
            yield (f"❌ اللقطة ممتدة على أكثر من {PREVIEW_MAX_SPAN_S:.0f} ثانية من البث، استخدم التحكم العادي",) + self.view(vod_id, uid)
            return
        yield ("⏳ نجهز معاينة خفيفة للمحرر من رابط البث... (عادة أقل من دقيقة)",) + self.view(vod_id, uid)
        try:
            streams = vd.streams(1080)
            offset = vd.av_offset(streams)
            out = vd.vp.preview_path(f"studio_{clip['uid']}")
            ensure_dir(vd.vp.previews_dir)
            info = make_window_media(streams, s, e, critic_out=None, preview_out=out, preview_pad_s=PREVIEW_PAD_S,
                                     time_offset=offset)
        except Exception as exc:
            yield (f"❌ تعذر تجهيز المعاينة: {short_error(exc, 200)}",) + self.view(vod_id, uid)
            return
        p0, p1 = info.get("p0"), info.get("p1")

        def change(c: dict) -> None:
            c["preview_path"] = info.get("preview") or out
            c["preview_window"] = [p0, p1]

        self.update_clip(vd, uid, change)
        yield ("✅ المعاينة جاهزة، المحرر الزمني شغال لهذي اللقطة",) + self.view(vod_id, uid)

    # ------------------------------------------------------------ rendering

    def _render_one(self, vd: _VodData, uid: str, max_height: int) -> tuple:
        """Render one clip at full quality. Returns (ok, message)."""
        _m, clip = find_clip(load_manifest(vd.vp), uid=uid)
        if clip is None:
            return False, "اللقطة مو موجودة"
        segs = clip.get("segments") or []
        if not segs:
            return False, "اللقطة ما فيها قطع"
        old_status = clip.get("status", "draft")
        self.update_clip(vd, uid, lambda c: c.update({"status": "rendering"}))
        t0 = time.time()
        try:
            streams = vd.streams(max_height)
            offset = vd.av_offset(streams)
            version = int(coerce_number(clip.get("version"), 1) or 1) + 1
            base = self._base_name(clip, version)
            ensure_dir(vd.vp.clips_dir)
            out_mp4 = os.path.join(vd.vp.clips_dir, base + ".mp4")
            out_srt = os.path.join(vd.vp.clips_dir, base + ".srt")
            info = render_pieces(streams, segs, out_mp4, out_srt, vd.words, time_offset=offset,
                                 max_height=int(max_height), srt_bom=False)
        except Exception as exc:
            err = short_error(exc, 200)
            self.update_clip(vd, uid, lambda c: c.update({"status": "failed" if old_status != "media_deleted" else old_status,
                                                          "render_error": err}))
            return False, err
        pieces = info.get("segments") or segs
        net = pieces_net(pieces)
        old_files = (clip.get("video_path"), clip.get("srt_path"))
        notes_path = clip.get("notes_path") or os.path.join(vd.vp.clips_dir, self._base_name(clip) + "_notes.txt")
        srt = info.get("srt") or {}

        def change(c: dict) -> None:
            c.update({
                "segments": pieces, "start": pieces[0][0], "end": pieces[-1][1], "duration": round(net, 2),
                "removed_sec": round(pieces[-1][1] - pieces[0][0] - net, 2),
                "video_path": out_mp4, "srt_path": out_srt, "notes_path": notes_path, "version": version,
                "status": "ready", "render_error": "", "rendered_at": now_iso(), "vod_url": c.get("vod_url") or vd.vod_url,
                "srt_stats": {k: srt.get(k) for k in ("cues", "min_cue_s", "avg_cps", "duration")},
                "render": {k: info.get(k) for k in ("encoder", "size_mb", "height", "fps", "refreshed", "peak_mb")},
            })

        updated = self.update_clip(vd, uid, change)
        for old in old_files:
            if old and os.path.isfile(old) and old not in (out_mp4, out_srt):
                try:
                    safe_delete(old, self.drive_root)
                except (PermissionError, OSError):
                    pass
        if updated is not None:
            try:
                write_clip_notes(updated, notes_path)
            except Exception as exc:
                log(f"⚠️ تعذر كتابة ملاحظات CapCut: {short_error(exc, 100)}")
        return True, (f"المدة {net:.1f} ث، الحجم {float(info.get('size_mb') or 0):.1f} ميقا، "
                      f"السبتايتل {srt.get('cues', 0)} سطر، الوقت {format_duration_ar(time.time() - t0)}")

    def media_view(self, vod_id: str, uid: str) -> tuple:
        """Video and info only (used after renders so the timeline editor is not reloaded)."""
        full = self.view(vod_id, uid)
        return full[0], full[1]

    def render_clip(self, vod_id: str, uid: str, max_height: Any):
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None:
            yield ("❌ اختر لقطة أول", gr.update(), gr.update(), gr.update())
            return
        height = int(coerce_number(max_height, 1080) or 1080)
        yield (f"⏳ جاري رندر #{int(clip.get('id') or 0):02d} «{clip.get('title')}» بجودة {height}p... "
               "تقدر تكمل مراجعة لقطات ثانية", gr.update(), gr.update(), gr.update())
        ok, detail = self._render_one(vd, uid, height)
        msg = (f"✅ رندر #{int(clip.get('id') or 0):02d} جاهز ({detail})" if ok
               else f"❌ فشل رندر #{int(clip.get('id') or 0):02d}: {detail}")
        video, info = self.media_view(vod_id, uid)
        yield (msg, video, info, gr.update(choices=self.clip_choices(vod_id), value=uid))

    def liked_to_render(self, vd: _VodData) -> list:
        manifest = load_manifest(vd.vp)
        return [c for c in manifest.get("clips") or []
                if c.get("rating") == "liked" and c.get("status") != "media_deleted" and clip_needs_render(c)]

    def render_all_liked(self, vod_id: str, uid: str, max_height: Any):
        vd = self.vod(vod_id)
        if vd is None:
            yield "❌ اختر بث أول", gr.update(), self.export_info(vod_id)
            return
        todo = sorted(self.liked_to_render(vd), key=lambda c: int(c.get("id") or 0))
        if not todo:
            yield ("✅ كل اللقطات اللي عجبتك 👍 جاهزة بجودة كاملة، ما فيه شي يحتاج رندر",
                   gr.update(choices=self.clip_choices(vod_id), value=uid), self.export_info(vod_id))
            return
        height = int(coerce_number(max_height, 1080) or 1080)
        t_start = time.time()
        done, failed = [], []
        lines = [f"🎬 رندر {len(todo)} لقطة عجبتك بجودة {height}p:"]
        for i, c in enumerate(todo, start=1):
            head = f"⏳ [{i}/{len(todo)}] #{int(c.get('id') or 0):02d} «{c.get('title')}»..."
            yield "\n\n".join(lines + [head]), gr.update(), gr.update()
            ok, detail = self._render_one(vd, c["uid"], height)
            if ok:
                done.append(c)
                lines.append(f"✅ #{int(c.get('id') or 0):02d} «{c.get('title')}» ({detail})")
            else:
                failed.append(c)
                lines.append(f"❌ #{int(c.get('id') or 0):02d} «{c.get('title')}»: {detail}")
            yield "\n\n".join(lines), gr.update(choices=self.clip_choices(vod_id), value=uid), gr.update()
        summary = f"🎉 خلصنا خلال {format_duration_ar(time.time() - t_start)}: {len(done)} جاهزة"
        if failed:
            summary += f"، و {len(failed)} فشلت (اضغط الزر مرة ثانية ونعيد اللي فشل فقط)"
        lines.append(summary)
        yield "\n\n".join(lines), gr.update(choices=self.clip_choices(vod_id), value=uid), self.export_info(vod_id)

    # ------------------------------------------------------------ SRT

    def save_srt(self, vod_id: str, uid: str, text: str) -> str:
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None or not clip.get("srt_path"):
            return "❌ اختر لقطة أول"
        errors = validate_srt(text, clip_duration=float(clip.get("duration") or 0) + 0.05)
        if errors:
            shown = "\n".join(f"• {e}" for e in errors[:8])
            return f"❌ ما حفظنا الملف، فيه أخطاء:\n{shown}"
        normalized = str(text).replace("\r\n", "\n").strip() + "\n"
        atomic_write_text(clip["srt_path"], normalized)
        return "✅ تم حفظ السبتايتل على Drive"

    def regen_srt(self, vod_id: str, uid: str) -> tuple:
        vd, _m, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return "❌ اختر لقطة أول", ""
        fps = (clip.get("render") or {}).get("fps")
        path, _notes = self._paths_for(vd, clip)
        ensure_dir(os.path.dirname(path))
        srt = generate_srt(vd.words, clip.get("segments") or [], path, fps=fps)
        if clip.get("srt_path") != path:
            self.update_clip(vd, uid, lambda c: c.update({"srt_path": path}))
        return f"✅ تم توليد السبتايتل من جديد ({srt['cues']} سطر)", srt["text"]

    # ------------------------------------------------------------ ratings and taste

    def _feedback_entry(self, vod_id: str, clip: dict, rating: str, score10: Any, tags: list, note: str,
                        base: Optional[dict], source: str = "") -> dict:
        entry = dict(base or {})
        entry.update({
            "uid": entry.get("uid") or clip["uid"],
            "vod_id": vod_id, "clip_id": clip.get("id"), "title": clip.get("title"),
            "category": clip.get("category"), "duration": clip.get("duration"),
            "removed_sec": clip.get("removed_sec"), "avg_wps": clip.get("avg_wps"),
            "excerpt": clip.get("excerpt"), "ai_reason": clip.get("reason"),
            "rating": rating, "score10": score10, "tags": list(tags or []), "note": str(note or ""),
            "segments": clip.get("segments"), "manual_trims": clip.get("manual_trims") or [],
            "visual_event": clip.get("visual_event", ""), "clip_source": clip.get("source", ""),
            "updated_at": now_iso(), "schema": 2,
        })
        if source:
            entry["source"] = source
        if clip.get("features"):
            entry["features"] = clip["features"]
            entry["features_source"] = "exact"
        return entry

    def save_rating(self, vod_id: str, uid: str, rating: str, score10: Any, tags: list, note: str) -> tuple:
        vd, _manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return "❌ اختر لقطة أول", gr.update(), self.export_info(vod_id)
        score = coerce_number(score10, None)
        score = round(max(0.0, min(10.0, score)), 1) if score is not None else None
        rating = rating if rating in ("liked", "disliked", "pending") else "pending"
        feats = clip.get("features") or self._clip_features(vd, clip, clip.get("segments") or [])

        def change(c: dict) -> None:
            c.update({"rating": rating, "score10": score, "tags": list(tags or []), "note": str(note or ""),
                      "features": feats})

        updated = self.update_clip(vd, uid, change) or clip
        with self.lock:
            entries = load_feedback(self.memory)
            base = self.feedback_for(entries, updated)
            if rating == "pending" and score is None and not tags and not note:
                msg = "✅ تم حفظ اللقطة بدون تقييم"
            else:
                entries = upsert_feedback(entries, self._feedback_entry(vod_id, updated, rating, score, tags, note, base))
                save_feedback(self.memory, entries)
                msg = "✅ تم حفظ تقييمك | " + self._retrain(entries)
        if rating == "liked" and clip_needs_render(updated) and updated.get("status") != "media_deleted":
            msg += " | 📝 اللقطة لسه مسودة، اضغط رندر عشان تطلع 1080p"
        return msg, gr.update(choices=self.clip_choices(vod_id), value=uid), self.export_info(vod_id)

    def _retrain(self, entries: list) -> str:
        try:
            filled, n = backfill_legacy(entries, self.drive_root)
            if n:
                save_feedback(self.memory, filled)
            prev = load_profile(self.memory)
            prof = train_profile(filled, prev)
            save_profile(self.memory, prof)
            return f"🧠 تدرب الذوق على {prof['n']} تقييم"
        except Exception as exc:
            return f"⚠️ تعذر تدريب الذوق: {short_error(exc, 120)}"

    def taste_md(self) -> str:
        profile = load_profile(self.memory)
        golden = load_golden(self.memory)
        stats = golden.get("stats") or golden_stats(golden)
        try:
            profile = apply_golden_priors(profile, stats, fade_n=GOLDEN_FADE_N)
        except Exception as exc:
            log(f"⚠️ تعذر دمج الحمض الذهبي في الملخص: {short_error(exc, 100)}")
        return taste_summary_ar(profile)

    def retrain_now(self) -> tuple:
        with self.lock:
            msg = self._retrain(load_feedback(self.memory))
        return msg, self.taste_md()

    # ------------------------------------------------------------ candidates

    @staticmethod
    def _cand_matches(c: dict, flt: str) -> bool:
        state = c.get("state", "new")
        if state in ("selected", "promoted"):
            return False
        if flt == "rejected":
            return state == "rejected"
        if flt == "unreviewed":
            return state in ("unreviewed", "new")
        if flt == "kept":
            return state == "kept"
        return state in ("rejected", "unreviewed", "new", "kept")

    @staticmethod
    def _cand_critic_mean(c: dict) -> Optional[float]:
        critic = c.get("critic") or {}
        vals = [float(critic[k]) for k in ("hook", "payoff", "reaction", "context", "pacing") if k in critic]
        return round(sum(vals) / len(vals), 1) if vals else None

    def _cand_label(self, c: dict) -> str:
        state = CANDIDATE_STATES.get(c.get("state", "new"), "")
        icon = state.split(" ")[0] if state else "•"
        span = c.get("pieces") or [[0, 0]]
        mean = self._cand_critic_mean(c)
        critic = f" | ناقد {mean}" if mean is not None else ""
        reasons = f" | {'، '.join(c.get('reject') or [])}" if c.get("reject") else ""
        eye = " 👀" if c.get("visual_event") or c.get("needs_visual") else ""
        radar = " 📡" if c.get("source") == "radar" else ""
        return (f"{icon} {sec_to_hms(span[0][0])[:8]} «{c.get('title', 'لقطة')}» {float(c.get('net') or 0):.0f}ث "
                f"| كشاف {c.get('scout_score', 0)}{critic}{reasons}{eye}{radar}")

    def candidate_choices(self, vod_id: str, flt: str) -> tuple:
        vd = self.vod(vod_id)
        if vd is None:
            return "", []
        run_id, cands = list_run_candidates(vd.vp)
        rows = [c for c in cands if self._cand_matches(c, flt or "all")]
        rows.sort(key=lambda c: float((c.get("pieces") or [[0]])[0][0]))
        return run_id, [(self._cand_label(c), f"{run_id}|{c.get('cid')}") for c in rows]

    def candidates_refresh(self, vod_id: str, flt: str) -> tuple:
        run_id, choices = self.candidate_choices(vod_id, flt)
        if not run_id:
            note = "ℹ️ ما فيه تشغيل للمرحلة 3 لهذا البث بعد"
        else:
            note = f"🗂️ التشغيل {run_id}: {len(choices)} مرشح في هذي القائمة"
        first = choices[0][1] if choices else None
        return (gr.update(choices=choices, value=first), note) + self.candidate_view(vod_id, first)

    def _find_candidate(self, vod_id: str, key: str) -> tuple:
        vd = self.vod(vod_id)
        if vd is None or not key or "|" not in str(key):
            return None, None, None, None
        run_id, cid = str(key).split("|", 1)
        _rid, cands = list_run_candidates(vd.vp, run_id)
        for c in cands:
            if c.get("cid") == cid:
                return vd, run_id, cands, c
        return vd, run_id, cands, None

    def candidate_view(self, vod_id: str, key: str) -> tuple:
        vd, run_id, _cands, c = self._find_candidate(vod_id, key)
        if c is None:
            return None, "ℹ️ اختر مرشح من القائمة"
        media = c.get("media") or {}
        preview = media.get("preview") if media.get("preview") and os.path.isfile(str(media.get("preview"))) else None
        pieces = c.get("pieces") or []
        lines = [f"### {self._cand_label(c)}",
                 f"📌 الحالة: {CANDIDATE_STATES.get(c.get('state', 'new'), c.get('state', ''))} | المصدر: "
                 f"{'الرادار 📡' if c.get('source') == 'radar' else 'الكشاف'} | {len(pieces)} قطعة "
                 f"| {float(c.get('net') or 0):.1f} ثانية | تصنيف: {c.get('category', '')}"]
        if c.get("why"):
            lines.append(f"💡 سبب الكشاف: {c['why']}")
        if c.get("context_note"):
            lines.append(f"🧭 السياق: {c['context_note']}")
        if c.get("visual_guess"):
            lines.append(f"🔮 توقع الكشاف للشاشة: {c['visual_guess']}")
        if c.get("visual_event"):
            lines.append(f"👀 الناقد شاف على الشاشة: {c['visual_event']}")
        critic = c.get("critic") or {}
        if critic:
            lines.append("🎯 الناقد: " + " | ".join(f"{RUBRIC_AR.get(k, k)} {v}" for k, v in critic.items()))
        if c.get("reject"):
            lines.append("❌ أسباب الرفض: " + "، ".join(c["reject"]))
        if c.get("voice_check") and c.get("voice_check") != "her":
            lines.append(f"🔊 مصدر الصوت حسب الناقد: {'اللعبة' if c['voice_check'] == 'game' else 'مختلط'}")
        if c.get("cut_notes"):
            lines.append(f"✂️ ملاحظات القص: {c['cut_notes']}")
        if pieces and vd is not None:
            text = " ".join(w["w"] for w in vd.words if any(s <= (w["s"] + w["e"]) / 2 < e for s, e in pieces))
            if text:
                lines.append(f"🗣️ الكلام: {text[:600]}{'…' if len(text) > 600 else ''}")
        if not preview:
            lines.append("ℹ️ ما فيه معاينة لهذا المرشح (الناقد سمع الصوت فقط). بعد الترقية تقدر تجهز معاينة من تبويب المراجعة.")
        return preview, "\n\n".join(lines)

    def promote(self, vod_id: str, key: str, flt: str) -> tuple:
        vd, run_id, cands, c = self._find_candidate(vod_id, key)
        if c is None:
            return ("❌ اختر مرشح أول", gr.update(), gr.update(), "", None, "")
        prev_state = str(c.get("state") or "")
        prev_reject = list(c.get("reject") or [])
        pieces = [list(p) for p in c.get("pieces") or []]
        if not pieces:
            return ("❌ المرشح ما فيه قطع", gr.update(), gr.update(), "", None, "")
        profile = load_profile(self.memory)
        peak = None
        try:
            phrases_path = os.path.join(vd.vp.run_dir(run_id), "phrases.json")
            phrases = load_json_safe(phrases_path, default=[], expected_types=(list,), quiet=True) or []
            for ph in phrases:
                if int(ph.get("id", -1)) == int(c.get("peak_id", -2)):
                    peak = float(ph["s"])
                    break
        except (TypeError, ValueError):
            peak = None
        protect = [[float(a), float(b)] for a, b in (c.get("protect") or [])] or None
        try:
            clean, report = cut_doctor(pieces, vd.words, vd.feats, vd.vod_duration, fps=None, peak_time=peak,
                                       prefs=doctor_prefs(profile), auto_shave=True, protect=protect)
        except Exception:
            clean, report = pieces, {}
        pieces = clean or pieces
        net = pieces_net(pieces)
        media = c.get("media") or {}
        has_preview = bool(media.get("preview")) and os.path.isfile(str(media.get("preview")))
        with vd.lock:
            manifest = load_manifest(vd.vp)
            clip_id = allocate_clip_id(manifest)
            uid = make_clip_uid(vd.vod_id, run_id, clip_id)
            title = str(c.get("title") or "لقطة")
            base = f"clip_{clip_id:02d}_{sanitize_filename(title)}"
            ensure_dir(vd.vp.clips_dir)
            srt_path = os.path.join(vd.vp.clips_dir, base + ".srt")
            notes_path = os.path.join(vd.vp.clips_dir, base + "_notes.txt")
            srt = generate_srt(vd.words, pieces, srt_path)
            temp = {"segments": pieces, "category": c.get("category")}
            feats = self._clip_features(vd, temp, pieces)
            excerpt = " ".join(w["w"] for w in vd.words if any(s <= (w["s"] + w["e"]) / 2 < e for s, e in pieces))[:600]
            clip = normalize_clip({
                "uid": uid, "id": clip_id, "run_id": run_id, "title": title, "category": c.get("category") or "ضحك",
                "segments": pieces, "llm_segments": c.get("pieces") or [], "duration": round(net, 2),
                "removed_sec": round(pieces[-1][1] - pieces[0][0] - net, 2),
                "scores": {"scout": c.get("scout_score"), "critic": c.get("critic"), "local": c.get("local")},
                "reviewed": bool(c.get("critic")), "features": feats, "reason": c.get("why", ""),
                "context_note": c.get("context_note", ""), "cut_notes": c.get("cut_notes", ""),
                "excerpt": excerpt, "video_path": os.path.join(vd.vp.clips_dir, base + ".mp4"),
                "srt_path": srt_path, "notes_path": notes_path, "status": "draft", "source": "promoted",
                "visual_event": c.get("visual_event", ""), "visual_beats": c.get("visual_beats") or [],
                "protect": c.get("protect") or [], "hook_line": c.get("hook_line", ""),
                "preview_path": media.get("preview") if has_preview else "",
                "preview_window": [media.get("p0"), media.get("p1")] if has_preview else [],
                "media_key": c.get("media_key", ""), "cid": c.get("cid"), "vod_url": vd.vod_url,
                "rating": "liked", "score10": 9.0, "tags": ["ترقية من المرشحين"],
                "srt_stats": {k: srt.get(k) for k in ("cues", "min_cue_s", "avg_cps", "duration")},
                "doctor": {k: (report or {}).get(k) for k in ("protected", "protect_extended_sec", "shaved_sec")},
                "promoted_from": {"run_id": run_id, "cid": c.get("cid"), "state": prev_state,
                                  "reject": prev_reject},
                "created_at": now_iso(), "edit_rev": 0,
            }, vd.vod_id)
            manifest["clips"].append(clip)
            save_manifest(vd.vp, manifest)
            for cand in cands:
                if cand.get("cid") == c.get("cid"):
                    cand["state"] = "promoted"
                    cand["promoted_uid"] = uid
            atomic_write_json(os.path.join(vd.vp.run_dir(run_id), "candidates.json"), cands, indent=None, backup=True)
        try:
            write_clip_notes(clip, notes_path)
        except Exception as exc:
            log(f"⚠️ تعذر كتابة ملاحظات CapCut: {short_error(exc, 100)}")
        with self.lock:
            entries = load_feedback(self.memory)
            if prev_state == "rejected":
                reasons = "، ".join(prev_reject) or "بدون سبب مكتوب"
                note = f"رقيتها يدوياً من المرشحين بعد ما رفضها الناقد ({reasons})"
            elif prev_state in ("unreviewed", "new"):
                note = "رقيتها يدوياً من المرشحين وهي ما وصلت للناقد"
            elif prev_state == "kept":
                note = "رقيتها يدوياً من المرشحين بعد ما قبلها الناقد وما اختيرت"
            else:
                note = "رقيتها يدوياً من المرشحين"
            entry = self._feedback_entry(vod_id, clip, "liked", 9.0, ["ترقية من المرشحين"], note, None, source="promoted")
            entries = upsert_feedback(entries, entry)
            save_feedback(self.memory, entries)
            retrain = self._retrain(entries)
        msg = (f"⬆️ رقينا «{title}» إلى لقطة #{clip_id:02d} كمسودة (سبتايتل وملاحظات جاهزة) | {retrain}. "
               "راجعها في تبويب المراجعة، وبعدها رندرها")
        run_id2, choices = self.candidate_choices(vod_id, flt)
        first = choices[0][1] if choices else None
        cand_preview, cand_md = self.candidate_view(vod_id, first)
        return (msg, gr.update(choices=choices, value=first), gr.update(choices=self.clip_choices(vod_id), value=uid),
                f"🗂️ التشغيل {run_id2}: {len(choices)} مرشح في هذي القائمة", cand_preview, cand_md)

    # ------------------------------------------------------------ rules

    def _rules_history(self) -> list:
        files = sorted(glob.glob(os.path.join(self.memory.rules_history_dir, "rules_*.md")), reverse=True)
        return [(os.path.basename(f).replace("rules_", "").replace(".md", ""), f) for f in files]

    def _backup_rules(self) -> Optional[str]:
        current = read_text_safe(self.memory.rules, "")
        if not current.strip():
            return None
        path = os.path.join(ensure_dir(self.memory.rules_history_dir), f"rules_{_stamp()}.md")
        atomic_write_text(path, current)
        return path

    @staticmethod
    def _pinned(text: str) -> list:
        return [line.strip() for line in str(text or "").splitlines() if line.strip().startswith(PIN)]

    def save_rules_manual(self, text: str) -> tuple:
        if not str(text or "").strip():
            return "❌ القواعد فاضية، ما حفظنا شي", gr.update()
        current = read_text_safe(self.memory.rules, "")
        missing = [p for p in self._pinned(current) if p not in [line.strip() for line in str(text).splitlines()]]
        if missing:
            shown = "\n".join(f"• {m}" for m in missing)
            return (f"⚠️ حفظنا القواعد، لكن انتبه: حذفت أسطر كانت مثبتة 📌 (النسخة القديمة محفوظة في السجل):\n{shown}",
                    self._save_rules_text(text))
        return "✅ تم حفظ القواعد (والنسخة القديمة محفوظة في السجل)", self._save_rules_text(text)

    def _save_rules_text(self, text: str) -> Any:
        self._backup_rules()
        atomic_write_text(self.memory.rules, str(text).strip() + "\n")
        return gr.update(choices=self._rules_history(), value=None)

    def propose_rules(self) -> tuple:
        gw = self.gateway()
        if gw is None:
            return "❌ مفتاح Gemini غير متوفر في الاستوديو", "", ""
        current = read_text_safe(self.memory.rules, "")
        entries = [e for e in load_feedback(self.memory) if e.get("rating") in ("liked", "disliked")]
        if not entries:
            return "ℹ️ ما فيه تقييمات كافية لتحديث القواعد", "", ""
        prof = load_profile(self.memory)
        lines = []
        for e in sorted(entries, key=lambda x: str(x.get("updated_at") or ""), reverse=True)[:40]:
            mark = "👍" if e["rating"] == "liked" else "👎"
            score = f" ({e['score10']:g} من 10)" if isinstance(e.get("score10"), (int, float)) else ""
            tags = " | ".join(e.get("tags") or [])
            note = str(e.get("note") or "")[:300]
            promoted = " [رقاها بنفسه بعد ما رفضها الذكاء]" if e.get("source") == "promoted" else ""
            visual = f" [على الشاشة: {e['visual_event']}]" if e.get("visual_event") else ""
            lines.append(f"{mark}{score} «{e.get('title', '')}» [{e.get('category', '')}، {float(e.get('duration') or 0):.0f}ث]"
                         f"{promoted}{visual} {tags} {note}".strip())
        trims = [t for e in entries for t in e.get("manual_trims") or []]
        trim_txt = ""
        if trims:
            ends = [float(coerce_number(t.get("end_delta"), 0.0) or 0.0) for t in trims]
            starts = [float(coerce_number(t.get("start_delta"), 0.0) or 0.0) for t in trims]
            dropped = sum(max(0, int(t.get("pieces_before") or 0) - int(t.get("pieces_after") or 0)) for t in trims)
            trim_txt = (f"\nتعديلاته اليدوية: {len(trims)} تعديل، متوسط تغيير النهاية {sum(ends) / len(ends):+.1f} ثانية "
                        f"ومتوسط تغيير البداية {sum(starts) / len(starts):+.1f} ثانية، وحذف {dropped} قطعة.")
        pinned = self._pinned(current)
        pin_txt = ("\nهذي الأسطر مثبتة 📌 ولازم تبقى حرفياً بدون أي تغيير:\n" + "\n".join(pinned)) if pinned else ""
        reasons = ", ".join(f"{k} ({v})" for k, v in list((prof.get("tag_counts") or {}).items())[:6])
        prompt = (
            "أنت تساعد ممنتج مقاطع تيك توك يحدّث دليل ذوقه. هذي قواعده الحالية:\n\n"
            f"{current}\n\n"
            "وهذي آخر تقييماته للقطات اللي اختارها الذكاء الاصطناعي:\n" + "\n".join(lines) + trim_txt +
            f"\n\nأكثر أسباب الرفض: {reasons}\n"
            f"{pin_txt}\n\n"
            "المطلوب: اكتب نسخة محدثة من القواعد بالعامية السعودية، مختصرة وعملية، بنفس تنسيق الماركداون "
            "(عناوين ونقاط). ادمج الدروس الجديدة من التقييمات، احذف المكرر، وخل القواعد قابلة للتطبيق على "
            "اختيار اللقطات وقصها. لا تتجاوز 25 نقطة. رجّع نص القواعد فقط بدون أي مقدمة أو شرح."
        )
        try:
            res = gw.generate("rules", [prompt], schema=None, json_mode=False, label="تحديث القواعد")
        except QuotaPause as pause:
            return pause.message_ar, "", ""
        except (GatewayError, ParseFailure) as exc:
            return f"❌ تعذر توليد القواعد: {short_error(exc, 200)}", "", ""
        finally:
            gw.flush()
        proposal = str(res.data or "").strip()
        if proposal.startswith("```"):
            proposal = proposal.strip("`").split("\n", 1)[-1].strip()
        missing = [p for p in pinned if p not in [line.strip() for line in proposal.splitlines()]]
        if missing:
            self.pending_rules = None
            shown = "\n".join(f"• {m}" for m in missing)
            return f"❌ رفضنا الاقتراح لأنه حذف أو غيّر أسطر مثبتة 📌:\n{shown}", proposal, ""
        if len(proposal) < 50:
            self.pending_rules = None
            return "❌ الاقتراح قصير جداً، رفضناه", proposal, ""
        self.pending_rules = proposal
        return "✅ جاهز الاقتراح، راجع الفروقات ثم اعتمده أو تجاهله", proposal, self._diff_md(current, proposal)

    @staticmethod
    def _diff_md(old: str, new: str) -> str:
        old_lines = [line.rstrip() for line in str(old).splitlines() if line.strip()]
        new_lines = [line.rstrip() for line in str(new).splitlines() if line.strip()]
        added, removed = [], []
        for line in difflib.ndiff(old_lines, new_lines):
            tag, body = line[:2], line[2:]
            if tag == "+ ":
                added.append(body)
            elif tag == "- ":
                removed.append(body)
        if not added and not removed:
            return "ℹ️ ما فيه أي فرق عن القواعد الحالية"
        parts = []
        if added:
            parts.append("#### ➕ أسطر جديدة أو معدلة\n\n" + "\n\n".join(f"➕ {a}" for a in added))
        if removed:
            parts.append("#### ➖ أسطر بتنحذف أو تتغير\n\n" + "\n\n".join(f"➖ {r}" for r in removed))
        return "\n\n".join(parts)

    def approve_rules(self, proposal_text: str) -> tuple:
        text = str(proposal_text or "").strip() or (self.pending_rules or "")
        if not text:
            return "❌ ما فيه اقتراح للاعتماد", gr.update(), gr.update()
        current = read_text_safe(self.memory.rules, "")
        missing = [p for p in self._pinned(current) if p not in [line.strip() for line in text.splitlines()]]
        if missing:
            return "❌ ما اعتمدنا: النص يحذف أسطر مثبتة 📌", gr.update(), gr.update()
        backup = self._backup_rules()
        atomic_write_text(self.memory.rules, text + "\n")
        self.pending_rules = None
        note = f" (النسخة القديمة: {os.path.basename(backup)})" if backup else ""
        return f"✅ تم اعتماد القواعد الجديدة{note}", text, gr.update(choices=self._rules_history(), value=None)

    def discard_rules(self) -> tuple:
        self.pending_rules = None
        return "🗑️ تم تجاهل الاقتراح", "", ""

    def restore_rules(self, path: str) -> tuple:
        if not path or not os.path.isfile(path):
            return "❌ اختر نسخة من السجل", gr.update(), gr.update()
        text = read_text_safe(path, "")
        if not text.strip():
            return "❌ النسخة فاضية", gr.update(), gr.update()
        self._backup_rules()
        atomic_write_text(self.memory.rules, text.strip() + "\n")
        return f"↩️ تم استرجاع النسخة {os.path.basename(path)}", text, gr.update(choices=self._rules_history(), value=None)

    # ------------------------------------------------------------ golden clips

    def golden_choices(self) -> tuple:
        golden = load_golden(self.memory)
        choices, included = [], []
        for key, it in sorted((golden.get("items") or {}).items(), key=lambda kv: str(kv[1].get("file") or "")):
            if it.get("status") != "done":
                continue
            a = it.get("analysis") or {}
            flags = " 🗑️ الملف محذوف" if it.get("file_missing") else ""
            label = f"{it.get('file') or key[:10]} | {a.get('moment_type', '')} | {float(it.get('duration_s') or 0):.0f}ث{flags}"
            choices.append((label, key))
            if not it.get("excluded"):
                included.append(key)
        return choices, included

    def golden_view(self) -> tuple:
        golden = load_golden(self.memory)
        choices, included = self.golden_choices()
        folder = self.memory.golden_dir
        count = 0
        if os.path.isdir(folder):
            for _root, _dirs, files in os.walk(folder):
                count += sum(1 for f in files if os.path.splitext(f)[1].lower() in (".mp4", ".mov", ".m4v", ".webm", ".mkv"))
        head = f"📂 المجلد: `{folder}` (فيه {count} فيديو)\n\n"
        return head + golden_summary_ar(golden).replace("\n", "\n\n"), gr.update(choices=choices, value=included)

    def golden_save_selection(self, selected: list) -> tuple:
        golden = load_golden(self.memory)
        selected = set(selected or [])
        changed = 0
        for key, it in (golden.get("items") or {}).items():
            if it.get("status") != "done":
                continue
            want_excluded = key not in selected
            if bool(it.get("excluded")) != want_excluded:
                set_excluded(self.memory, key, want_excluded)
                changed += 1
        msg = f"✅ حدثنا {changed} مقطع. الحمض الذهبي يتحدث بالتدريب الجاي" if changed else "ℹ️ ما فيه تغيير"
        summary, group = self.golden_view()
        return (msg, summary, group, self.taste_md())

    def golden_train(self, max_new: Any, redistill: bool):
        gw = self.gateway()
        if gw is None:
            summary, group = self.golden_view()
            yield "❌ مفتاح Gemini غير متوفر في الاستوديو", summary, group, self.taste_md()
            return
        limit = coerce_number(max_new, None)
        limit = int(limit) if limit is not None and limit > 0 else None
        summary, group = self.golden_view()
        yield ("⏳ جاري تحليل اللقطات الذهبية الجديدة (كل مقطع طلب واحد من Gemini)... التفاصيل في سجل Colab",
               summary, group, gr.update())
        try:
            result = train_golden(self.drive_root, self.api_key, max_new=limit, redistill=bool(redistill), gateway=gw)
        except Exception as exc:
            summary, group = self.golden_view()
            yield f"❌ تعذر التدريب: {short_error(exc, 200)}", summary, group, self.taste_md()
            return
        finally:
            gw.flush()
        parts = [f"حللنا {result.get('analyzed', 0)}", f"فشل {result.get('failed', 0)}", f"تخطينا {result.get('skipped', 0)}"]
        msg = "🌟 " + "، ".join(parts)
        if result.get("distilled"):
            msg += " | 🧬 الحمض الذهبي تحدث"
        if result.get("paused"):
            msg += f"\n\n{result['paused']}"
        summary, group = self.golden_view()
        yield msg, summary, group, self.taste_md()

    # ------------------------------------------------------------ Gemini health

    def health(self) -> tuple:
        gw = self.gateway()
        if gw is None:
            return "ℹ️ مفتاح Gemini غير متوفر في الاستوديو", ""
        return gw.usage_report_ar(), gw.health_report_ar()

    def reset_health(self) -> tuple:
        gw = self.gateway()
        if gw is None:
            return "ℹ️ مفتاح Gemini غير متوفر", "", ""
        gw.reset_model_health()
        gw.flush()
        usage, health = self.health()
        return "✅ صفرنا حالة الموديلات (الموقوفة والقواطع)", usage, health

    # ------------------------------------------------------------ export and cleanup

    def export_info(self, vod_id: str) -> str:
        vd = self.vod(vod_id)
        if vd is None:
            return "ℹ️ اختر بث أول"
        manifest = load_manifest(vd.vp)
        clips = manifest.get("clips") or []
        liked = [c for c in clips if c.get("rating") == "liked"]
        ready_liked = [c for c in liked if not clip_needs_render(c) and c.get("status") != "media_deleted"]
        drafts = [c for c in liked if clip_needs_render(c) and c.get("status") != "media_deleted"]
        lines = [f"📊 عندك {len(clips)} لقطة، منها {len(liked)} عجبتك 👍 ({len(ready_liked)} جاهزة 1080p)."]
        if drafts:
            names = "، ".join(f"#{int(c.get('id') or 0):02d}" for c in drafts[:12])
            lines.append(f"⚠️ **تنبيه:** {len(drafts)} لقطة عجبتك لسه مسودة وما انرندرت 1080p ({names}). "
                         "اضغط «🎬 رندر كل اللقطات المقبولة (👍)» في تبويب المراجعة قبل التصدير، وإلا ما بتدخل الملف.")
        return "\n\n".join(lines)

    def export_zip(self, vod_id: str, only_liked: bool) -> tuple:
        vd = self.vod(vod_id)
        if vd is None:
            return "❌ اختر بث أول", None, self.export_info(vod_id)
        manifest = load_manifest(vd.vp)
        all_clips = manifest.get("clips") or []
        pool = [c for c in all_clips if c.get("rating") == "liked"] if only_liked else list(all_clips)
        drafts = [c for c in pool if clip_needs_render(c) and c.get("status") != "media_deleted"]
        clips = [c for c in pool if not clip_needs_render(c) and c.get("video_path") and os.path.isfile(c["video_path"])]
        warn = ""
        if drafts:
            names = "، ".join(f"#{int(c.get('id') or 0):02d} «{c.get('title')}»" for c in drafts[:8])
            warn = (f"\n\n⚠️ **انتبه:** {len(drafts)} لقطة {'عجبتك' if only_liked else ''} لسه مسودة وما دخلت الملف: {names}. "
                    "اضغط «🎬 رندر كل اللقطات المقبولة (👍)» أول، وبعدها صدّر من جديد.")
        if not clips:
            return "ℹ️ ما فيه لقطات جاهزة تطابق الاختيار" + warn, None, self.export_info(vod_id)
        out_dir = ensure_dir(os.path.join(TMP_ROOT, "exports"))
        path = os.path.join(out_dir, f"{vod_id}_{'liked' if only_liked else 'all'}_{_stamp()}.zip")
        with zipfile.ZipFile(path, "w") as zf:
            for c in sorted(clips, key=lambda x: int(x.get("id") or 0)):
                zf.write(c["video_path"], os.path.basename(c["video_path"]), compress_type=zipfile.ZIP_STORED)
                if c.get("srt_path") and os.path.isfile(c["srt_path"]):
                    zf.write(c["srt_path"], os.path.basename(c["srt_path"]), compress_type=zipfile.ZIP_DEFLATED)
                notes = c.get("notes_path")
                if notes and os.path.isfile(notes):
                    zf.write(notes, os.path.basename(notes), compress_type=zipfile.ZIP_DEFLATED)
        msg = f"✅ جاهز: {len(clips)} لقطة ({human_size(os.path.getsize(path))}) مع السبتايتل وملاحظات CapCut" + warn
        return msg, path, self.export_info(vod_id)

    def _cleanup_targets(self, vd: _VodData) -> list:
        vp = vd.vp
        targets = []
        for path in (vp.audio, vp.clips_dir, vp.clips_archive_dir, vp.previews_dir):
            if os.path.exists(path):
                targets.append(path)
        targets.extend(sorted(glob.glob(os.path.join(vp.vod_dir, "*.zip"))))
        memory_root = os.path.abspath(self.memory.dir)
        return [t for t in targets if not (os.path.abspath(t) + os.sep).startswith(memory_root + os.sep)]

    def cleanup_info(self, vod_id: str) -> str:
        vd = self.vod(vod_id)
        if vd is None:
            return "ℹ️ اختر بث أول"
        targets = self._cleanup_targets(vd)
        tmp_zips = glob.glob(os.path.join(TMP_ROOT, "exports", f"{vod_id}_*.zip"))
        if not targets and not tmp_zips:
            return "✅ ما فيه ملفات ثقيلة لهذا البث على Drive"
        total = sum(dir_size_bytes(t) for t in targets)
        lines = [f"🧹 اللي بينحذف للبث {vod_id} (المجموع {human_size(total)}):"]
        for t in targets:
            lines.append(f"* {os.path.relpath(t, vd.vp.vod_dir)}: {human_size(dir_size_bytes(t))}")
        if tmp_zips:
            lines.append(f"* ملفات ZIP مؤقتة في Colab: {len(tmp_zips)}")
        drafts = [c for c in load_manifest(vd.vp).get("clips") or [] if c.get("rating") == "liked" and clip_needs_render(c)
                  and c.get("status") != "media_deleted"]
        lines.append("")
        if drafts:
            lines.append(f"⚠️ عندك {len(drafts)} لقطة عجبتك لسه مسودة. بعد التنظيف تنحذف معايناتها وسبتايتلها، رندرها وصدّرها أول.")
        lines.append("✅ يبقى محفوظ: التفريغ النصي، الشات، الخصائص الصوتية، ملف اللقطات، وكل مجلد memory "
                     "(القواعد والتقييمات والذوق واللقطات الذهبية).")
        lines.append("⚠️ بعد الحذف يبقى الرندر من الاستوديو شغال (يقص من رابط البث)، لكن لقطات إضافية بالناقد "
                     "تحتاج تشغيل المرحلة 2 مرة ثانية عشان يرجع ملف الصوت.")
        return "\n".join(lines)

    def cleanup(self, vod_id: str, confirmed: bool) -> tuple:
        vd = self.vod(vod_id)
        if vd is None:
            return "❌ اختر بث أول", False, self.cleanup_info(vod_id)
        if not confirmed:
            return "⚠️ لازم تأشر على مربع التأكيد قبل الحذف", False, self.cleanup_info(vod_id)
        freed = 0
        errors = []
        for t in self._cleanup_targets(vd):
            try:
                freed += safe_delete(t, self.drive_root)
            except (PermissionError, OSError) as exc:
                errors.append(f"{os.path.basename(t)}: {short_error(exc, 80)}")
        for z in glob.glob(os.path.join(TMP_ROOT, "exports", f"{vod_id}_*.zip")):
            try:
                os.remove(z)
            except OSError:
                pass
        for d in ("stage2", "stage3"):
            p = os.path.join(TMP_ROOT, d, f"{vod_id}.mp3")
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        with vd.lock:
            manifest = load_manifest(vd.vp)
            for c in manifest.get("clips") or []:
                video_gone = not (c.get("video_path") and os.path.isfile(c["video_path"]))
                if c.get("preview_path") and not os.path.isfile(c["preview_path"]):
                    c["preview_path"] = ""
                    c["preview_window"] = []
                if video_gone:
                    c["status"] = "media_deleted"
            save_manifest(vd.vp, manifest)
        self._vods.pop(vod_id, None)
        msg = f"✅ تم التنظيف ووفرنا {human_size(freed)} من Drive"
        if errors:
            msg += "\n⚠️ تعذر حذف: " + "، ".join(errors)
        return msg, False, self.cleanup_info(vod_id)


# ============================================================================
# UI
# ============================================================================

def _initial_vod(studio: Studio, vod_dir: Optional[str]) -> Optional[str]:
    choices = studio.vod_choices()
    ids = [v for _, v in choices]
    if vod_dir:
        vid = os.path.basename(os.path.normpath(vod_dir))
        if vid in ids:
            return vid
    return ids[0] if ids else None


def build_ui(studio: Studio, vod_dir: Optional[str] = None) -> Any:
    first_vod = _initial_vod(studio, vod_dir)
    first_clips = studio.clip_choices(first_vod) if first_vod else []
    first_uid = first_clips[0][1] if first_clips else None
    golden_choices, golden_included = studio.golden_choices()

    demo = _blocks("استوديو اللقطات")
    with demo:
        gr.Markdown("# 🎬 استوديو مراجعة اللقطات والتعلم الذاتي")
        with gr.Row():
            vod_dd = _mk(gr.Dropdown, choices=studio.vod_choices(), value=first_vod, label="🎥 البث", scale=4, interactive=True)
            refresh_btn = _mk(gr.Button, value="🔄 تحديث القوائم", scale=1)
        status = _mk(gr.Markdown, value="" if studio.timeline_ok else
                     "ℹ️ نسخة Gradio الحالية ما تدعم المحرر الزمني، بنستخدم التحكم العادي.")

        with gr.Tabs():
            # ---------------------------------------------------------- Tab 1
            with gr.Tab("✂️ المراجعة والمحرر"):
                with gr.Row():
                    prev_btn = _mk(gr.Button, value="◀ السابقة", scale=1)
                    clip_dd = _mk(gr.Dropdown, choices=first_clips, value=first_uid, label="🎞️ اللقطة", scale=5, interactive=True)
                    next_btn = _mk(gr.Button, value="التالية ▶", scale=1)
                with gr.Row():
                    render_one_btn = _mk(gr.Button, value="🎬 رندر هذي اللقطة (1080p)", variant="primary", scale=2)
                    render_all_btn = _mk(gr.Button, value="🎬 رندر كل اللقطات المقبولة (👍)", variant="primary", scale=2)
                    height_dd = _mk(gr.Dropdown, choices=RENDER_HEIGHTS, value=1080, label="جودة الرندر", scale=1, interactive=True)
                render_log = _mk(gr.Markdown, value="")
                with _mk(gr.Column, visible=studio.timeline_ok) as tl_col:
                    if studio.timeline_ok:
                        timeline = build_timeline(TIMELINE_ELEM)
                        bridge = build_bridge(TIMELINE_ELEM)
                    else:
                        timeline = gr.State(None)
                        bridge = gr.State("")
                with gr.Row():
                    with _mk(gr.Column, scale=3):
                        video = _mk(gr.Video, label="🎞️ الفيديو (النسخة المرندرة أو المعاينة)", interactive=False, height=380)
                        info_md = _mk(gr.Markdown, value="")
                        preview_btn = _mk(gr.Button, value="🎞️ جهّز معاينة للمحرر")
                    with _mk(gr.Column, scale=2):
                        with _mk(gr.Column, visible=not studio.timeline_ok) as fb_col:
                            pieces_cb = _mk(gr.CheckboxGroup, choices=[], value=[],
                                            label="✂️ القطع (شيل العلامة عن القطعة اللي تبي تحذفها)")
                            with gr.Row():
                                start_minus = _mk(gr.Button, value="البداية ◀ 0.5ث")
                                start_plus = _mk(gr.Button, value="البداية ▶ 0.5ث")
                                end_minus = _mk(gr.Button, value="النهاية ◀ 0.5ث")
                                end_plus = _mk(gr.Button, value="النهاية ▶ 0.5ث")
                            with gr.Row():
                                start_delta = _mk(gr.Number, value=0.0, label="تغيير البداية (ث، السالب أبكر)", step=NUDGE)
                                end_delta = _mk(gr.Number, value=0.0, label="تغيير النهاية (ث، الموجب أطول)", step=NUDGE)
                            snap_cb = _mk(gr.Checkbox, value=True, label="التزم بحدود الكلمات")
                            apply_btn = _mk(gr.Button, value="✂️ طبّق التعديل (مسودة، بدون رندر)", variant="primary")
                        srt_box = _mk(gr.Textbox, label="📝 السبتايتل (SRT)", lines=10, max_lines=30, rtl=True, interactive=True)
                        with gr.Row():
                            srt_save_btn = _mk(gr.Button, value="✅ تحقق واحفظ")
                            srt_regen_btn = _mk(gr.Button, value="♻️ ولّد من جديد")
                        rating = _mk(gr.Radio, choices=RATING_CHOICES, value="pending", label="التقييم")
                        score = _mk(gr.Number, value=None, label="الدرجة من 10 (اختياري)", minimum=0, maximum=10, step=0.5)
                        tags = _mk(gr.CheckboxGroup, choices=POSITIVE_TAGS + NEGATIVE_TAGS, value=[], label="وسوم سريعة")
                        note = _mk(gr.Textbox, label="ملاحظتك", lines=3, rtl=True)
                        save_btn = _mk(gr.Button, value="💾 احفظ التقييم ودرّب الذوق", variant="primary")

            # ---------------------------------------------------------- Tab 2
            with gr.Tab("🗂️ المرشحون"):
                gr.Markdown("لقطات اقترحها الكشاف أو الرادار وما صارت لقطات. إذا شفت وحدة تستاهل، رقّها بضغطة.")
                with gr.Row():
                    cand_filter = _mk(gr.Radio, choices=CANDIDATE_FILTERS, value="all", label="عرض", scale=3)
                    cand_refresh_btn = _mk(gr.Button, value="🔄 تحديث المرشحين", scale=1)
                cand_note = _mk(gr.Markdown, value="")
                cand_dd = _mk(gr.Dropdown, choices=[], value=None, label="🧩 المرشح", interactive=True)
                with gr.Row():
                    with _mk(gr.Column, scale=3):
                        cand_video = _mk(gr.Video, label="معاينة المرشح", interactive=False, height=340)
                    with _mk(gr.Column, scale=2):
                        cand_md = _mk(gr.Markdown, value="")
                        promote_btn = _mk(gr.Button, value="⬆️ ترقية إلى لقطة", variant="primary")

            # ---------------------------------------------------------- Tab 3
            with gr.Tab("🧠 الذوق والقواعد"):
                taste_md = _mk(gr.Markdown, value=studio.taste_md())
                train_btn = _mk(gr.Button, value="🔄 درّب الذوق الرقمي الحين")
                rules_box = _mk(gr.Textbox, value=read_text_safe(studio.memory.rules, ""),
                                label="📜 قواعد الممنتج (الأسطر اللي تبدأ بـ 📌 مثبتة)", lines=14, max_lines=40, rtl=True, interactive=True)
                rules_save_btn = _mk(gr.Button, value="💾 احفظ القواعد يدوياً")
                propose_btn = _mk(gr.Button, value="🧠 اقترح تحديث القواعد من التقييمات", variant="primary")
                proposal_box = _mk(gr.Textbox, label="الاقتراح", lines=12, max_lines=40, rtl=True, interactive=True)
                diff_md = _mk(gr.Markdown, value="")
                with gr.Row():
                    approve_btn = _mk(gr.Button, value="✅ اعتمد التحديث", variant="primary")
                    discard_btn = _mk(gr.Button, value="❌ تجاهل")
                with gr.Row():
                    history_dd = _mk(gr.Dropdown, choices=studio._rules_history(), value=None, label="🗂️ نسخ سابقة", scale=4)
                    restore_btn = _mk(gr.Button, value="↩️ استرجع النسخة", scale=1)

            # ---------------------------------------------------------- Tab 4
            with gr.Tab("🌟 اللقطات الذهبية"):
                golden_md = _mk(gr.Markdown, value="")
                golden_cb = _mk(gr.CheckboxGroup, choices=golden_choices, value=golden_included,
                                label="المقاطع المحللة (شيل العلامة عن المقطع اللي ما تبيه يأثر على الذوق)")
                golden_save_btn = _mk(gr.Button, value="💾 احفظ الاختيار")
                with gr.Row():
                    golden_max = _mk(gr.Number, value=None, label="أقصى عدد مقاطع جديدة هالمرة (فاضي يعني الكل)", minimum=0, step=1)
                    golden_redistill = _mk(gr.Checkbox, value=False, label="أعد استخلاص الحمض الذهبي حتى لو ما فيه جديد")
                golden_train_btn = _mk(gr.Button, value="🌟 درّب على اللقطات الذهبية", variant="primary")

            # ---------------------------------------------------------- Tab 5
            with gr.Tab("🩺 صحة Gemini"):
                usage_md = _mk(gr.Markdown, value="")
                health_md = _mk(gr.Markdown, value="")
                with gr.Row():
                    health_btn = _mk(gr.Button, value="🔄 تحديث")
                    reset_btn = _mk(gr.Button, value="♻️ صفّر حالة الموديلات", variant="stop")

            # ---------------------------------------------------------- Tab 6
            with gr.Tab("📦 التصدير والتنظيف"):
                export_md = _mk(gr.Markdown, value=studio.export_info(first_vod) if first_vod else "")
                only_liked = _mk(gr.Checkbox, value=True, label="اللقطات اللي عجبتني فقط 👍")
                zip_btn = _mk(gr.Button, value="📦 جهّز ملف ZIP", variant="primary")
                zip_file = _mk(gr.File, label="ملف التحميل", interactive=False)
                gr.Markdown("### 🧹 تنظيف مساحة Drive")
                cleanup_md = _mk(gr.Markdown, value=studio.cleanup_info(first_vod) if first_vod else "")
                with gr.Row():
                    size_btn = _mk(gr.Button, value="🔎 احسب المساحة")
                    confirm_cb = _mk(gr.Checkbox, value=False, label="أؤكد إني صدّرت اللي أبيه وأبي أحذف ملفات هذا البث الثقيلة")
                cleanup_btn = _mk(gr.Button, value="🧹 تنظيف مساحة هذا البث من Drive", variant="stop")

        view_outputs = [video, info_md, pieces_cb, srt_box, rating, score, tags, note, start_delta, end_delta,
                        timeline, tl_col, fb_col]

        # ---------------- wiring: review

        def on_vod(vod_id):
            choices = studio.clip_choices(vod_id)
            uid = choices[0][1] if choices else None
            return ((gr.update(choices=choices, value=uid),) + studio.view(vod_id, uid)
                    + (studio.cleanup_info(vod_id), studio.export_info(vod_id)))

        vod_dd.change(on_vod, inputs=[vod_dd], outputs=[clip_dd] + view_outputs + [cleanup_md, export_md])

        def on_refresh(vod_id):
            vods = studio.vod_choices()
            ids = [v for _, v in vods]
            vid = vod_id if vod_id in ids else (ids[0] if ids else None)
            studio._vods.pop(vid, None)
            clips = studio.clip_choices(vid) if vid else []
            return (gr.update(choices=vods, value=vid), gr.update(choices=clips, value=clips[0][1] if clips else None),
                    "🔄 تم تحديث القوائم")

        refresh_btn.click(on_refresh, inputs=[vod_dd], outputs=[vod_dd, clip_dd, status])
        clip_dd.change(lambda v, u: studio.view(v, u), inputs=[vod_dd, clip_dd], outputs=view_outputs)

        def step(vod_id, uid, delta):
            uids = [u for _, u in studio.clip_choices(vod_id)]
            if not uids:
                return gr.update()
            i = uids.index(uid) if uid in uids else 0
            return gr.update(value=uids[(i + delta) % len(uids)])

        prev_btn.click(lambda v, u: step(v, u, -1), inputs=[vod_dd, clip_dd], outputs=[clip_dd])
        next_btn.click(lambda v, u: step(v, u, 1), inputs=[vod_dd, clip_dd], outputs=[clip_dd])

        def nudge(value, delta):
            return round(float(coerce_number(value, 0.0) or 0.0) + delta, 2)

        start_minus.click(lambda x: nudge(x, -NUDGE), inputs=[start_delta], outputs=[start_delta])
        start_plus.click(lambda x: nudge(x, NUDGE), inputs=[start_delta], outputs=[start_delta])
        end_minus.click(lambda x: nudge(x, -NUDGE), inputs=[end_delta], outputs=[end_delta])
        end_plus.click(lambda x: nudge(x, NUDGE), inputs=[end_delta], outputs=[end_delta])

        apply_btn.click(studio.apply_fallback, inputs=[vod_dd, clip_dd, pieces_cb, start_delta, end_delta, snap_cb],
                        outputs=[status] + view_outputs)
        if studio.timeline_ok:
            commit_inputs = [vod_dd, clip_dd, timeline, bridge]
            commit_outputs = [status] + view_outputs
            try:
                commit_listener = getattr(timeline, COMMIT_EVENT, None)
            except Exception:
                commit_listener = None
            wired = False
            if callable(commit_listener):
                try:
                    commit_listener(studio.commit_timeline, inputs=commit_inputs, outputs=commit_outputs)
                    wired = True
                except Exception as exc:
                    log(f"⚠️ تعذر ربط حدث الاعتماد في المحرر ({short_error(exc, 100)})، نستخدم القناة الاحتياطية")
            if not wired:
                # The editor writes the JSON into the hidden textbox from script. Gradio fires
                # .change for that (value changed) but not .input (kept for real typing only),
                # so .change is the primary listener and .input the last resort.
                bridge_listener = getattr(bridge, "change", None) or getattr(bridge, "input", None)
                if callable(bridge_listener):
                    bridge_listener(studio.commit_timeline, inputs=commit_inputs, outputs=commit_outputs)
                    log("ℹ️ اعتماد المحرر يوصل عن طريق القناة الاحتياطية")
                else:
                    log("⚠️ ما قدرنا نربط زر الاعتماد في المحرر، استخدم التحكم العادي")
        preview_btn.click(studio.make_preview, inputs=[vod_dd, clip_dd], outputs=[status] + view_outputs)
        _on(render_one_btn.click, studio.render_clip, inputs=[vod_dd, clip_dd, height_dd],
            outputs=[render_log, video, info_md, clip_dd], concurrency_id="render", concurrency_limit=1)
        _on(render_all_btn.click, studio.render_all_liked, inputs=[vod_dd, clip_dd, height_dd],
            outputs=[render_log, clip_dd, export_md], concurrency_id="render", concurrency_limit=1)
        srt_save_btn.click(studio.save_srt, inputs=[vod_dd, clip_dd, srt_box], outputs=[status])
        srt_regen_btn.click(studio.regen_srt, inputs=[vod_dd, clip_dd], outputs=[status, srt_box])
        save_btn.click(studio.save_rating, inputs=[vod_dd, clip_dd, rating, score, tags, note],
                       outputs=[status, clip_dd, export_md])

        # ---------------- wiring: candidates

        cand_outputs = [cand_dd, cand_note, cand_video, cand_md]
        cand_refresh_btn.click(studio.candidates_refresh, inputs=[vod_dd, cand_filter], outputs=cand_outputs)
        cand_filter.change(studio.candidates_refresh, inputs=[vod_dd, cand_filter], outputs=cand_outputs)
        vod_dd.change(studio.candidates_refresh, inputs=[vod_dd, cand_filter], outputs=cand_outputs)
        cand_dd.change(studio.candidate_view, inputs=[vod_dd, cand_dd], outputs=[cand_video, cand_md])
        promote_btn.click(studio.promote, inputs=[vod_dd, cand_dd, cand_filter],
                          outputs=[status, cand_dd, clip_dd, cand_note, cand_video, cand_md])

        # ---------------- wiring: taste and rules

        train_btn.click(studio.retrain_now, inputs=[], outputs=[status, taste_md])
        rules_save_btn.click(studio.save_rules_manual, inputs=[rules_box], outputs=[status, history_dd])
        _on(propose_btn.click, studio.propose_rules, inputs=[], outputs=[status, proposal_box, diff_md],
            concurrency_id="gemini", concurrency_limit=1)
        approve_btn.click(studio.approve_rules, inputs=[proposal_box], outputs=[status, rules_box, history_dd])
        discard_btn.click(studio.discard_rules, inputs=[], outputs=[status, proposal_box, diff_md])
        restore_btn.click(studio.restore_rules, inputs=[history_dd], outputs=[status, rules_box, history_dd])

        # ---------------- wiring: golden clips

        golden_save_btn.click(studio.golden_save_selection, inputs=[golden_cb], outputs=[status, golden_md, golden_cb, taste_md])
        _on(golden_train_btn.click, studio.golden_train, inputs=[golden_max, golden_redistill],
            outputs=[status, golden_md, golden_cb, taste_md], concurrency_id="gemini", concurrency_limit=1)

        # ---------------- wiring: Gemini health

        health_btn.click(studio.health, inputs=[], outputs=[usage_md, health_md])
        reset_btn.click(studio.reset_health, inputs=[], outputs=[status, usage_md, health_md])

        # ---------------- wiring: export and cleanup

        zip_btn.click(studio.export_zip, inputs=[vod_dd, only_liked], outputs=[status, zip_file, export_md])
        size_btn.click(studio.cleanup_info, inputs=[vod_dd], outputs=[cleanup_md])
        cleanup_btn.click(studio.cleanup, inputs=[vod_dd, confirm_cb], outputs=[status, confirm_cb, cleanup_md])

        # ---------------- initial load

        def on_load(vod_id, uid, flt):
            golden_text, golden_group = studio.golden_view()
            usage, health = studio.health()
            return (studio.view(vod_id, uid) + studio.candidates_refresh(vod_id, flt)
                    + (golden_text, golden_group, usage, health))

        demo.load(on_load, inputs=[vod_dd, clip_dd, cand_filter],
                  outputs=view_outputs + cand_outputs + [golden_md, golden_cb, usage_md, health_md])
    return demo


def launch_studio(vod_dir: Optional[str] = None, drive_root: str = DRIVE_ROOT_DEFAULT,
                  api_key: Optional[str] = None, share: bool = True, server_port: Optional[int] = None) -> Any:
    """Build and launch the Studio. Returns the Blocks object."""
    t0 = time.time()
    log(f"🎛️ جاري تشغيل الاستوديو (Gradio {getattr(gr, '__version__', '?')})...")
    ensure_dir(os.path.join(TMP_ROOT, "exports"))
    studio = Studio(drive_root, api_key)
    if studio.timeline_ok:
        log("✂️ المحرر الزمني شغال (Gradio 6)")
    else:
        log("ℹ️ المحرر الزمني غير متاح بهذي النسخة من Gradio، بنستخدم التحكم العادي")
    demo = build_ui(studio, vod_dir)
    try:
        demo.queue(default_concurrency_limit=2)
    except TypeError:
        try:
            demo.queue()
        except TypeError:
            pass
    _launch(demo, share=share, server_port=server_port, allowed_paths=[drive_root, TMP_ROOT],
            show_error=True, inbrowser=False, quiet=False)
    log(f"✅ الاستوديو جاهز خلال {format_duration_ar(time.time() - t0)}")
    return demo


# === END OF stage4_studio.py ===
