"""
stage4_studio.py
Stage 4 of TwitchClipper: the Gradio Review and Self Learning Studio (Colab CPU).

Features
  * VOD picker (every VOD folder on Drive) and ◀ ▶ clip navigation
  * Video preview, scores, reasons, jump cut piece checklist (untick a piece to drop it)
  * Start and end nudges in 0.5 s steps, optional snap to word edges, one click recut with a
    new versioned file name (the browser never shows a cached video) and SRT resync
  * SRT editor with validation before saving, and SRT regeneration from the transcript
  * Rating (👍 / 👎), a 0 to 10 score, quick tags and a note; the numeric taste profile is
    retrained instantly on every save, and manual trims become padding preferences
  * Rules: manual edit, Gemini proposal with a diff, approval with a timestamped backup,
    restore from history; lines starting with 📌 can never be removed by a proposal
  * Export: ZIP in /tmp (MP4 stored without recompression, SRT deflated)
  * Drive cleanup for one VOD (audio.mp3, clips/, clips_archive/, old ZIPs) behind a
    confirmation checkbox, through common.safe_delete, never touching TwitchClipper/memory
  * Works on Gradio 5 and 6: only stable components, keyword arguments filtered against the
    installed signatures, no theme or css arguments

Public contract
  launch_studio(vod_dir=None, drive_root=DRIVE_ROOT_DEFAULT, api_key=None, share=True,
                server_port=None) -> gradio Blocks
"""

from __future__ import annotations

import difflib
import glob
import inspect
import os
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
    atomic_write_text,
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
from media_engine import (
    build_phrases,
    collect_words,
    cut_and_stitch,
    drop_pieces,
    generate_srt,
    load_features,
    pieces_net,
    resolve_stream,
    snap_to_frames,
    trim_pieces,
    validate_srt,
)
from taste_engine import (
    backfill_legacy,
    build_signals,
    compute_clip_features,
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
]
RATING_CHOICES = [("⏳ بدون تقييم", "pending"), ("👍 عجبني", "liked"), ("👎 ما عجبني", "disliked")]
RATING_BADGE = {"liked": "👍", "disliked": "👎", "pending": "⏳"}
PIN = "📌"
NUDGE = 0.5


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


# ============================================================================
# Per VOD cache
# ============================================================================

class _VodData:
    """Lazy per VOD data for recuts, SRT regeneration and features."""

    def __init__(self, vp: VodPaths):
        self.vp = vp
        self._words: Optional[list] = None
        self._feats = None
        self._feats_loaded = False
        self._signals = None
        self.stream = None
        self.meta = load_json_safe(vp.meta, default={}, expected_types=(dict,), quiet=True) or {}

    @property
    def vod_url(self) -> str:
        return self.meta.get("vod_url") or f"https://www.twitch.tv/videos/{self.vp.vod_id}"

    @property
    def words(self) -> list:
        if self._words is None:
            tr = load_json_safe(self.vp.transcript, default={}, expected_types=(dict,), quiet=True) or {}
            self._words = collect_words(tr)
            self.duration = float(coerce_number(tr.get("duration"), 0.0) or 0.0)
        return self._words

    @property
    def feats(self):
        if not self._feats_loaded:
            self._feats = load_features(self.vp, words=self.words, allow_build=False)
            self._feats_loaded = True
        return self._feats

    @property
    def vod_duration(self) -> float:
        _ = self.words
        dur = getattr(self, "duration", 0.0) or 0.0
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

    def get_stream(self):
        if self.stream is None:
            self.stream = resolve_stream(self.vod_url)
        return self.stream

    def av_offset(self) -> float:
        data = load_json_safe(os.path.join(self.vp.stage3_dir, "av_offset.json"), default={}, expected_types=(dict,), quiet=True) or {}
        return float(coerce_number(data.get("offset"), 0.0) or 0.0)


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

    # ------------------------------------------------------------ helpers

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
            label = f"🎥 {v['vod_id']} | {v['clips']} لقطة{dur}"
            out.append((label, v["vod_id"]))
        return out

    @staticmethod
    def clip_label(c: dict) -> str:
        badge = RATING_BADGE.get(c.get("rating", "pending"), "⏳")
        warn = " ⚠️" if c.get("reviewed") is False else ""
        gone = " 🗑️" if c.get("status") == "media_deleted" else ""
        return f"{badge} #{int(c.get('id') or 0):02d} {c.get('title', 'لقطة')} ({float(c.get('duration') or 0):.0f}ث){warn}{gone}"

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

    def feedback_for(self, entries: list, vod_id: str, clip: dict) -> Optional[dict]:
        for e in entries:
            if e.get("uid") == clip.get("uid"):
                return e
        key = clip.get("legacy_key")
        if key:
            for e in entries:
                if e.get("uid") == f"legacy:{key}":
                    return e
        return None

    # ------------------------------------------------------------ views

    def view(self, vod_id: str, uid: str) -> tuple:
        vd, _manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            empty_pieces = gr.update(choices=[], value=[])
            return (None, "ℹ️ اختر بث ولقطة من القائمة.", empty_pieces, "", "pending", None, [], "", 0.0, 0.0)
        video = clip.get("video_path") if clip.get("video_path") and os.path.isfile(clip["video_path"]) else None
        srt_text = read_text_safe(clip.get("srt_path") or "", "") if clip.get("srt_path") else ""
        segs = clip.get("segments") or []
        choices = []
        for i, (s, e) in enumerate(segs):
            choices.append((f"قطعة {i + 1}: {sec_to_hms(s)} إلى {sec_to_hms(e)} ({e - s:.1f}ث)", str(i)))
        pieces = gr.update(choices=choices, value=[str(i) for i in range(len(segs))])
        tags = [t for t in clip.get("tags") or [] if t in POSITIVE_TAGS + NEGATIVE_TAGS]
        return (video, self.info_md(clip), pieces, srt_text, clip.get("rating", "pending"),
                clip.get("score10"), tags, clip.get("note", ""), 0.0, 0.0)

    @staticmethod
    def info_md(clip: dict) -> str:
        scores = clip.get("scores") or {}
        critic = scores.get("critic") or {}
        lines = [f"### {Studio.clip_label(clip)}",
                 f"🏷️ **{clip.get('category', '')}** | ⏱️ {float(clip.get('duration') or 0):.1f} ثانية "
                 f"| يبدأ من {sec_to_hms(clip.get('start') or 0)} | {len(clip.get('segments') or [])} قطعة "
                 f"| محذوف {float(clip.get('removed_sec') or 0):.1f} ثانية"]
        if clip.get("reviewed") is False:
            lines.append("⚠️ هذي اللقطة ما راجعها الناقد")
        if critic:
            names = {"hook": "البداية", "payoff": "القفلة", "reaction": "ردة الفعل", "context": "السياق", "pacing": "الإيقاع"}
            lines.append("🎯 الناقد: " + " | ".join(f"{names.get(k, k)} {v}" for k, v in critic.items()))
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
        return "\n\n".join(lines)

    # ------------------------------------------------------------ actions

    def recut(self, vod_id: str, uid: str, keep: list, start_delta: Any, end_delta: Any, snap: bool):
        vd, manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            yield ("❌ اختر لقطة أول",) + self.view(vod_id, uid)
            return
        segs = clip.get("segments") or []
        keep_idx = sorted(int(k) for k in keep or [] if str(k).isdigit() and int(k) < len(segs))
        if not keep_idx:
            yield ("❌ لازم تبقي قطعة وحدة على الأقل",) + self.view(vod_id, uid)
            return
        sd = float(coerce_number(start_delta, 0.0) or 0.0)
        ed = float(coerce_number(end_delta, 0.0) or 0.0)
        pieces = drop_pieces(segs, keep_idx)
        pieces = trim_pieces(pieces, sd, ed, words=vd.words, snap_to_words=bool(snap), vod_duration=vd.vod_duration)
        if not pieces:
            yield ("❌ التعديل حذف اللقطة كاملة، خفف القص",) + self.view(vod_id, uid)
            return
        yield (f"⏳ جاري إعادة القص ({len(pieces)} قطعة، {pieces_net(pieces):.1f} ثانية)... انتظر شوي",) + self.view(vod_id, uid)
        try:
            stream = vd.get_stream()
            pieces = snap_to_frames(pieces, stream.fps)
            version = int(coerce_number(clip.get("version"), 1) or 1) + 1
            base = f"clip_{int(clip['id']):02d}_{sanitize_filename(clip.get('title'))}_v{version}"
            out_mp4 = os.path.join(vd.vp.clips_dir, base + ".mp4")
            out_srt = os.path.join(vd.vp.clips_dir, base + ".srt")
            info = cut_and_stitch(stream, pieces, out_mp4, time_offset=vd.av_offset())
            srt = generate_srt(vd.words, pieces, out_srt, fps=stream.fps)
        except Exception as exc:
            yield (f"❌ فشل إعادة القص: {short_error(exc, 200)}",) + self.view(vod_id, uid)
            return

        for old in (clip.get("video_path"), clip.get("srt_path")):
            if old and os.path.isfile(old) and old not in (out_mp4, out_srt):
                try:
                    safe_delete(old, self.drive_root)
                except PermissionError:
                    pass
        net = pieces_net(pieces)
        trim_entry = {"at": now_iso(), "start_delta": sd, "end_delta": ed,
                      "removed_pieces": [i for i in range(len(segs)) if i not in keep_idx], "snap": bool(snap)}
        try:
            feats = compute_clip_features(pieces, vd.signals, critic=(clip.get("scores") or {}).get("critic"),
                                          category=clip.get("category"))
        except Exception:
            feats = clip.get("features") or {}
        clip.update({
            "segments": pieces, "start": pieces[0][0], "end": pieces[-1][1], "duration": round(net, 2),
            "removed_sec": round(pieces[-1][1] - pieces[0][0] - net, 2), "video_path": out_mp4, "srt_path": out_srt,
            "version": version, "features": feats, "status": "ready",
            "srt_stats": {k: srt[k] for k in ("cues", "min_cue_s", "avg_cps", "avg_wps", "bands")},
            "render": {k: info.get(k) for k in ("encoder", "size_mb", "height", "fps", "refreshed", "peak_mb")},
        })
        clip["manual_trims"] = list(clip.get("manual_trims") or []) + [trim_entry]
        idx, _ = find_clip(manifest, uid=uid)
        manifest["clips"][idx] = normalize_clip(clip, vd.vp.vod_id)
        save_manifest(vd.vp, manifest)

        entries = load_feedback(self.memory)
        fb = self.feedback_for(entries, vod_id, clip)
        msg_extra = ""
        if fb is not None:
            fb = dict(fb)
            fb.update({"segments": pieces, "features": feats, "duration": round(net, 2),
                       "manual_trims": list(fb.get("manual_trims") or []) + [trim_entry], "updated_at": now_iso()})
            entries = upsert_feedback(entries, fb)
            save_feedback(self.memory, entries)
            msg_extra = " | " + self._retrain(entries)
        yield (f"✅ تم إعادة القص والسبتايتل ({net:.1f} ثانية، {srt['cues']} سطر){msg_extra}",) + self.view(vod_id, uid)

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
        vd, manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return "❌ اختر لقطة أول", ""
        fps = (clip.get("render") or {}).get("fps")
        path = clip.get("srt_path") or os.path.join(vd.vp.clips_dir, f"clip_{int(clip['id']):02d}_{sanitize_filename(clip.get('title'))}.srt")
        srt = generate_srt(vd.words, clip.get("segments") or [], path, fps=fps)
        if clip.get("srt_path") != path:
            clip["srt_path"] = path
            idx, _ = find_clip(manifest, uid=uid)
            manifest["clips"][idx] = clip
            save_manifest(vd.vp, manifest)
        return f"✅ تم توليد السبتايتل من جديد ({srt['cues']} سطر)", srt["text"]

    def _feedback_entry(self, vod_id: str, clip: dict, rating: str, score10: Any, tags: list, note: str,
                        base: Optional[dict]) -> dict:
        entry = dict(base or {})
        entry.update({
            "uid": entry.get("uid") or clip["uid"],
            "vod_id": vod_id, "clip_id": clip.get("id"), "title": clip.get("title"),
            "category": clip.get("category"), "duration": clip.get("duration"),
            "removed_sec": clip.get("removed_sec"), "avg_wps": clip.get("avg_wps"),
            "excerpt": clip.get("excerpt"), "ai_reason": clip.get("reason"),
            "rating": rating, "score10": score10, "tags": list(tags or []), "note": str(note or ""),
            "segments": clip.get("segments"), "manual_trims": clip.get("manual_trims") or [],
            "updated_at": now_iso(), "schema": 2,
        })
        if clip.get("features"):
            entry["features"] = clip["features"]
            entry["features_source"] = "exact"
        return entry

    def save_rating(self, vod_id: str, uid: str, rating: str, score10: Any, tags: list, note: str) -> tuple:
        vd, manifest, clip = self.get_clip(vod_id, uid)
        if clip is None:
            return "❌ اختر لقطة أول", gr.update()
        score = coerce_number(score10, None)
        score = round(max(0.0, min(10.0, score)), 1) if score is not None else None
        rating = rating if rating in ("liked", "disliked", "pending") else "pending"
        if not clip.get("features"):
            try:
                clip["features"] = compute_clip_features(clip.get("segments") or [], vd.signals,
                                                         critic=(clip.get("scores") or {}).get("critic"),
                                                         category=clip.get("category"))
            except Exception:
                clip["features"] = {}
        clip.update({"rating": rating, "score10": score, "tags": list(tags or []), "note": str(note or "")})
        idx, _ = find_clip(manifest, uid=uid)
        manifest["clips"][idx] = clip
        save_manifest(vd.vp, manifest)

        entries = load_feedback(self.memory)
        base = self.feedback_for(entries, vod_id, clip)
        if rating == "pending" and score is None and not tags and not note:
            msg = "✅ تم حفظ اللقطة بدون تقييم"
        else:
            entries = upsert_feedback(entries, self._feedback_entry(vod_id, clip, rating, score, tags, note, base))
            save_feedback(self.memory, entries)
            msg = "✅ تم حفظ تقييمك | " + self._retrain(entries)
        return msg, gr.update(choices=self.clip_choices(vod_id), value=uid)

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

    def retrain_now(self) -> tuple:
        msg = self._retrain(load_feedback(self.memory))
        return msg, taste_summary_ar(load_profile(self.memory))

    # ------------------------------------------------------------ rules

    def _rules_history(self) -> list:
        files = sorted(glob.glob(os.path.join(self.memory.rules_history_dir, "rules_*.md")), reverse=True)
        return [(os.path.basename(f).replace("rules_", "").replace(".md", ""), f) for f in files]

    def _backup_rules(self) -> Optional[str]:
        current = read_text_safe(self.memory.rules, "")
        if not current.strip():
            return None
        stamp = now_utc().astimezone(RIYADH_TZ).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(ensure_dir(self.memory.rules_history_dir), f"rules_{stamp}.md")
        atomic_write_text(path, current)
        return path

    @staticmethod
    def _pinned(text: str) -> list:
        return [line.strip() for line in str(text or "").splitlines() if line.strip().startswith(PIN)]

    def save_rules_manual(self, text: str) -> tuple:
        if not str(text or "").strip():
            return "❌ القواعد فاضية، ما حفظنا شي", gr.update()
        self._backup_rules()
        atomic_write_text(self.memory.rules, str(text).strip() + "\n")
        return "✅ تم حفظ القواعد (والنسخة القديمة محفوظة في السجل)", gr.update(choices=self._rules_history(), value=None)

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
            lines.append(f"{mark}{score} «{e.get('title', '')}» [{e.get('category', '')}، {float(e.get('duration') or 0):.0f}ث] {tags} {note}".strip())
        trims = [t for e in entries for t in e.get("manual_trims") or []]
        trim_txt = ""
        if trims:
            ends = [float(t.get("end_delta") or 0) for t in trims]
            starts = [float(t.get("start_delta") or 0) for t in trims]
            trim_txt = (f"\nتعديلاته اليدوية: {len(trims)} تعديل، متوسط تغيير النهاية {sum(ends) / len(ends):+.1f} ثانية "
                        f"ومتوسط تغيير البداية {sum(starts) / len(starts):+.1f} ثانية.")
        pinned = self._pinned(current)
        pin_txt = ("\nهذي الأسطر مثبتة 📌 ولازم تبقى حرفياً بدون أي تغيير:\n" + "\n".join(pinned)) if pinned else ""
        prompt = (
            "أنت تساعد ممنتج مقاطع تيك توك يحدّث دليل ذوقه. هذي قواعده الحالية:\n\n"
            f"{current}\n\n"
            "وهذي آخر تقييماته للقطات اللي اختارها الذكاء الاصطناعي:\n" + "\n".join(lines) + trim_txt +
            f"\n\nأكثر أسباب الرفض: {', '.join(f'{k} ({v})' for k, v in list((prof.get('tag_counts') or {}).items())[:6])}\n"
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
        old_lines = [l.rstrip() for l in str(old).splitlines() if l.strip()]
        new_lines = [l.rstrip() for l in str(new).splitlines() if l.strip()]
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

    def usage(self) -> str:
        gw = self.gateway()
        return gw.usage_report_ar() if gw is not None else "ℹ️ مفتاح Gemini غير متوفر"

    # ------------------------------------------------------------ export and cleanup

    def export_zip(self, vod_id: str, only_liked: bool) -> tuple:
        vd = self.vod(vod_id)
        if vd is None:
            return "❌ اختر بث أول", None
        manifest = load_manifest(vd.vp)
        clips = [c for c in manifest.get("clips") or [] if c.get("video_path") and os.path.isfile(c["video_path"])]
        if only_liked:
            clips = [c for c in clips if c.get("rating") == "liked"]
        if not clips:
            return "ℹ️ ما فيه لقطات تطابق الاختيار", None
        out_dir = ensure_dir(os.path.join(TMP_ROOT, "exports"))
        stamp = now_utc().astimezone(RIYADH_TZ).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"{vod_id}_{'liked' if only_liked else 'all'}_{stamp}.zip")
        with zipfile.ZipFile(path, "w") as zf:
            for c in sorted(clips, key=lambda x: int(x.get("id") or 0)):
                zf.write(c["video_path"], os.path.basename(c["video_path"]), compress_type=zipfile.ZIP_STORED)
                if c.get("srt_path") and os.path.isfile(c["srt_path"]):
                    zf.write(c["srt_path"], os.path.basename(c["srt_path"]), compress_type=zipfile.ZIP_DEFLATED)
        return f"✅ جاهز: {len(clips)} لقطة ({human_size(os.path.getsize(path))})", path

    def _cleanup_targets(self, vd: _VodData) -> list:
        vp = vd.vp
        targets = []
        for path in (vp.audio, vp.clips_dir, vp.clips_archive_dir):
            if os.path.exists(path):
                targets.append(path)
        targets.extend(sorted(glob.glob(os.path.join(vp.vod_dir, "*.zip"))))
        return targets

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
        lines.append("")
        lines.append("✅ يبقى محفوظ: التفريغ النصي، الشات، الخصائص الصوتية، ملف اللقطات، وكل مجلد memory (القواعد والتقييمات والذوق).")
        lines.append("⚠️ بعد الحذف تبقى إعادة القص من الاستوديو شغالة (تقص من رابط البث)، لكن لقطات إضافية بالناقد الصوتي تحتاج تشغيل المرحلة 2 مرة ثانية عشان يرجع ملف الصوت.")
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
        tmp_audio = [os.path.join(TMP_ROOT, d, f"{vod_id}.mp3") for d in ("stage2", "stage3")]
        for p in tmp_audio:
            if os.path.isfile(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        manifest = load_manifest(vd.vp)
        for c in manifest.get("clips") or []:
            if c.get("video_path") and not os.path.isfile(c["video_path"]):
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

    demo = _blocks("استوديو اللقطات")
    with demo:
        gr.Markdown("# 🎬 استوديو مراجعة اللقطات والتعلم الذاتي")
        with gr.Row():
            vod_dd = _mk(gr.Dropdown, choices=studio.vod_choices(), value=first_vod, label="🎥 البث", scale=4, interactive=True)
            refresh_btn = _mk(gr.Button, value="🔄 تحديث القوائم", scale=1)
        status = _mk(gr.Markdown, value="")

        with gr.Tabs():
            with gr.Tab("✂️ المراجعة"):
                with gr.Row():
                    prev_btn = _mk(gr.Button, value="◀ السابقة", scale=1)
                    clip_dd = _mk(gr.Dropdown, choices=first_clips, value=first_uid, label="🎞️ اللقطة", scale=5, interactive=True)
                    next_btn = _mk(gr.Button, value="التالية ▶", scale=1)
                with gr.Row():
                    with _mk(gr.Column, scale=3):
                        video = _mk(gr.Video, label="المعاينة", interactive=False, height=420)
                        info_md = _mk(gr.Markdown, value="")
                    with _mk(gr.Column, scale=2):
                        pieces_cb = _mk(gr.CheckboxGroup, choices=[], value=[], label="✂️ القطع (شيل العلامة عن القطعة اللي تبي تحذفها)")
                        with gr.Row():
                            start_minus = _mk(gr.Button, value="البداية ◀ 0.5ث")
                            start_plus = _mk(gr.Button, value="البداية ▶ 0.5ث")
                            end_minus = _mk(gr.Button, value="النهاية ◀ 0.5ث")
                            end_plus = _mk(gr.Button, value="النهاية ▶ 0.5ث")
                        with gr.Row():
                            start_delta = _mk(gr.Number, value=0.0, label="تغيير البداية (ث، السالب أبكر)", step=NUDGE)
                            end_delta = _mk(gr.Number, value=0.0, label="تغيير النهاية (ث، الموجب أطول)", step=NUDGE)
                        snap_cb = _mk(gr.Checkbox, value=True, label="التزم بحدود الكلمات")
                        recut_btn = _mk(gr.Button, value="🔁 إعادة القص والسبتايتل", variant="primary")
                        srt_box = _mk(gr.Textbox, label="📝 السبتايتل (SRT)", lines=12, max_lines=30, rtl=True, interactive=True)
                        with gr.Row():
                            srt_save_btn = _mk(gr.Button, value="✅ تحقق واحفظ")
                            srt_regen_btn = _mk(gr.Button, value="♻️ ولّد من جديد")
                        rating = _mk(gr.Radio, choices=RATING_CHOICES, value="pending", label="التقييم")
                        score = _mk(gr.Number, value=None, label="الدرجة من 10 (اختياري)", minimum=0, maximum=10, step=0.5)
                        tags = _mk(gr.CheckboxGroup, choices=POSITIVE_TAGS + NEGATIVE_TAGS, value=[], label="وسوم سريعة")
                        note = _mk(gr.Textbox, label="ملاحظتك", lines=3, rtl=True)
                        save_btn = _mk(gr.Button, value="💾 احفظ التقييم ودرّب الذوق", variant="primary")

            with gr.Tab("🧠 الذوق"):
                taste_md = _mk(gr.Markdown, value=taste_summary_ar(load_profile(studio.memory)))
                with gr.Row():
                    train_btn = _mk(gr.Button, value="🔄 درّب الذوق الرقمي الحين")
                    usage_btn = _mk(gr.Button, value="📊 استهلاك Gemini اليوم")
                usage_md = _mk(gr.Markdown, value="")
                rules_box = _mk(gr.Textbox, value=read_text_safe(studio.memory.rules, ""), label="📜 قواعد الممنتج (الأسطر اللي تبدأ بـ 📌 مثبتة)",
                                lines=14, max_lines=40, rtl=True, interactive=True)
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

            with gr.Tab("📦 التصدير"):
                only_liked = _mk(gr.Checkbox, value=True, label="اللقطات اللي عجبتني فقط 👍")
                zip_btn = _mk(gr.Button, value="📦 جهّز ملف ZIP", variant="primary")
                zip_file = _mk(gr.File, label="ملف التحميل", interactive=False)
                gr.Markdown("### 🧹 تنظيف مساحة Drive")
                cleanup_md = _mk(gr.Markdown, value=studio.cleanup_info(first_vod) if first_vod else "")
                with gr.Row():
                    size_btn = _mk(gr.Button, value="🔎 احسب المساحة")
                    confirm_cb = _mk(gr.Checkbox, value=False, label="أؤكد إني صدّرت اللي أبيه وأبي أحذف ملفات هذا البث الثقيلة")
                cleanup_btn = _mk(gr.Button, value="🧹 تنظيف مساحة هذا البث من Drive", variant="stop")

        view_outputs = [video, info_md, pieces_cb, srt_box, rating, score, tags, note, start_delta, end_delta]

        # ---------------- wiring

        def on_vod(vod_id):
            choices = studio.clip_choices(vod_id)
            uid = choices[0][1] if choices else None
            return (gr.update(choices=choices, value=uid),) + studio.view(vod_id, uid) + (studio.cleanup_info(vod_id),)

        vod_dd.change(on_vod, inputs=[vod_dd], outputs=[clip_dd] + view_outputs + [cleanup_md])

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

        recut_btn.click(studio.recut, inputs=[vod_dd, clip_dd, pieces_cb, start_delta, end_delta, snap_cb],
                        outputs=[status] + view_outputs)
        srt_save_btn.click(studio.save_srt, inputs=[vod_dd, clip_dd, srt_box], outputs=[status])
        srt_regen_btn.click(studio.regen_srt, inputs=[vod_dd, clip_dd], outputs=[status, srt_box])
        save_btn.click(studio.save_rating, inputs=[vod_dd, clip_dd, rating, score, tags, note], outputs=[status, clip_dd])

        train_btn.click(studio.retrain_now, inputs=[], outputs=[status, taste_md])
        usage_btn.click(studio.usage, inputs=[], outputs=[usage_md])
        rules_save_btn.click(studio.save_rules_manual, inputs=[rules_box], outputs=[status, history_dd])
        propose_btn.click(studio.propose_rules, inputs=[], outputs=[status, proposal_box, diff_md])
        approve_btn.click(studio.approve_rules, inputs=[proposal_box], outputs=[status, rules_box, history_dd])
        discard_btn.click(studio.discard_rules, inputs=[], outputs=[status, proposal_box, diff_md])
        restore_btn.click(studio.restore_rules, inputs=[history_dd], outputs=[status, rules_box, history_dd])

        zip_btn.click(studio.export_zip, inputs=[vod_dd, only_liked], outputs=[status, zip_file])
        size_btn.click(studio.cleanup_info, inputs=[vod_dd], outputs=[cleanup_md])
        cleanup_btn.click(studio.cleanup, inputs=[vod_dd, confirm_cb], outputs=[status, confirm_cb, cleanup_md])

        demo.load(lambda v, u: studio.view(v, u), inputs=[vod_dd, clip_dd], outputs=view_outputs)
    return demo


def launch_studio(vod_dir: Optional[str] = None, drive_root: str = DRIVE_ROOT_DEFAULT,
                  api_key: Optional[str] = None, share: bool = True, server_port: Optional[int] = None) -> Any:
    """Build and launch the Studio. Returns the Blocks object."""
    t0 = time.time()
    log(f"🎛️ جاري تشغيل الاستوديو (Gradio {getattr(gr, '__version__', '?')})...")
    ensure_dir(os.path.join(TMP_ROOT, "exports"))
    studio = Studio(drive_root, api_key)
    demo = build_ui(studio, vod_dir)
    try:
        demo.queue()
    except TypeError:
        pass
    _launch(demo, share=share, server_port=server_port, allowed_paths=[drive_root, TMP_ROOT],
            show_error=True, inbrowser=False, quiet=False)
    log(f"✅ الاستوديو جاهز خلال {format_duration_ar(time.time() - t0)}")
    return demo


# === END OF stage4_studio.py ===
