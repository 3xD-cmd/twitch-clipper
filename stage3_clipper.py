"""
stage3_clipper.py
Stage 3 of TwitchClipper (Colab CPU, v3): high recall selection, a Critic that watches
video, protected visual beats, and drafts or full renders.

Pipeline (every step checkpointed on Drive under vods/<id>/stage3/run_<run_id>/)
  1. Phrase index with line ids (frozen per run so saved answers always map to the same lines)
  2. Local scores (taste weighted, Golden priors blended in), 🔥 hints, Reaction Radar 👀 hints
  3. Active timeline (timeline.json): AFK and dead stretches removed, adaptive chunks of about
     45 active minutes cut at speech gaps with 90 s overlap
  4. AFK aware request plan (gw.plan on active chunks only) with a low budget fallback order:
     smaller Critic pool, then batch 4, then longer chunks up to 55 active minutes
  5. Scout v3 (one request per chunk): 4 to 12 candidates, visual comedy radar rule, Golden DNA
  6. Radar candidates: radar peaks that no Scout candidate covers go straight to the Critic
  7. Merge, dedupe, rank, Critic pool capped by active hours (radar candidates always included)
  8. One shared StreamSet and one cached A/V offset for Critic windows, previews and renders
  9. Window media in a background thread: a small Critic video in /tmp and a Studio preview on
     Drive, prepared for the next batch while Gemini answers the current one (audio fallback)
 10. Multimodal Critic v3 (3 candidates per request): video + transcript + scout hypothesis +
     Golden DNA; visual beats become protected spans; skipped candidates get one more chance
 11. Cut Doctor with protected beats, final ranking with the taste model
 12. Clip ids reserved in the manifest before any file is written (ids are never reused)
 13. render_mode "on_approve" (default): drafts with preview, initial SRT and CapCut notes;
     the 1080p render happens in the Studio after approval.
     render_mode "all": full quality render (render_pieces) + SRT + notes now.

QuotaPause (daily quota or a long 503 storm) saves everything, flushes the gateway state and
returns cleanly; the next run resumes the same run directory without repeating finished
chunks, radar work or Critic batches.

Public contract
  run_stage3(vod_dir, drive_root, api_key, extra_clips=False, redo_all=False,
             auto_trim_silence=True, max_clips=None, allow_unreviewed=False,
             max_height=1080, srt_bom=False, critic_video=True, render_mode="on_approve",
             chunk_minutes=45.0, critic_batch=3) -> list[clip dict] (manifest clips)
  write_clip_notes(clip, path) -> str
  load_av_offset(vp, streams=None, duration=0.0) -> float      (shared with the Studio)
  list_run_candidates(vp, run_id=None) -> (run_id, candidates)  (Studio candidate browser)
  RENDER_MODES, CATEGORIES, REJECT_CODES
"""

from __future__ import annotations

import bisect
import concurrent.futures
import math
import os
import shutil
import time
from collections import Counter
from typing import Any, Optional, Sequence

import numpy as np
from pydantic import BaseModel, Field, field_validator

from common import (
    TMP_ROOT,
    VodPaths,
    allocate_clip_id,
    atomic_write_json,
    atomic_write_text,
    clamp,
    coerce_int,
    coerce_number,
    ensure_dir,
    format_duration_ar,
    load_feedback,
    load_json_safe,
    load_manifest,
    log,
    make_clip_uid,
    merge_spans,
    mmss_to_sec,
    new_run_id,
    normalize_clip,
    now_iso,
    parse_time_to_sec,
    read_text_safe,
    safe_delete,
    sanitize_filename,
    save_manifest,
    sec_to_clock,
    sec_to_hms,
    sec_to_mmss,
    short_error,
    spans_total,
    subtract_spans,
    text_hash,
)
from gemini_gateway import GatewayError, GeminiGateway, ParseFailure, QuotaPause, video_item
from golden_engine import golden_context, golden_stats, load_golden
from media_engine import (
    StreamSet,
    audio_snippet,
    build_phrases,
    collect_words,
    cut_doctor,
    generate_srt,
    ids_to_pieces,
    load_features,
    make_window_media,
    measure_av_offset,
    pieces_net,
    render_pieces,
    resolve_streams,
    snap_to_frames,
)
from taste_engine import (
    apply_golden_priors,
    build_signals,
    compute_clip_features,
    doctor_prefs,
    load_profile,
    local_score_for_span,
    mark_hot_phrases,
    mark_radar_phrases,
    predict_taste,
    prompt_context,
    radar_peaks,
    score_windows,
)


# ============================================================================
# Configuration
# ============================================================================

RENDER_MODES = ("on_approve", "all")

# Active timeline and chunks
DEAD_MIN_S = 120.0
DEAD_REACTION_THR = 0.3
DEAD_EDGE_KEEP_S = 5.0
CHUNK_ACTIVE_S_DEFAULT = 45 * 60
CHUNK_ACTIVE_MIN_S = 15 * 60
CHUNK_ACTIVE_MAX_S = 55 * 60
CHUNK_TOKEN_CAP = 45_000
CHUNK_PROMPT_OVERHEAD = 4_500
OVERLAP_S = 90.0
BOUNDARY_SNAP_S = 90.0
COMPACT_GUARD_S = 20.0

# Scout
CLIPS_MIN = 4
CLIPS_MAX = 12
MIN_CAND_NET = 4.0
MAX_CAND_NET = 90.0

# Radar
RADAR_PER_HOUR = 3.0
RADAR_MIN_SCORE = 0.45
RADAR_MIN_GAP_S = 60.0
RADAR_PRE_S = 25.0
RADAR_POST_S = 8.0
RADAR_SCOUT_SCORE = 6
RADAR_COVER_PAD_S = 3.0

# Critic pool and batches
POOL_PER_HOUR = 6.0
POOL_MIN = 12
POOL_MAX = 36
POOL_LOW_FLOOR = 20
CRITIC_BATCH_DEFAULT = 3
CRITIC_BATCH_LOW = 4
CRITIC_TEXT_PRE_S = 60.0
CRITIC_TEXT_POST_S = 20.0
MAX_CRITIC_PASSES = 3
MAX_PARSE_FAILS = 2

# Window media
WINDOW_PRE_S = 12.0
WINDOW_POST_S = 6.0
WINDOW_MAX_S = 120.0
CRITIC_FILE_FPS = 4.0
CRITIC_SEND_FPS = 2.0
CRITIC_HEIGHT = 360
PREVIEW_HEIGHT = 360
PREVIEW_PAD_S = 15.0
MEDIA_WAIT_S = 900.0
BEAT_PAD_S = 0.4
BEAT_EXTEND_BEFORE_S = 20.0
BEAT_EXTEND_AFTER_S = 8.0

# Audio fallback (v2 values)
AUDIO_PRE_S = 15.0
AUDIO_POST_S = 10.0
AUDIO_MAX_S = 150.0

# Final selection and render
MIN_FINAL_NET = 8.0
HARD_MAX_NET = 60.0
AV_OFFSET_MIN = 0.04
MAX_CUT_ATTEMPTS = 2

CATEGORIES = ("ضحك", "صراخ_وتوتر", "تفاعل_شات", "سالفة", "قيمينق", "مقلب_وغدر")
_CATEGORY_ALIASES = {
    "ضحك": "ضحك", "كوميدي": "ضحك", "مضحك": "ضحك", "فشل": "ضحك",
    "صراخ": "صراخ_وتوتر", "توتر": "صراخ_وتوتر", "رعب": "صراخ_وتوتر", "صراخ وتوتر": "صراخ_وتوتر", "فجعة": "صراخ_وتوتر",
    "تفاعل شات": "تفاعل_شات", "شات": "تفاعل_شات", "تفاعل": "تفاعل_شات",
    "سالفة": "سالفة", "قصة": "سالفة", "سوالف": "سالفة",
    "قيمينق": "قيمينق", "لعب": "قيمينق", "gaming": "قيمينق",
    "مقلب": "مقلب_وغدر", "غدر": "مقلب_وغدر", "مقلب وغدر": "مقلب_وغدر", "خيانة": "مقلب_وغدر", "دفة": "مقلب_وغدر",
}
REJECT_CODES = ("كلام_عادي", "بدون_ذروة", "صراخ_لعبة_فقط", "سياق_ناقص", "نهاية_مبتورة",
                "طويلة_مملة", "موضوع_ممنوع", "مكرر")
RUBRIC_KEYS = ("hook", "payoff", "reaction", "context", "pacing", "visual")
_TEXT_RUBRIC_KEYS = ("hook", "payoff", "reaction", "context", "pacing")


def _norm_category(value: Any) -> str:
    text = str(value or "").strip()
    if text in CATEGORIES:
        return text
    key = text.replace("_", " ").strip()
    for alias, cat in _CATEGORY_ALIASES.items():
        if alias == key or alias in key:
            return cat
    return "ضحك"


def _lenient_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "نعم", "اي", "ايه", "keep", "صح")


def _lenient_text(v: Any, limit: int = 400) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        v = "، ".join(str(x) for x in v if str(x).strip())
    return " ".join(str(v).split())[:limit]


def _mmss_text(v: Any) -> str:
    if isinstance(v, (int, float)):
        return sec_to_mmss(v)
    sec = mmss_to_sec(v)
    return sec_to_mmss(sec) if sec is not None else ""


# ============================================================================
# Structured output schemas (lenient, as in v2)
# ============================================================================

class IdRange(BaseModel):
    from_id: int
    to_id: int

    @field_validator("from_id", "to_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)


class Topic(BaseModel):
    from_id: int = -1
    to_id: int = -1
    topic: str = ""

    @field_validator("from_id", "to_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)

    @field_validator("topic", mode="before")
    @classmethod
    def _topic(cls, v: Any) -> str:
        return _lenient_text(v, 160)


class ScoutClip(BaseModel):
    title: str = "لقطة"
    category: str = "ضحك"
    ranges: list[IdRange] = []
    peak_id: int = -1
    payoff_end_id: int = -1
    context_note: str = ""
    removed: list[str] = []
    scout_score: int = 0
    why: str = ""
    visual_guess: str = ""
    needs_visual: bool = False
    confidence: int = 5

    @field_validator("peak_id", "payoff_end_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)

    @field_validator("scout_score", mode="before")
    @classmethod
    def _score(cls, v: Any) -> int:
        return coerce_int(v, 0, lo=0, hi=10)

    @field_validator("confidence", mode="before")
    @classmethod
    def _confidence(cls, v: Any) -> int:
        return coerce_int(v, 5, lo=1, hi=10)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v: Any) -> str:
        return _norm_category(v)

    @field_validator("title", "context_note", "why", "visual_guess", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _lenient_text(v, 400)

    @field_validator("needs_visual", mode="before")
    @classmethod
    def _bool(cls, v: Any) -> bool:
        return _lenient_bool(v)

    @field_validator("removed", mode="before")
    @classmethod
    def _removed(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v or []]


class ScoutResponse(BaseModel):
    clips: list[ScoutClip] = []
    topics: list[Topic] = []


class Rubric(BaseModel):
    hook: int = 5
    payoff: int = 5
    reaction: int = 5
    context: int = 5
    pacing: int = 5
    visual: int = 5

    @field_validator("hook", "payoff", "reaction", "context", "pacing", "visual", mode="before")
    @classmethod
    def _score(cls, v: Any) -> int:
        return coerce_int(v, 5, lo=0, hi=10)


class VisualBeat(BaseModel):
    start: str = ""
    end: str = ""
    what: str = ""

    @field_validator("start", "end", mode="before")
    @classmethod
    def _mmss(cls, v: Any) -> str:
        return _mmss_text(v)

    @field_validator("what", mode="before")
    @classmethod
    def _what(cls, v: Any) -> str:
        return _lenient_text(v, 160)


class CriticVerdict(BaseModel):
    candidate: int = -1
    keep: bool = False
    reject: list[str] = []
    ranges: list[IdRange] = []
    end_anchor_id: Optional[int] = None
    rubric: Rubric = Field(default_factory=Rubric)
    voice_check: str = "her"
    long_ok: bool = False
    title: str = ""
    category: str = ""
    cut_notes: str = ""
    visual_event: str = ""
    visual_beats: list[VisualBeat] = []
    hook_line: str = ""

    @field_validator("candidate", mode="before")
    @classmethod
    def _cand(cls, v: Any) -> int:
        return coerce_int(v, -1)

    @field_validator("end_anchor_id", mode="before")
    @classmethod
    def _anchor(cls, v: Any) -> Optional[int]:
        if v is None or v == "":
            return None
        val = coerce_int(v, -1)
        return val if val >= 0 else None

    @field_validator("keep", "long_ok", mode="before")
    @classmethod
    def _bool(cls, v: Any) -> bool:
        return _lenient_bool(v)

    @field_validator("rubric", mode="before")
    @classmethod
    def _rubric(cls, v: Any) -> Any:
        return v if isinstance(v, dict) else {}

    @field_validator("voice_check", mode="before")
    @classmethod
    def _voice(cls, v: Any) -> str:
        text = str(v or "").strip().lower()
        if "game" in text or "لعب" in text:
            return "game"
        if "mix" in text or "مختلط" in text:
            return "mixed"
        return "her"

    @field_validator("reject", mode="before")
    @classmethod
    def _reject(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v or [] if str(x).strip()]

    @field_validator("title", "category", "cut_notes", "visual_event", "hook_line", mode="before")
    @classmethod
    def _text(cls, v: Any) -> str:
        return _lenient_text(v, 400)

    @field_validator("visual_beats", mode="before")
    @classmethod
    def _beats(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, dict):
            v = [v]
        if isinstance(v, str):
            out = []
            for part in v.replace("،", ";").split(";"):
                times = [t for t in part.replace("إلى", "-").replace("الى", "-").split("-") if t.strip()]
                if times:
                    out.append({"start": times[0], "end": times[-1], "what": ""})
            return out
        out = []
        for item in v if isinstance(v, (list, tuple)) else []:
            if isinstance(item, dict):
                out.append(item)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                out.append({"start": item[0], "end": item[1], "what": item[2] if len(item) > 2 else ""})
        return out[:8]


class CriticResponse(BaseModel):
    verdicts: list[CriticVerdict] = []


# ============================================================================
# Prompts (Arabic)
# ============================================================================

LINE_FORMAT_AR = """طريقة قراءة النص:
• كل سطر يبدأ برقم مثل #412 ثم الوقت ثم كلام الستريمر.
• أسطر بين قوسين مثل (ضحك 2.4ث) أو (صراخ 1.2ث) أو (صوت لعبة عالي 1.8ث) أو (صمت 3.1ث) هي أحداث صوتية بدون كلام، ولها أرقام وتقدر تدخلها أو تحذفها.
• العلامة 🔥 تعني مؤشرات صوتية قوية حول هذا السطر، والوسم [ضحك] أو [صراخ] يعني فيه ضحك أو صراخ أثناء الكلام.
• العلامة 👀 تعني الرادار لقط ردة فعل بدون كلام يمهد لها: غالباً صار شي على الشاشة.
• أسطر 💬 هي عينة من شات تويتش، بدون أرقام، ما تنفع كبداية أو نهاية.
• سطر ⏭️ يعني تخطينا وقت ميت (AFK أو سكوت طويل) وما فيه شي.
• الستريمر تبث على تويتش وتيك توك مع بعض، فممكن تقرأ أو ترد على تعليق ما يظهر في شات تويتش. هذا تفاعل حقيقي، لا تحذفه."""

SCOUT_SYSTEM_AR = """أنت كشاف لقطات لمقاطع تيك توك وريلز، متخصص في بثوث تويتش عربية بلهجة خليجية سعودية.
مهمتك ما يفوتك شي: اقترح كل لحظة محتملة في هذا الجزء. بعدك ناقد يشوف فيديو كل لقطة ويفلتر، فالأفضل تقترح لقطة زيادة من إنك تفوّت لقطة حلوة.

قواعد إلزامية:
1. اقترح من 4 إلى 12 لقطة حسب الجزء. القائمة الفاضية مسموحة فقط إذا الجزء ما فيه أي ضحك أو ردة فعل أبداً.
2. رادار الكوميديا البصرية: سطر عليه 👀 فيه ضحك أو اعتذار (سوري، ما قصدت، بالغلط) أو فوضى (وش سويت، اهرب، لا لا لا) وما قبله نكتة كلامية، معناه صار شي على الشاشة (دفة، طيحة، غدر، موتة). اقترحه حتى لو الكلام قبله عادي أو ساكت، واكتب توقعك لوش صار في visual_guess وخل needs_visual صح.
3. كل لقطة لها ذروة تحدد سطرها في peak_id: ضحكة قوية، فجعة، قفلة مضحكة، غدر، صدمة.
4. احفظ النهاية: النطاقات لازم تمتد لين payoff_end_id، وأي سطر ضحك أو صراخ بعد الذروة إلزامي. لا تقطع وهي تضحك.
5. البداية: ابدأ من أول سطر يحتاجه مشاهد جديد عشان يفهم. في اللقطات البصرية خل من 10 إلى 20 ثانية قبل الضحكة حتى لو كانت صمت أو كلام عادي، لأن الحدث يصير فيها.
6. القص الذكي: مسموح تحذف ترحيب بالشات خارج السالفة، مواضيع جانبية، وأسطر الصمت البعيدة عن الحدث. ممنوع تحذف أسطر الضحك أو السياق أو قراءتها وردها على التعليقات، وممنوع تحذف الصمت اللي قبل ضحكة بصرية.
7. ممنوع نهائياً: بريك الصلاة، كلام الميكب، التذمر العادي، المشي الطويل بدون حدث، صوت اللعبة بدون ردة فعل منها.
8. افهم السياق: لا تخلط بين الألعاب أو المواضيع، واكتب في context_note وش اللعبة أو الموضوع ومين الأطراف.
9. المدة الصافية بعد القص من 12 إلى 45 ثانية غالباً.
10. scout_score من 1 إلى 10 لقوة اللقطة بصراحة، و confidence من 1 إلى 10 لثقتك إن فيه لحظة فعلاً.
11. استخدم أرقام الأسطر فقط في ranges و peak_id و payoff_end_id، ولا تكتب أوقات.

""" + LINE_FORMAT_AR

CRITIC_SYSTEM_AR = """أنت الناقد النهائي والممنتج الأخير قبل النشر على تيك توك. معك لكل مرشح فيديو قصير بصوته (أو صوت فقط إذا الفيديو غير متوفر) ونص بأرقام الأسطر.
شوف الفيديو واسمع الصوت فعلاً قبل الحكم: وش صار على الشاشة، هل الضحك أو الصراخ منها أو من اللعبة، وهل الضحكة خلصت.
احكم على اللي على الشاشة: إذا الضحك جا بعد حدث بصري (دفة، طيحة، غدر، موتة، شي غريب)، القص لازم يشمل الحدث كامل من بدايته. الضحكة اللي ما قبلها كلام مو سبب رفض إذا الفيديو يوضح السبب.

لكل مرشح احكم:
• keep: true فقط إذا اللقطة تستاهل النشر فعلاً حسب ذوق الممنتج. اللقطة العادية أو الباردة ترفضها.
• voice_check: "her" إذا ردة الفعل منها، "game" إذا الصوت العالي من اللعبة فقط، "mixed" إذا الاثنين.
• ranges: القص النهائي بأرقام الأسطر داخل النافذة المسموحة فقط. ابدأ من الهوك، احذف الترحيب والمواضيع الجانبية والصمت البعيد، ولا تحذف أي ضحك أو سياق أو صمت يصير فيه الحدث البصري.
• end_anchor_id: آخر سطر لازم يدخل عشان النهاية تكتمل (نهاية الضحكة أو ردة الفعل).
• visual_event: جملة وحدة بالعربي وش صار على الشاشة (فاضية إذا ما فيه شي بصري مهم).
• visual_beats: لحظات الشاشة اللي لازم تبقى في القص، كل وحدة {start, end, what} بصيغة MM:SS محسوبة من بداية فيديو هذا المرشح نفسه.
• rubric من 0 إلى 10: hook قوة البداية، payoff قوة القفلة، reaction صدق ردة الفعل، context وضوح السياق، pacing سرعة الإيقاع، visual قد إيش اللقطة تعتمد على الشاشة وتوصله (0 ما لها علاقة بالشاشة، 10 الحدث البصري هو كل شي).
• reject: أسباب الرفض من هذه فقط: كلام_عادي، بدون_ذروة، صراخ_لعبة_فقط، سياق_ناقص، نهاية_مبتورة، طويلة_مملة، موضوع_ممنوع، مكرر.
• long_ok: true فقط إذا السالفة تحتاج أكثر من 45 ثانية فعلاً.
• title: عنوان قصير جذاب بالعامية (2 إلى 6 كلمات)، category من: ضحك، صراخ_وتوتر، تفاعل_شات، سالفة، قيمينق، مقلب_وغدر.
• hook_line: جملة قصيرة تنكتب على الشاشة في CapCut كهوك (نص فقط، ما تنحرق في الفيديو).
• cut_notes: ملاحظة قصيرة للممنتج عن القص.
لازم ترجع حكم لكل مرشح برقمه، حتى لو رفضته.

""" + LINE_FORMAT_AR


# ============================================================================
# Context
# ============================================================================

class _Ctx:
    """Everything a run needs, loaded once."""

    def __init__(self, vp: VodPaths, drive_root: str):
        self.vp = vp
        self.drive_root = drive_root
        self.memory = vp.memory.ensure()
        self.transcript = load_json_safe(vp.transcript, default={}, expected_types=(dict,)) or {}
        if not self.transcript.get("segments"):
            raise FileNotFoundError("❌ ما لقيت تفريغ نصي لهذا البث. شغّل المرحلة 2 أول")
        self.words = collect_words(self.transcript)
        self.feats = load_features(vp, words=self.words, allow_build=True, save_built=True)
        self.duration = float(coerce_number(self.transcript.get("duration"), 0.0) or 0.0)
        if self.feats is not None:
            self.duration = max(self.duration, self.feats.duration)
        if self.duration <= 0 and self.words:
            self.duration = self.words[-1]["e"]
        self.chat = load_json_safe(vp.chat, default=[], expected_types=(list,), quiet=True) or []
        self.afk = _clean_afk(self.transcript.get("afk") or [], self.duration)
        self.meta = load_json_safe(vp.meta, default={}, expected_types=(dict,), quiet=True) or {}
        self.vod_url = self.meta.get("vod_url") or f"https://www.twitch.tv/videos/{vp.vod_id}"

        self.golden = load_golden(self.memory)
        g_stats = self.golden.get("stats") or golden_stats(self.golden)
        self.profile = apply_golden_priors(load_profile(self.memory), g_stats, fade_n=35)
        self.rules = read_text_safe(self.memory.rules, "")
        self.feedback = load_feedback(self.memory)
        scout_golden = golden_context(self.golden, "scout")
        self.taste_scout = prompt_context(self.profile, self.rules, self.feedback, golden_text=scout_golden)
        self.golden_info = self.profile.get("golden") or {}

        self.phrases: list = []
        self.by_id: dict = {}
        self.phrase_starts: list = []
        self.signals = None
        self.scores: dict = {}
        self.hot_ids: set = set()
        self.radar_ids: set = set()
        self.guard_ids: set = set()
        self.chat_marks: dict = {}
        self.timeline: dict = {}
        self.active: list = [[0.0, round(self.duration, 3)]]
        self._active_starts: list = [0.0]

        self.streams: Optional[StreamSet] = None
        self.streams_error = ""
        self.offset: Optional[float] = None
        self.max_height = 1080
        self._local_audio = ""

    # -------------------------------------------------- phrases and signals

    def set_phrases(self, phrases: list) -> None:
        self.phrases = phrases
        self.by_id = {int(p["id"]): p for p in phrases}
        self.phrase_starts = [float(p["s"]) for p in phrases]
        self.signals = build_signals(self.feats, self.words, phrases, self.chat, self.duration)
        self.scores = score_windows(self.signals, self.profile)
        self.hot_ids = set(mark_hot_phrases(self.scores, phrases))
        self.chat_marks = _chat_bursts(self.chat, phrases)

    def set_radar(self, radar_ids: Sequence[int]) -> None:
        self.radar_ids = set(int(i) for i in radar_ids)
        self._rebuild_guard()

    def _rebuild_guard(self) -> None:
        marked = [self.by_id[i] for i in (self.hot_ids | self.radar_ids) if i in self.by_id]
        guard = set()
        if marked:
            times = sorted(float(p["s"]) for p in marked)
            for ph in self.phrases:
                k = bisect.bisect_left(times, float(ph["s"]) - COMPACT_GUARD_S)
                if k < len(times) and times[k] <= float(ph["e"]) + COMPACT_GUARD_S:
                    guard.add(int(ph["id"]))
        self.guard_ids = guard

    # -------------------------------------------------- active timeline

    def set_timeline(self, timeline: dict) -> None:
        self.timeline = timeline
        active = [[float(a), float(b)] for a, b in timeline.get("active") or [] if float(b) > float(a)]
        self.active = active or [[0.0, round(self.duration, 3)]]
        self._active_starts = [a for a, _ in self.active]

    def in_active(self, t: float) -> bool:
        k = bisect.bisect_right(self._active_starts, t) - 1
        return k >= 0 and self.active[k][0] <= t < self.active[k][1]

    @property
    def active_seconds(self) -> float:
        return spans_total(self.active)

    @property
    def active_hours(self) -> float:
        return max(0.1, self.active_seconds / 3600.0)

    # -------------------------------------------------- media helpers

    @property
    def fps(self) -> Optional[float]:
        return self.streams.master.fps if self.streams is not None else None

    def get_streams(self, max_height: Optional[int] = None) -> Optional[StreamSet]:
        """Resolve the shared StreamSet once. Returns None when resolving failed (the error
        is kept so we do not retry yt-dlp for every candidate)."""
        if self.streams is not None:
            return self.streams
        if self.streams_error:
            return None
        try:
            self.streams = resolve_streams(self.vod_url, max_height=int(max_height or self.max_height))
        except Exception as exc:
            self.streams_error = short_error(exc, 200)
            log(f"⚠️ تعذر استخراج روابط البث ({self.streams_error})")
            return None
        return self.streams

    def get_offset(self) -> float:
        if self.offset is None:
            self.offset = load_av_offset(self.vp, self.streams, self.duration)
        return self.offset

    def local_audio(self) -> str:
        """Local copy of audio.mp3 for audio snippets (fast reads, Drive is slow)."""
        if self._local_audio and os.path.isfile(self._local_audio):
            return self._local_audio
        path = os.path.join(ensure_dir(os.path.join(TMP_ROOT, "stage3")), f"{self.vp.vod_id}.mp3")
        if not os.path.isfile(path) or os.path.getsize(path) != os.path.getsize(self.vp.audio):
            shutil.copyfile(self.vp.audio, path)
        self._local_audio = path
        return path

    def media_tmp(self, run_id: str) -> str:
        return ensure_dir(os.path.join(TMP_ROOT, "stage3", self.vp.vod_id, f"run_{run_id}"))


def _clean_afk(raw: Any, duration: float) -> list:
    spans = []
    for item in raw or []:
        s = e = None
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            s, e = coerce_number(item[0], None), coerce_number(item[1], None)
        elif isinstance(item, dict):
            s, e = coerce_number(item.get("start", item.get("s")), None), coerce_number(item.get("end", item.get("e")), None)
        if s is not None and e is not None and e > s:
            spans.append([max(0.0, float(s)), min(float(duration), float(e))])
    return merge_spans(spans)


def _chat_bursts(chat: Sequence[dict], phrases: Sequence[dict], min_msgs: int = 3) -> dict:
    """phrase id -> (count, samples) for phrases that have a burst of at least min_msgs
    chat messages within 5 s of their span."""
    if not chat or not phrases:
        return {}
    times = sorted((coerce_number(m.get("t"), None), str(m.get("m", ""))) for m in chat
                   if isinstance(m, dict) and coerce_number(m.get("t"), None) is not None)
    out = {}
    j = 0
    for ph in phrases:
        if ph.get("kind") != "speech":
            continue
        lo, hi = ph["s"] - 5.0, ph["e"] + 5.0
        while j < len(times) and times[j][0] < lo:
            j += 1
        k = j
        msgs = []
        while k < len(times) and times[k][0] <= hi:
            msgs.append(times[k][1])
            k += 1
        if len(msgs) >= min_msgs:
            out[int(ph["id"])] = (len(msgs), [m[:30] for m in msgs[:3]])
    return out


# ============================================================================
# Shared A/V offset (Stage 3 and Studio)
# ============================================================================

def load_av_offset(vp: VodPaths, streams: Optional[StreamSet] = None, duration: float = 0.0) -> float:
    """A/V offset between audio.mp3 and the stream, measured once per VOD on the master
    rendition and cached in stage3/av_offset.json. Without streams only the cache is read."""
    path = os.path.join(vp.stage3_dir, "av_offset.json")
    cached = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
    if cached and "offset" in cached:
        return float(cached.get("offset") or 0.0)
    if streams is None:
        return 0.0
    log("🎚️ فحص تزامن الصوت بين ملف الصوت ورابط الفيديو (مرة وحدة للبث)...")
    try:
        offset = measure_av_offset(streams.master, vp.audio, float(duration))
    except Exception as exc:
        log(f"   ⚠️ تعذر قياس التزامن ({short_error(exc, 80)})، نفترض صفر")
        offset = None
    value = float(offset) if offset is not None and abs(offset) > AV_OFFSET_MIN else 0.0
    atomic_write_json(path, {"offset": value, "measured": offset, "at": now_iso()}, backup=False)
    if value:
        log(f"   🎚️ تم تطبيق فرق تزامن {value * 1000:.0f} ms")
    else:
        log("   ✅ التزامن سليم")
    return value


# ============================================================================
# Active timeline and adaptive chunks
# ============================================================================

def _per_second_chat(chat: Sequence[dict], n: int) -> np.ndarray:
    counts = np.zeros(n, dtype=np.float32)
    for m in chat or []:
        if not isinstance(m, dict):
            continue
        t = coerce_number(m.get("t"), None)
        if t is not None and 0 <= t < n:
            counts[int(t)] += 1.0
    return counts


def _runs(mask: np.ndarray) -> list:
    """[start, end) index runs where mask is True."""
    out = []
    if len(mask) == 0:
        return out
    padded = np.concatenate(([False], mask.astype(bool), [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def compute_active_timeline(ctx: _Ctx) -> dict:
    """Dead time = AFK spans from Stage 2 plus stretches of DEAD_MIN_S or more with no speech,
    no reaction above DEAD_REACTION_THR and no chat burst. Active = duration minus dead."""
    dur = float(ctx.duration)
    n = max(1, int(math.ceil(dur)))
    sig = ctx.signals
    alive = np.zeros(n, dtype=bool)
    if sig is not None:
        m = min(n, len(sig.words_per_s))
        alive[:m] |= sig.words_per_s[:m] > 0
        m = min(n, len(sig.R))
        alive[:m] |= sig.R[:m] > DEAD_REACTION_THR
    for w in ctx.words:
        a, b = int(max(0, math.floor(w["s"]))), int(min(n, math.ceil(w["e"])))
        alive[a:b] = True
    chat = _per_second_chat(ctx.chat, n)
    if chat.sum() > 0:
        rolling = np.convolve(chat, np.ones(10, dtype=np.float32), mode="same")
        nonzero = rolling[rolling > 0]
        thr = max(3.0, 2.0 * float(np.median(nonzero))) if len(nonzero) else 3.0
        alive |= rolling >= thr
    dead_runs = []
    for a, b in _runs(~alive):
        if b - a >= DEAD_MIN_S:
            s, e = a + DEAD_EDGE_KEEP_S, b - DEAD_EDGE_KEEP_S
            if e > s:
                dead_runs.append([float(s), float(min(e, dur))])
    afk = [list(x) for x in ctx.afk]
    dead = merge_spans(afk + dead_runs)
    active = subtract_spans(0.0, dur, dead, min_len=DEAD_EDGE_KEEP_S)
    if not active:
        active = [[0.0, round(dur, 3)]]
    return {
        "duration": round(dur, 2),
        "active_s": round(spans_total(active), 2),
        "dead_s": round(dur - spans_total(active), 2),
        "afk_s": round(spans_total(afk), 2),
        "afk": [[round(a, 2), round(b, 2)] for a, b in afk],
        "dead": [[round(a, 2), round(b, 2)] for a, b in dead],
        "active": [[round(a, 2), round(b, 2)] for a, b in active],
        "chunks": [],
        "chunk_active_s": 0,
        "made_at": now_iso(),
    }


def _wall_at_active(active: Sequence[Sequence[float]], a: float) -> float:
    """Wall clock time where the active time counter reaches a."""
    acc = 0.0
    for s, e in active:
        length = e - s
        if acc + length >= a:
            return s + (a - acc)
        acc += length
    return float(active[-1][1]) if active else 0.0


def _active_in(active: Sequence[Sequence[float]], t0: float, t1: float) -> float:
    return sum(max(0.0, min(e, t1) - max(s, t0)) for s, e in active)


def _snap_boundary(ctx: _Ctx, t: float) -> float:
    """Move a boundary to the middle of the longest speech gap within BOUNDARY_SNAP_S."""
    words = ctx.words
    if len(words) < 2:
        return t
    starts = [w["s"] for w in words]
    lo = bisect.bisect_left(starts, t - BOUNDARY_SNAP_S - 30.0)
    hi = bisect.bisect_right(starts, t + BOUNDARY_SNAP_S + 30.0)
    best, best_len = t, -1.0
    for i in range(max(1, lo), min(len(words), hi + 1)):
        g0, g1 = float(words[i - 1]["e"]), float(words[i]["s"])
        if g1 <= g0:
            continue
        mid = (g0 + g1) / 2.0
        if abs(mid - t) <= BOUNDARY_SNAP_S and (g1 - g0) > best_len:
            best, best_len = mid, g1 - g0
    return best


def _chunk_phrases(ctx: _Ctx, t0: float, t1: float) -> list:
    """Phrases inside [t0, t1] whose middle lies in active time."""
    out = []
    k = bisect.bisect_left(ctx.phrase_starts, t0 - 10.0)
    for ph in ctx.phrases[k:]:
        if ph["s"] >= t1:
            break
        mid = (float(ph["s"]) + float(ph["e"])) / 2.0
        if t0 <= mid < t1 and ctx.in_active(mid):
            out.append(ph)
    return out


def _est_tokens(text: str) -> int:
    return int(len(text) / 3.0) + 200


def _chunk_tokens(ctx: _Ctx, t0: float, t1: float) -> int:
    return _est_tokens(render_lines(ctx, _chunk_phrases(ctx, t0, t1))) + CHUNK_PROMPT_OVERHEAD


def _split_chunk(ctx: _Ctx, t0: float, t1: float, depth: int = 0) -> list:
    """Split a chunk in two at a speech gap while it is over the token cap."""
    act = _active_in(ctx.active, t0, t1)
    if depth >= 3 or act < CHUNK_ACTIVE_MIN_S or _chunk_tokens(ctx, t0, t1) <= CHUNK_TOKEN_CAP:
        return [[t0, t1]]
    inside = [[max(s, t0), min(e, t1)] for s, e in ctx.active if e > t0 and s < t1]
    mid = _snap_boundary(ctx, _wall_at_active(inside, act / 2.0))
    if not (t0 + 60.0 < mid < t1 - 60.0):
        return [[t0, t1]]
    left = _split_chunk(ctx, t0, mid, depth + 1)
    right = _split_chunk(ctx, max(t0, mid - OVERLAP_S), t1, depth + 1)
    return left + right


def build_chunks(ctx: _Ctx, chunk_active_s: float) -> list:
    """Adaptive chunks over active time: about chunk_active_s of active time each (hard cap
    CHUNK_ACTIVE_MAX_S and CHUNK_TOKEN_CAP estimated tokens), boundaries snapped to the
    longest speech gap within 90 s, 90 s overlap. Dead stretches are never counted."""
    active = ctx.active
    total = spans_total(active)
    if total <= 0:
        return []
    size_target = float(clamp(chunk_active_s, CHUNK_ACTIVE_MIN_S, CHUNK_ACTIVE_MAX_S))
    n = 1 if total <= size_target * 1.15 else int(math.ceil(total / size_target))
    size = total / n
    start, end = float(active[0][0]), float(active[-1][1])
    bounds = []
    for k in range(1, n):
        b = _snap_boundary(ctx, _wall_at_active(active, size * k))
        if bounds and b <= bounds[-1] + 60.0:
            continue
        if start + 60.0 < b < end - 60.0:
            bounds.append(b)
    edges = [start] + bounds + [end]
    raw = []
    for i in range(len(edges) - 1):
        t0 = max(start, edges[i] - (OVERLAP_S if i > 0 else 0.0))
        raw.append([t0, edges[i + 1]])
    chunks = []
    for t0, t1 in raw:
        for a, b in _split_chunk(ctx, t0, t1):
            chunks.append({"t0": round(a, 2), "t1": round(b, 2),
                           "active_s": round(_active_in(active, a, b), 1)})
    return chunks


def _normalize_chunks(raw: Any) -> list:
    """v2 runs saved chunks as [t0, t1] lists."""
    out = []
    for c in raw or []:
        if isinstance(c, dict):
            out.append({"t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
                        "active_s": float(c.get("active_s", 0.0) or 0.0)})
        elif isinstance(c, (list, tuple)) and len(c) >= 2:
            out.append({"t0": float(c[0]), "t1": float(c[1]), "active_s": float(c[1]) - float(c[0])})
    return out


# ============================================================================
# Rendering transcript lines for prompts
# ============================================================================

def _line(ctx: _Ctx, ph: dict, marker: str = "") -> str:
    pid = int(ph["id"])
    fire = " 🔥" if pid in ctx.hot_ids else ""
    eyes = " 👀" if pid in ctx.radar_ids else ""
    tags = "".join(f" [{t}]" for t in ph.get("tags") or [])
    text = ph.get("text", "")
    line = f"{marker}#{pid} [{sec_to_clock(ph['s'])}]{fire}{eyes} {text}{tags}"
    burst = ctx.chat_marks.get(pid)
    if burst:
        line += f"\n   💬 ×{burst[0]}: " + " | ".join(burst[1])
    return line


def _is_cold(ctx: _Ctx, ph: dict) -> bool:
    pid = int(ph["id"])
    if pid in ctx.guard_ids or pid in ctx.hot_ids or pid in ctx.radar_ids:
        return False
    if ph.get("kind") != "speech":
        return ph.get("kind") == "silence"
    if ph.get("tags"):
        return False
    if local_score_for_span(ctx.scores, ph["s"], ph["e"]) >= 0.35:
        return False
    return True


def render_lines(ctx: _Ctx, phrases: Sequence[dict], compact: bool = False,
                 ranges: Optional[Sequence[Sequence[int]]] = None) -> str:
    """Prompt text for a list of phrases. compact collapses runs of 6+ cold lines into one
    summary line (last resort above the token cap; never within 20 s of 🔥 or 👀). ranges
    marks the lines a candidate currently keeps with ▶. Gaps of dead time between two
    consecutive phrases are shown as one ⏭️ line."""
    keep_ids = set()
    for a, b in ranges or []:
        keep_ids.update(range(int(a), int(b) + 1))
    out: list = []
    run: list = []
    prev_end: Optional[float] = None

    def mark(p: dict) -> str:
        if not ranges:
            return ""
        return "▶ " if int(p["id"]) in keep_ids else "  "

    def flush_run() -> None:
        if len(run) >= 6:
            out.append(f"… ({len(run)} سطر كلام عادي بدون مؤشرات، من #{run[0]['id']} إلى #{run[-1]['id']}، "
                       f"{sec_to_clock(run[0]['s'])} إلى {sec_to_clock(run[-1]['e'])})")
        else:
            for p in run:
                out.append(_line(ctx, p, mark(p)))
        run.clear()

    for ph in phrases:
        if prev_end is not None and float(ph["s"]) - prev_end >= DEAD_MIN_S:
            flush_run()
            out.append(f"⏭️ (تخطينا {format_duration_ar(float(ph['s']) - prev_end)} بدون نشاط)")
        prev_end = float(ph["e"])
        if compact and _is_cold(ctx, ph):
            run.append(ph)
            continue
        flush_run()
        out.append(_line(ctx, ph, mark(ph)))
    flush_run()
    return "\n".join(out)


def _phrases_between(ctx: _Ctx, t0: float, t1: float) -> list:
    k = bisect.bisect_left(ctx.phrase_starts, t0 - 10.0)
    out = []
    for p in ctx.phrases[k:]:
        if p["s"] >= t1:
            break
        if p["e"] > t0:
            out.append(p)
    return out


def _nearest_phrase(ctx: _Ctx, t: float) -> Optional[int]:
    if not ctx.phrases:
        return None
    k = bisect.bisect_right(ctx.phrase_starts, t)
    best, best_d = None, float("inf")
    for j in (k - 2, k - 1, k, k + 1):
        if 0 <= j < len(ctx.phrases):
            ph = ctx.phrases[j]
            d = 0.0 if ph["s"] <= t <= ph["e"] else min(abs(ph["s"] - t), abs(ph["e"] - t))
            if d < best_d:
                best, best_d = int(ph["id"]), d
    return best


# ============================================================================
# Run directory management
# ============================================================================

def _transcript_tag(ctx: _Ctx) -> str:
    segs = ctx.transcript.get("segments") or []
    sample = f"{len(segs)}|{round(ctx.duration)}|{len(ctx.words)}|{segs[0].get('text', '') if segs else ''}|{segs[-1].get('text', '') if segs else ''}"
    return text_hash(sample)


def _list_runs(vp: VodPaths) -> list:
    if not os.path.isdir(vp.stage3_dir):
        return []
    names = sorted(n for n in os.listdir(vp.stage3_dir) if n.startswith("run_") and os.path.isdir(os.path.join(vp.stage3_dir, n)))
    out = []
    for name in names:
        state = load_json_safe(os.path.join(vp.stage3_dir, name, "run.json"), default=None, expected_types=(dict,), quiet=True)
        if state:
            out.append(state)
    return out


def _run_path(vp: VodPaths, run_id: str, name: str) -> str:
    return os.path.join(vp.run_dir(run_id), name)


def _save_run(vp: VodPaths, run: dict) -> None:
    run["updated_at"] = now_iso()
    atomic_write_json(_run_path(vp, run["run_id"], "run.json"), run)


def _new_run(ctx: _Ctx, kind: str, parent: Optional[str] = None) -> dict:
    run_id = new_run_id()
    ensure_dir(ctx.vp.run_dir(run_id))
    phrases = build_phrases(ctx.words, ctx.feats, vod_duration=ctx.duration)
    atomic_write_json(_run_path(ctx.vp, run_id, "phrases.json"), phrases, indent=None, backup=False)
    atomic_write_json(ctx.vp.phrases, phrases, indent=None, backup=False)
    run = {
        "run_id": run_id, "kind": kind, "parent": parent, "created_at": now_iso(),
        "transcript_tag": _transcript_tag(ctx), "status": "scout", "mode": "normal",
        "chunks": [], "critic_batch": CRITIC_BATCH_DEFAULT, "pool": 0, "selected": False,
        "version": 3,
    }
    _save_run(ctx.vp, run)
    return run


def _load_run_phrases(ctx: _Ctx, run: dict) -> list:
    phrases = load_json_safe(_run_path(ctx.vp, run["run_id"], "phrases.json"), default=None, expected_types=(list,))
    if not phrases:
        phrases = build_phrases(ctx.words, ctx.feats, vod_duration=ctx.duration)
        atomic_write_json(_run_path(ctx.vp, run["run_id"], "phrases.json"), phrases, indent=None, backup=False)
    return phrases


def _load_cands(vp: VodPaths, run_id: str) -> list:
    cands = load_json_safe(_run_path(vp, run_id, "candidates.json"), default=[], expected_types=(list,), quiet=True) or []
    return [_upgrade_cand(c, run_id) for c in cands if isinstance(c, dict) and c.get("pieces")]


def _save_cands(ctx: _Ctx, run: dict, cands: list) -> None:
    atomic_write_json(_run_path(ctx.vp, run["run_id"], "candidates.json"), cands, indent=None, backup=True)


def _upgrade_cand(c: dict, run_id: str) -> dict:
    """Fill v3 candidate keys (v2 runs and older saves load as is)."""
    c.setdefault("source", "scout")
    c.setdefault("visual_guess", "")
    c.setdefault("needs_visual", False)
    c.setdefault("confidence", 5)
    c.setdefault("media", {})
    c.setdefault("visual_event", "")
    c.setdefault("visual_beats", [])
    c.setdefault("protect", [])
    c.setdefault("hook_line", "")
    c.setdefault("miss_count", 0)
    c.setdefault("parse_fail", 0)
    c.setdefault("state", "new")
    if c.get("cid"):
        c.setdefault("media_key", f"{run_id}_{c['cid']}")
    return c


def list_run_candidates(vp: VodPaths, run_id: Optional[str] = None) -> tuple:
    """(run_id, candidates) of the given run, or of the newest run that has candidates.
    Used by the Studio candidate browser."""
    runs = _list_runs(vp)
    if run_id:
        return run_id, _load_cands(vp, run_id)
    for run in reversed(runs):
        cands = _load_cands(vp, run["run_id"])
        if cands:
            return run["run_id"], cands
    return "", []


# ============================================================================
# Request budget plan (AFK aware)
# ============================================================================

def _critic_pool(active_hours: float) -> int:
    return int(clamp(round(active_hours * POOL_PER_HOUR), POOL_MIN, POOL_MAX))


def plan_budget(ctx: _Ctx, gw: GeminiGateway, chunk_active_s: float, batch: int) -> dict:
    """Pick chunk size, Critic batch and pool so the day's quota covers the run.
    Fallback order: shrink the Critic pool, then batch 4, then longer chunks (max 55 active
    minutes). Returns {"chunks", "batch", "pool", "chunk_active_s", "mode", "budget"}."""
    pool = _critic_pool(ctx.active_hours)
    floor = min(pool, POOL_LOW_FLOOR)
    configs = [(chunk_active_s, batch, pool)]
    p = pool - 3
    while p >= floor:
        configs.append((chunk_active_s, batch, p))
        p -= 3
    if configs[-1][2] != floor:
        configs.append((chunk_active_s, batch, floor))
    if batch < CRITIC_BATCH_LOW:
        configs.append((chunk_active_s, CRITIC_BATCH_LOW, floor))
    for size in (3000.0, float(CHUNK_ACTIVE_MAX_S)):
        if size > chunk_active_s:
            configs.append((size, max(batch, CRITIC_BATCH_LOW), floor))

    cache: dict = {}
    chosen = None
    for idx, (size, b, pl) in enumerate(configs):
        if size not in cache:
            cache[size] = build_chunks(ctx, size)
        chunks = cache[size]
        needs = {"scout": len(chunks), "critic": int(math.ceil(pl / max(1, b)))}
        budget = gw.plan(needs)
        chosen = {"chunks": chunks, "batch": b, "pool": pl, "chunk_active_s": size,
                  "mode": "normal" if idx == 0 else "low_budget", "budget": budget}
        if budget.ok:
            break
    if chosen["mode"] == "low_budget":
        if chosen["budget"].ok:
            log(f"💡 الحصة قليلة اليوم، شغّلنا وضع التوفير: {len(chosen['chunks'])} أجزاء، "
                f"الناقد {chosen['pool']} مرشح بدفعات {chosen['batch']}")
        else:
            log("⚠️ حتى وضع التوفير أكبر من الحصة المتبقية اليوم. بنشتغل وإذا خلصت الحصة نوقف ونحفظ ونكمل بعد التجديد")
    return chosen


# ============================================================================
# Scout
# ============================================================================

def _seconds_to_ids(ctx: _Ctx, s: Any, e: Any) -> Optional[dict]:
    s_val, e_val = parse_time_to_sec(s), parse_time_to_sec(e)
    if s_val is None or e_val is None or e_val <= s_val:
        return None
    ids = [p["id"] for p in _phrases_between(ctx, s_val, e_val)]
    if not ids:
        return None
    return {"from_id": min(ids), "to_id": max(ids)}


def _normalize_ranges(ctx: _Ctx, raw: Any) -> list:
    out = []
    if isinstance(raw, dict):
        raw = [raw]
    for r in raw or []:
        if isinstance(r, dict):
            if "from_id" in r or "to_id" in r:
                out.append({"from_id": r.get("from_id"), "to_id": r.get("to_id", r.get("from_id"))})
            elif "start" in r or "end" in r:
                conv = _seconds_to_ids(ctx, r.get("start"), r.get("end"))
                if conv:
                    out.append(conv)
        elif isinstance(r, (list, tuple)) and len(r) >= 2:
            a, b = r[0], r[1]
            if isinstance(a, float) or isinstance(b, float) or (isinstance(a, str) and ":" in a):
                conv = _seconds_to_ids(ctx, a, b)
                if conv:
                    out.append(conv)
            else:
                out.append({"from_id": a, "to_id": b})
    return out


def _scout_normalizer(ctx: _Ctx):
    def normalize(obj: Any) -> Any:
        if isinstance(obj, list):
            obj = {"clips": obj}
        if not isinstance(obj, dict):
            return obj
        clips = obj.get("clips")
        if not isinstance(clips, list):
            clips = obj.get("candidates") if isinstance(obj.get("candidates"), list) else None
        if not isinstance(clips, list):
            return obj
        fixed = []
        for c in clips:
            if not isinstance(c, dict):
                continue
            c = dict(c)
            if "ranges" not in c:
                c["ranges"] = c.pop("segments", None) or c.pop("cuts", None) or []
            c["ranges"] = _normalize_ranges(ctx, c["ranges"])
            if "scout_score" not in c:
                c["scout_score"] = c.get("virality_score", c.get("score", 0))
            if "peak_id" not in c and c["ranges"]:
                c["peak_id"] = c["ranges"][0].get("from_id")
            if "payoff_end_id" not in c and c["ranges"]:
                c["payoff_end_id"] = c["ranges"][-1].get("to_id")
            c.setdefault("title", "لقطة")
            c.setdefault("category", "ضحك")
            fixed.append(c)
        out = dict(obj)
        out["clips"] = fixed
        topics = out.get("topics")
        out["topics"] = [t for t in topics if isinstance(t, dict)] if isinstance(topics, list) else []
        return out
    return normalize


def _validate_candidate(ctx: _Ctx, raw_ranges: Sequence[Any], peak_id: int, payoff_id: int,
                        lo_id: int, hi_id: int) -> Optional[dict]:
    """Clean id ranges: inside [lo_id, hi_id], sorted, merged; peak inside; extended to the payoff.
    Returns {"ranges": [[a, b], ...], "pieces": [[s, e], ...], "net": float} or None."""
    rng = []
    for r in raw_ranges:
        a = coerce_int(r.get("from_id") if isinstance(r, dict) else getattr(r, "from_id", None), -1)
        b = coerce_int(r.get("to_id") if isinstance(r, dict) else getattr(r, "to_id", None), -1)
        if a < 0 or b < 0:
            continue
        a, b = min(a, b), max(a, b)
        a, b = max(a, lo_id), min(b, hi_id)
        if a <= b and a in ctx.by_id and b in ctx.by_id:
            rng.append([a, b])
    if not rng:
        return None
    rng.sort()
    merged = [rng[0]]
    for a, b in rng[1:]:
        if a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    if peak_id in ctx.by_id and lo_id <= peak_id <= hi_id and not any(a <= peak_id <= b for a, b in merged):
        peak_t = ctx.by_id[peak_id]["s"]
        nearest = min(range(len(merged)), key=lambda i: min(abs(ctx.by_id[merged[i][0]]["s"] - peak_t),
                                                           abs(ctx.by_id[merged[i][1]]["e"] - peak_t)))
        a, b = merged[nearest]
        gap = min(abs(ctx.by_id[a]["s"] - peak_t), abs(ctx.by_id[b]["e"] - peak_t))
        if gap > 3.0:
            return None
        merged[nearest] = [min(a, peak_id), max(b, peak_id)]
    if payoff_id in ctx.by_id and lo_id <= payoff_id <= hi_id and payoff_id > merged[-1][1]:
        if ctx.by_id[payoff_id]["e"] - ctx.by_id[merged[-1][1]]["e"] <= 15.0:
            merged[-1][1] = payoff_id
    pieces = ids_to_pieces([{"from_id": a, "to_id": b} for a, b in merged], ctx.phrases)
    net = pieces_net(pieces)
    if net < MIN_CAND_NET or net > MAX_CAND_NET:
        return None
    return {"ranges": merged, "pieces": pieces, "net": round(net, 2)}


def _existing_ranges(manifest: dict) -> list:
    out = []
    for c in manifest.get("clips") or []:
        segs = c.get("segments") or []
        if segs:
            out.append((float(segs[0][0]), float(segs[-1][1])))
    return out


def _scout_prompt(ctx: _Ctx, idx: int, n: int, chunk: dict, body: str,
                  carry_topics: list, existing: list, radar_here: int) -> str:
    t0, t1 = chunk["t0"], chunk["t1"]
    parts = [ctx.taste_scout, ""]
    parts.append(f"📍 هذا الجزء {idx + 1} من {n} من البث ({sec_to_clock(t0)} إلى {sec_to_clock(t1)}، "
                 f"فيه {format_duration_ar(chunk.get('active_s') or (t1 - t0))} نشاط فعلي).")
    if carry_topics:
        parts.append("🧭 آخر مواضيع الجزء اللي قبله: " + " ← ".join(carry_topics))
    near = [r for r in existing if r[1] > t0 and r[0] < t1]
    if near:
        parts.append("⛔ لقطات موجودة مسبقاً لا تكررها: " + "، ".join(f"{sec_to_clock(a)} إلى {sec_to_clock(b)}" for a, b in near))
    if radar_here:
        parts.append(f"👀 الرادار لقط {radar_here} لحظة في هذا الجزء فيها ردة فعل بدون كلام يمهد لها. راجعها كلها.")
    parts.append("")
    parts.append("النص:")
    parts.append(body)
    parts.append("")
    parts.append(
        f"المطلوب: ارسم خريطة مواضيع هذا الجزء في topics، ثم اقترح كل لحظة محتملة في clips (من {CLIPS_MIN} إلى {CLIPS_MAX}). "
        "لكل لقطة: title و category و ranges و peak_id و payoff_end_id و context_note و removed (وش حذفت بالقص ولماذا) "
        "و scout_score و confidence و why و visual_guess و needs_visual."
    )
    return "\n".join(parts)


def _topic_for(topics: Sequence[dict], pid: int) -> str:
    for t in topics:
        if coerce_int(t.get("from_id"), -1) <= pid <= coerce_int(t.get("to_id"), -1):
            return str(t.get("topic", ""))
    return ""


def run_scout(ctx: _Ctx, gw: GeminiGateway, run: dict, manifest: dict) -> list:
    """Scout every pending chunk (one call each). Finished chunks are read from Drive."""
    chunks = _normalize_chunks(run.get("chunks"))
    existing = _existing_ranges(manifest)
    carry: list = []
    pending_parse = 0
    for i, chunk in enumerate(chunks):
        t0, t1 = chunk["t0"], chunk["t1"]
        path = _run_path(ctx.vp, run["run_id"], f"scout_{i:02d}.json")
        saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
        if saved and saved.get("status") in ("done", "empty"):
            carry = [t.get("topic", "") for t in (saved.get("topics") or [])][-3:]
            continue
        phrases = _chunk_phrases(ctx, t0, t1)
        if not any(p.get("kind") == "speech" for p in phrases) and not any(int(p["id"]) in ctx.radar_ids for p in phrases):
            atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "empty", "candidates": [], "topics": []})
            log(f"⏭️ الجزء {i + 1} من {len(chunks)} ({sec_to_clock(t0)} إلى {sec_to_clock(t1)}) بدون كلام")
            continue
        body = render_lines(ctx, phrases)
        if _est_tokens(body) + CHUNK_PROMPT_OVERHEAD > CHUNK_TOKEN_CAP:
            body = render_lines(ctx, phrases, compact=True)
            log(f"   🗜️ الجزء {i + 1} طويل، ضغطنا الأسطر الباردة البعيدة عن 🔥 و 👀 ({_est_tokens(body) // 1000}k توكن تقريباً)")
        radar_here = sum(1 for p in phrases if int(p["id"]) in ctx.radar_ids)
        prompt = _scout_prompt(ctx, i, len(chunks), chunk, body, carry, existing, radar_here)
        log(f"🔍 الكشاف: الجزء {i + 1} من {len(chunks)} ({sec_to_clock(t0)} إلى {sec_to_clock(t1)})")
        try:
            res = gw.generate("scout", [prompt], schema=ScoutResponse, system_instruction=SCOUT_SYSTEM_AR,
                              normalizer=_scout_normalizer(ctx), label=f"جزء {i + 1}/{len(chunks)}")
        except ParseFailure as exc:
            atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "pending_parse",
                                     "raw": (exc.raw_text or "")[:20000], "model": exc.model})
            pending_parse += 1
            log(f"   ⚠️ رد الجزء {i + 1} ما انقرأ، محفوظ وبنعيد طلبه في التشغيل الجاي")
            continue
        data: ScoutResponse = res.data
        lo_id = min(p["id"] for p in phrases)
        hi_id = max(p["id"] for p in phrases)
        cands = []
        for clip in data.clips:
            checked = _validate_candidate(ctx, clip.ranges, clip.peak_id, clip.payoff_end_id, lo_id, hi_id)
            if checked is None:
                continue
            cands.append({
                "title": clip.title.strip()[:80] or "لقطة", "category": clip.category,
                "ranges": checked["ranges"], "pieces": checked["pieces"], "net": checked["net"],
                "peak_id": clip.peak_id, "payoff_end_id": clip.payoff_end_id,
                "context_note": clip.context_note[:300], "removed": clip.removed[:6],
                "scout_score": clip.scout_score, "confidence": clip.confidence, "why": clip.why[:300],
                "visual_guess": clip.visual_guess[:300], "needs_visual": bool(clip.needs_visual),
                "chunk": i, "scout_model": res.model, "source": "scout",
            })
        topics = [{"from_id": t.from_id, "to_id": t.to_id, "topic": t.topic[:120]} for t in data.topics]
        atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "done", "model": res.model,
                                 "candidates": cands, "topics": topics, "dropped": len(data.clips) - len(cands)})
        carry = [t["topic"] for t in topics][-3:]
        visual = sum(1 for c in cands if c["needs_visual"])
        extra = f" (استبعدنا {len(data.clips) - len(cands)} غير صالح)" if len(data.clips) > len(cands) else ""
        log(f"   ✅ {len(cands)} مرشح من الجزء {i + 1}" + (f"، منها {visual} بصرية" if visual else "") + extra)
    if pending_parse:
        raise QuotaPause("scout", "busy", None,
                         f"⏸️ {pending_parse} جزء من الكشاف رده ما انقرأ. التقدم محفوظ، شغّل الخلية مرة ثانية وبنعيد هذي الأجزاء فقط.")
    all_cands = []
    for i in range(len(chunks)):
        saved = load_json_safe(_run_path(ctx.vp, run["run_id"], f"scout_{i:02d}.json"), default={}, expected_types=(dict,), quiet=True) or {}
        topics = saved.get("topics") or []
        for c in saved.get("candidates") or []:
            c = dict(c)
            c["topic"] = _topic_for(topics, c["ranges"][0][0]) if c.get("ranges") else ""
            all_cands.append(c)
    return all_cands


# ============================================================================
# Reaction Radar (local, zero requests)
# ============================================================================

def load_or_make_radar(ctx: _Ctx, run: dict) -> list:
    """Radar peaks for this run (saved once in radar.json so 👀 marks stay stable)."""
    path = _run_path(ctx.vp, run["run_id"], "radar.json")
    saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
    if saved and isinstance(saved.get("peaks"), list):
        ctx.set_radar(saved.get("ids") or [])
        return saved["peaks"]
    peaks = radar_peaks(ctx.signals, per_hour=RADAR_PER_HOUR, min_score=RADAR_MIN_SCORE,
                        min_gap_s=RADAR_MIN_GAP_S, active_spans=ctx.active)
    ids = mark_radar_phrases(peaks, ctx.phrases)
    atomic_write_json(path, {"peaks": peaks, "ids": ids, "at": now_iso()}, backup=False)
    ctx.set_radar(ids)
    if peaks:
        log(f"👀 الرادار لقط {len(peaks)} لحظة فيها ردة فعل بدون كلام يمهد لها (بدون أي طلب من Gemini)")
    return peaks


def radar_candidates(ctx: _Ctx, peaks: Sequence[dict], scout_cands: Sequence[dict], manifest: dict) -> list:
    """Radar peaks that no Scout candidate covers become direct Critic candidates:
    25 s before the peak to 8 s after, snapped to phrase ids, scout_score 6."""
    covers = [(float(c["pieces"][0][0]) - RADAR_COVER_PAD_S, float(c["pieces"][-1][1]) + RADAR_COVER_PAD_S)
              for c in scout_cands if c.get("pieces")]
    existing = _existing_ranges(manifest)
    out = []
    for pk in peaks:
        t = float(pk.get("t", 0.0))
        if any(a <= t <= b for a, b in covers) or any(a - 2.0 <= t <= b + 2.0 for a, b in existing):
            continue
        t0, t1 = max(0.0, t - RADAR_PRE_S), min(ctx.duration, t + RADAR_POST_S)
        phrases = [p for p in _phrases_between(ctx, t0, t1) if ctx.in_active((p["s"] + p["e"]) / 2.0)]
        if not phrases:
            continue
        lo_id, hi_id = min(int(p["id"]) for p in phrases), max(int(p["id"]) for p in phrases)
        peak_id = _nearest_phrase(ctx, t)
        if peak_id is None or not (lo_id <= peak_id <= hi_id):
            peak_id = hi_id
        checked = _validate_candidate(ctx, [{"from_id": lo_id, "to_id": hi_id}], peak_id, hi_id, lo_id, hi_id)
        if checked is None:
            continue
        why = str(pk.get("why") or "لحظة مشبوهة بصرياً")
        laughy = "ضحك" in why
        out.append({
            "title": "لحظة بصرية محتملة", "category": "ضحك" if laughy else "صراخ_وتوتر",
            "ranges": checked["ranges"], "pieces": checked["pieces"], "net": checked["net"],
            "peak_id": peak_id, "payoff_end_id": hi_id,
            "context_note": "", "removed": [], "scout_score": RADAR_SCOUT_SCORE, "confidence": 5,
            "why": f"الرادار: {why} (قوة {float(pk.get('score', 0.0)):.2f})",
            "visual_guess": f"غالباً صار شي على الشاشة قبل {sec_to_clock(t)} ({why})", "needs_visual": True,
            "chunk": -1, "scout_model": "radar", "source": "radar", "radar_t": round(t, 2),
            "radar_score": float(pk.get("score", 0.0)),
        })
        covers.append((t0 - RADAR_COVER_PAD_S, t1 + RADAR_COVER_PAD_S))
    return out


# ============================================================================
# Merge and pre rank
# ============================================================================

def _span(c: dict) -> tuple:
    return float(c["pieces"][0][0]), float(c["pieces"][-1][1])


def _iou(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _overlap_short(a: tuple, b: tuple) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    shorter = min(a[1] - a[0], b[1] - b[0])
    return inter / shorter if shorter > 0 else 0.0


def merge_and_rank(ctx: _Ctx, cands: list, manifest: dict, run_id: str = "") -> list:
    """Score, dedupe (IoU 0.3) and exclude spans already in the manifest. Radar candidates
    keep their place (they are already uncovered by construction)."""
    existing = _existing_ranges(manifest)
    for c in cands:
        s, e = _span(c)
        c["local"] = local_score_for_span(ctx.scores, s, e)
        prelim = compute_clip_features(c["pieces"], ctx.signals, critic=None, category=c["category"])
        c["taste_pre"] = predict_taste(ctx.profile, prelim)
        eff = 0.75 * float(c.get("scout_score", 0)) + 0.25 * float(c.get("confidence", 5))
        c["rank"] = round(0.5 * eff / 10.0 + 0.3 * c["local"] + 0.2 * c["taste_pre"], 4)
    cands.sort(key=lambda c: -c["rank"])
    kept: list = []
    for c in cands:
        sp = _span(c)
        if any(_overlap_short(sp, ex) > 0.3 for ex in existing):
            continue
        if any(_iou(sp, _span(k)) > 0.3 for k in kept):
            continue
        kept.append(c)
    for k, c in enumerate(kept):
        c["cid"] = f"c{k:03d}"
        c["state"] = "new"
        _upgrade_cand(c, run_id)
        c["media_key"] = f"{run_id}_{c['cid']}"
    return kept


def pick_pool(ranked: list, pool: int) -> list:
    """Critic pool: radar candidates first (at most half the pool), then the best Scout ones."""
    radar = [c for c in ranked if c.get("source") == "radar"][: max(1, pool // 2)]
    rest = [c for c in ranked if c.get("source") != "radar"]
    chosen = radar + rest[: max(0, pool - len(radar))]
    ids = {id(c) for c in chosen}
    return [c for c in ranked if id(c) in ids]


# ============================================================================
# Window media (Critic video + Studio preview)
# ============================================================================

def _video_window(ctx: _Ctx, c: dict) -> tuple:
    s, e = _span(c)
    w0, w1 = s - WINDOW_PRE_S, e + WINDOW_POST_S
    if w1 - w0 > WINDOW_MAX_S:
        peak = ctx.by_id.get(c.get("peak_id"), {}).get("s")
        center = float(peak) if peak is not None else (s + e) / 2.0
        w0 = max(w0, center - WINDOW_MAX_S * 0.6)
        w1 = w0 + WINDOW_MAX_S
    w0 = max(0.0, w0)
    w1 = min(ctx.duration, w1)
    return round(w0, 3), round(w1, 3)


def _file_ok(path: Any, min_bytes: int = 2048) -> bool:
    return bool(path) and os.path.isfile(str(path)) and os.path.getsize(str(path)) >= min_bytes


def prepare_window_media(ctx: _Ctx, streams: StreamSet, offset: float, run_id: str, c: dict,
                         want_critic: bool = True) -> dict:
    """Critic MP4 in /tmp and Studio preview MP4 on Drive in one FFmpeg pass. Existing files
    are reused. Never raises: returns {"status": "failed", "error"} on failure."""
    w0, w1 = _video_window(ctx, c)
    key = c.get("media_key") or f"{run_id}_{c.get('cid', 'x')}"
    old = dict(c.get("media") or {})
    critic_out = os.path.join(ctx.media_tmp(run_id), f"{sanitize_filename(key, max_len=80)}.mp4") if want_critic else None
    preview_out = ctx.vp.preview_path(key)
    same_window = abs(float(old.get("t0", -1)) - w0) < 0.01 and abs(float(old.get("t1", -1)) - w1) < 0.01
    have_preview = same_window and _file_ok(old.get("preview"))
    have_critic = (not want_critic) or (same_window and _file_ok(old.get("critic")))
    if have_preview and have_critic and old.get("status") == "ok":
        return old
    try:
        ensure_dir(ctx.vp.previews_dir)
        info = make_window_media(streams, w0, w1,
                                 critic_out=None if have_critic else critic_out,
                                 preview_out=None if have_preview else preview_out,
                                 preview_pad_s=PREVIEW_PAD_S, time_offset=offset,
                                 critic_fps=CRITIC_FILE_FPS, critic_height=CRITIC_HEIGHT,
                                 preview_height=PREVIEW_HEIGHT)
    except Exception as exc:
        return {"status": "failed", "error": short_error(exc, 160), "t0": w0, "t1": w1,
                "preview": old.get("preview") if have_preview else "",
                "p0": old.get("p0"), "p1": old.get("p1")}
    media = {
        "critic": info.get("critic") or (old.get("critic") if have_critic else ""),
        "preview": info.get("preview") or (old.get("preview") if have_preview else ""),
        "t0": info.get("t0", w0), "t1": info.get("t1", w1),
        "p0": info.get("p0") if info.get("preview") else old.get("p0", max(0.0, w0 - PREVIEW_PAD_S)),
        "p1": info.get("p1") if info.get("preview") else old.get("p1", min(ctx.duration, w1 + PREVIEW_PAD_S)),
        "seconds": info.get("seconds") if info.get("critic") else old.get("seconds", round(w1 - w0, 3)),
        "source": info.get("source", ""), "status": "ok", "at": now_iso(),
    }
    return media


# ============================================================================
# Critic
# ============================================================================

def _critic_text_window(ctx: _Ctx, c: dict) -> tuple:
    s, e = _span(c)
    w0, w1 = _video_window(ctx, c)
    phrases = _phrases_between(ctx, min(s - CRITIC_TEXT_PRE_S, w0), max(e + CRITIC_TEXT_POST_S, w1))
    phrases = [p for p in phrases if ctx.in_active((p["s"] + p["e"]) / 2.0)] or phrases
    if not phrases:
        return [], -1, -1
    return phrases, min(int(p["id"]) for p in phrases), max(int(p["id"]) for p in phrases)


def _audio_range(c: dict, duration: float) -> tuple:
    s, e = _span(c)
    a0 = max(0.0, s - AUDIO_PRE_S)
    a1 = min(duration, e + AUDIO_POST_S)
    if a1 - a0 > AUDIO_MAX_S:
        a0 = max(0.0, s - 5.0)
        a1 = min(duration, a0 + AUDIO_MAX_S)
    return a0, a1


def _batch_golden_text(ctx: _Ctx, batch: list) -> str:
    cats = Counter(c.get("category", "ضحك") for c in batch)
    category = cats.most_common(1)[0][0] if cats else None
    visual = any(c.get("needs_visual") for c in batch)
    moment = "فشل_بصري" if visual else None
    return golden_context(ctx.golden, "critic", category=category, moment_type=moment)


def _critic_contents(ctx: _Ctx, batch: list, use_video: bool) -> tuple:
    """Prompt parts for one Critic request. Returns (contents, windows) where windows maps
    the candidate number to {"lo", "hi", "m0", "m1", "kind"} (m0 = transcript time where the
    attached video or audio starts)."""
    taste = prompt_context(ctx.profile, ctx.rules, ctx.feedback, golden_text=_batch_golden_text(ctx, batch))
    contents: list = [taste, "",
                      f"قدامك {len(batch)} مرشحين. لكل مرشح: معلوماته، النافذة المسموحة بأرقام الأسطر، "
                      "النص (▶ يعني السطر داخل قص الكشاف الحالي)، ثم فيديو اللقطة بصوتها."]
    windows: dict = {}
    for n, c in enumerate(batch, start=1):
        phrases, lo, hi = _critic_text_window(ctx, c)
        media = c.get("media") or {}
        source_txt = "الرادار الصوتي (ردة فعل بدون كلام قبلها)" if c.get("source") == "radar" else (
            "ترقية من الممنتج" if c.get("source") == "promoted" else "الكشاف")
        header = (
            f"\n━━━━ مرشح {n} ━━━━\n"
            f"المصدر: {source_txt} | العنوان المقترح: {c['title']} | التصنيف: {c['category']} | "
            f"تقييم الكشاف: {c.get('scout_score', 0)}/10 (ثقة {c.get('confidence', 5)}/10)\n"
            f"السبب: {c.get('why', '')}\n"
            f"السياق: {c.get('context_note', '')} {('| الموضوع: ' + c['topic']) if c.get('topic') else ''}\n"
            + (f"توقع الكشاف لللي على الشاشة: {c['visual_guess']}\n" if c.get("visual_guess") else "")
            + f"المحذوف بالقص: {'، '.join(c.get('removed') or []) or 'لا شي'}\n"
            f"النافذة المسموحة: من #{lo} إلى #{hi} | المدة الحالية بعد القص {float(c.get('net', 0)):.0f} ثانية\n"
            f"النص:\n{render_lines(ctx, phrases, ranges=c['ranges'])}\n"
        )
        attached = False
        if use_video and media.get("status") == "ok" and _file_ok(media.get("critic")):
            try:
                with open(media["critic"], "rb") as fh:
                    data = fh.read()
                m0, m1 = float(media["t0"]), float(media["t1"])
                contents.append(header + f"🎬 فيديو المرشح {n} يبدأ عند {sec_to_clock(m0)} من البث وطوله "
                                         f"{format_duration_ar(m1 - m0)} (أوقات visual_beats تنحسب من بدايته):")
                contents.append(video_item(data, seconds=float(media.get("seconds") or (m1 - m0)), fps=CRITIC_SEND_FPS))
                windows[n] = {"lo": lo, "hi": hi, "m0": m0, "m1": m1, "kind": "video"}
                attached = True
            except OSError as exc:
                log(f"   ⚠️ تعذر قراءة فيديو المرشح {c['cid']} ({short_error(exc, 80)})، نرسل الصوت")
        if not attached:
            a0, a1 = _audio_range(c, ctx.duration)
            note = "(الفيديو غير متوفر) " if use_video else ""
            contents.append(header + f"🎧 {note}الصوت التالي من {sec_to_clock(a0)} إلى {sec_to_clock(a1)} "
                                     "(أوقات visual_beats تنحسب من بدايته):")
            try:
                contents.append(audio_snippet(ctx.local_audio(), a0, a1))
            except Exception as exc:
                contents.append(f"(تعذر إرفاق الصوت: {short_error(exc, 80)})")
            windows[n] = {"lo": lo, "hi": hi, "m0": a0, "m1": a1, "kind": "audio"}
            media = dict(media)
            media["sent_as"] = "audio"
            if media.get("status") not in ("ok", "failed"):
                media["status"] = "audio_only"
            c["media"] = media
    contents.append(
        "\nالمطلوب: verdicts فيها حكم لكل مرشح برقم candidate (1 إلى "
        f"{len(batch)}) مع keep و reject و ranges و end_anchor_id و rubric (فيه visual) و voice_check و long_ok "
        "و title و category و cut_notes و visual_event و visual_beats و hook_line."
    )
    return contents, windows


def _critic_normalizer(obj: Any) -> Any:
    if isinstance(obj, list):
        obj = {"verdicts": obj}
    if isinstance(obj, dict) and "verdicts" not in obj:
        for key in ("results", "candidates", "items"):
            if isinstance(obj.get(key), list):
                obj = dict(obj)
                obj["verdicts"] = obj.pop(key)
                break
    if isinstance(obj, dict) and isinstance(obj.get("verdicts"), list):
        fixed = []
        for v in obj["verdicts"]:
            if not isinstance(v, dict):
                continue
            v = dict(v)
            if "candidate" not in v:
                v["candidate"] = v.get("id", v.get("index", v.get("n", -1)))
            if "ranges" in v and not isinstance(v["ranges"], list):
                v["ranges"] = [v["ranges"]] if isinstance(v["ranges"], dict) else []
            fixed.append(v)
        obj = dict(obj)
        obj["verdicts"] = fixed
    return obj


def run_critic(ctx: _Ctx, gw: GeminiGateway, run: dict, cands: list, audio_path: Optional[str] = None,
               batch_size: int = CRITIC_BATCH_DEFAULT, critic_video: bool = True) -> None:
    """Critic every candidate in state "new". Window media for the next batch is prepared in
    a background thread while Gemini answers the current one. Skipped candidates return to
    "new" once; a second miss rejects them. Each batch is saved on Drive.
    audio_path (kept from v2) is an optional local copy of audio.mp3 for audio fallbacks."""
    if audio_path and os.path.isfile(audio_path):
        ctx._local_audio = audio_path
    batch_size = int(clamp(batch_size, 1, 6))
    streams = ctx.get_streams() if critic_video else None
    offset = ctx.get_offset() if streams is not None else 0.0
    use_video = streams is not None
    if critic_video and not use_video:
        log("⚠️ الناقد بيشتغل بالصوت فقط لهذا التشغيل لأن روابط الفيديو ما اشتغلت")
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tc_media") if use_video else None
    futures: dict = {}
    pending_parse = 0

    def submit(batch: list) -> None:
        if executor is None:
            return
        for c in batch:
            if c["cid"] not in futures:
                futures[c["cid"]] = executor.submit(prepare_window_media, ctx, streams, offset, run["run_id"], c, True)

    try:
        for pass_no in range(MAX_CRITIC_PASSES):
            todo = [c for c in cands if c.get("state") == "new" and int(c.get("parse_fail", 0)) < MAX_PARSE_FAILS]
            if not todo:
                break
            batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
            if pass_no > 0:
                log(f"🔁 الناقد: نعيد {len(todo)} مرشح ما وصلنا حكمهم في المرة الأولى")
            for bi, batch in enumerate(batches):
                tag = text_hash("|".join(f"{c['cid']}:{c.get('miss_count', 0)}" for c in batch), 8)
                path = _run_path(ctx.vp, run["run_id"], f"critic_{tag}.json")
                saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
                if saved and saved.get("status") == "done":
                    _apply_verdicts(ctx, batch, saved.get("verdicts") or [], saved.get("windows") or {}, saved.get("model", ""))
                    _save_cands(ctx, run, cands)
                    continue
                submit(batch)
                if bi + 1 < len(batches):
                    submit(batches[bi + 1])
                if executor is not None:
                    for c in batch:
                        fut = futures.get(c["cid"])
                        try:
                            c["media"] = fut.result(timeout=MEDIA_WAIT_S) if fut else (c.get("media") or {})
                        except Exception as exc:
                            c["media"] = {"status": "failed", "error": short_error(exc, 120)}
                    _save_cands(ctx, run, cands)
                n_video = sum(1 for c in batch if (c.get("media") or {}).get("status") == "ok")
                log(f"🎬 الناقد: الدفعة {bi + 1} من {len(batches)} ({len(batch)} مرشح"
                    + (f"، {n_video} بفيديو" if use_video else "، صوت فقط") + ")")
                contents, windows = _critic_contents(ctx, batch, use_video)
                try:
                    res = gw.generate("critic", contents, schema=CriticResponse, system_instruction=CRITIC_SYSTEM_AR,
                                      normalizer=_critic_normalizer, label=f"دفعة {bi + 1}/{len(batches)}")
                except ParseFailure as exc:
                    atomic_write_json(path, {"status": "pending_parse", "raw": (exc.raw_text or "")[:20000],
                                             "cids": [c["cid"] for c in batch]})
                    for c in batch:
                        c["parse_fail"] = int(c.get("parse_fail", 0)) + 1
                        if c["parse_fail"] >= MAX_PARSE_FAILS:
                            c["state"] = "rejected"
                            c["reject"] = ["بدون_حكم"]
                    _save_cands(ctx, run, cands)
                    pending_parse += 1
                    log("   ⚠️ رد الناقد ما انقرأ لهذي الدفعة، نكمل الباقي ونعيدها بالتشغيل الجاي")
                    continue
                verdicts = [v.model_dump() for v in res.data.verdicts]
                windows_json = {str(k): v for k, v in windows.items()}
                atomic_write_json(path, {"status": "done", "model": res.model, "cids": [c["cid"] for c in batch],
                                         "verdicts": verdicts, "windows": windows_json})
                _apply_verdicts(ctx, batch, verdicts, windows_json, res.model)
                _save_cands(ctx, run, cands)
                _cleanup_critic_files(batch)
                kept = sum(1 for c in batch if c.get("state") == "kept")
                missed = sum(1 for c in batch if c.get("state") == "new")
                rejected = len(batch) - kept - missed
                log(f"   ✅ الناقد قبل {kept} ورفض {rejected}" + (f"، و {missed} ما وصلنا حكمهم (نعيدهم)" if missed else ""))
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if pending_parse and any(c.get("state") == "new" for c in cands):
        raise QuotaPause("critic", "busy", None,
                         f"⏸️ {pending_parse} دفعة من الناقد ردها ما انقرأ. التقدم محفوظ، شغّل الخلية مرة ثانية وبنعيد هذي الدفعات فقط.")


def _cleanup_critic_files(batch: list) -> None:
    for c in batch:
        if c.get("state") == "new":
            continue
        path = (c.get("media") or {}).get("critic")
        if path and os.path.isfile(path) and str(path).startswith(TMP_ROOT):
            try:
                os.remove(path)
            except OSError:
                pass


def _beats_to_vod(beats: Sequence[dict], m0: float, m1: float) -> list:
    """Visual beats in MM:SS relative to the attached media -> [[s, e, what], ...] in VOD
    seconds, padded by BEAT_PAD_S and clamped to the media window."""
    out = []
    for b in beats or []:
        if not isinstance(b, dict):
            continue
        rs, re_ = mmss_to_sec(b.get("start")), mmss_to_sec(b.get("end"))
        if rs is None:
            continue
        if re_ is None or re_ <= rs:
            re_ = rs + 1.5
        s = clamp(m0 + rs - BEAT_PAD_S, m0, m1)
        e = clamp(m0 + re_ + BEAT_PAD_S, m0, m1)
        if e - s >= 0.3:
            out.append([round(s, 3), round(e, 3), str(b.get("what") or "")[:160]])
    return out


def _extend_for_beats(ctx: _Ctx, c: dict, lo: int, hi: int) -> None:
    """Make sure the id ranges reach the visual beats: a beat up to 20 s before the first
    piece pulls the start back, a beat up to 8 s after the last piece pushes the end."""
    beats = c.get("visual_beats") or []
    if not beats or not c.get("ranges"):
        return
    ranges = [list(r) for r in c["ranges"]]
    first_s = ctx.by_id[ranges[0][0]]["s"]
    last_e = ctx.by_id[ranges[-1][1]]["e"]
    changed = False
    for s, e, _what in beats:
        if s < first_s and first_s - s <= BEAT_EXTEND_BEFORE_S:
            ids = [int(p["id"]) for p in _phrases_between(ctx, s, first_s) if lo <= int(p["id"]) <= hi]
            if ids and min(ids) < ranges[0][0]:
                ranges[0][0] = min(ids)
                first_s = ctx.by_id[ranges[0][0]]["s"]
                changed = True
        if e > last_e and e - last_e <= BEAT_EXTEND_AFTER_S:
            ids = [int(p["id"]) for p in _phrases_between(ctx, last_e, e) if lo <= int(p["id"]) <= hi]
            if ids and max(ids) > ranges[-1][1]:
                ranges[-1][1] = max(ids)
                last_e = ctx.by_id[ranges[-1][1]]["e"]
                changed = True
    if changed:
        c["ranges"] = ranges
        c["pieces"] = ids_to_pieces([{"from_id": a, "to_id": b} for a, b in ranges], ctx.phrases)
        c["net"] = round(pieces_net(c["pieces"]), 2)


def _apply_verdicts(ctx: _Ctx, batch: list, verdicts: list, windows: dict, model: str) -> None:
    by_num: dict = {}
    for v in verdicts:
        num = coerce_int(v.get("candidate"), -1)
        if 1 <= num <= len(batch) and num not in by_num:
            by_num[num] = v
    for n, c in enumerate(batch, start=1):
        v = by_num.get(n)
        if v is None:
            if int(c.get("miss_count", 0)) < 1:
                c["miss_count"] = int(c.get("miss_count", 0)) + 1
                c["state"] = "new"
            else:
                c["state"] = "rejected"
                c["reject"] = ["بدون_حكم"]
            continue
        win = windows.get(str(n), windows.get(n)) or {}
        if isinstance(win, (list, tuple)):
            win = {"lo": win[0], "hi": win[1]}
        lo, hi = coerce_int(win.get("lo"), -1), coerce_int(win.get("hi"), -1)
        rubric = v.get("rubric") or {}
        c["critic"] = {k: coerce_int(rubric.get(k), 5, lo=0, hi=10) for k in RUBRIC_KEYS}
        c["voice_check"] = v.get("voice_check", "her")
        c["long_ok"] = bool(v.get("long_ok"))
        c["cut_notes"] = str(v.get("cut_notes") or "")[:300]
        c["critic_model"] = model
        c["reject"] = [r for r in v.get("reject") or [] if r]
        c["visual_event"] = str(v.get("visual_event") or "")[:300]
        c["hook_line"] = str(v.get("hook_line") or "")[:120]
        m0 = coerce_number(win.get("m0"), None)
        m1 = coerce_number(win.get("m1"), None)
        beats = _beats_to_vod(v.get("visual_beats") or [], float(m0), float(m1)) if m0 is not None and m1 is not None else []
        c["visual_beats"] = beats
        c["protect"] = merge_spans([[b[0], b[1]] for b in beats])
        if v.get("title"):
            c["title"] = str(v["title"]).strip()[:80]
        if v.get("category"):
            c["category"] = _norm_category(v["category"])
        keep = bool(v.get("keep"))
        if c["voice_check"] == "game" and c["category"] == "صراخ_وتوتر":
            keep = False
            c["reject"] = sorted(set(c["reject"]) | {"صراخ_لعبة_فقط"})
        new_ranges = v.get("ranges") or []
        anchor = v.get("end_anchor_id")
        if keep and lo >= 0:
            if new_ranges:
                checked = _validate_candidate(ctx, new_ranges, c["peak_id"], anchor if anchor is not None else -1, lo, hi)
                if checked is not None:
                    c["ranges"], c["pieces"], c["net"] = checked["ranges"], checked["pieces"], checked["net"]
            elif anchor is not None and lo <= anchor <= hi and anchor > c["ranges"][-1][1] and anchor in ctx.by_id:
                c["ranges"][-1][1] = anchor
                c["pieces"] = ids_to_pieces([{"from_id": a, "to_id": b} for a, b in c["ranges"]], ctx.phrases)
                c["net"] = round(pieces_net(c["pieces"]), 2)
            _extend_for_beats(ctx, c, lo, hi)
        c["state"] = "kept" if keep else "rejected"


# ============================================================================
# Final selection
# ============================================================================

def _excerpt(words: Sequence[dict], pieces: Sequence[Sequence[float]], limit: int = 600) -> str:
    out = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2.0
        if any(s <= mid < e for s, e in pieces):
            out.append(w["w"])
    text = " ".join(out)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def select_finals(ctx: _Ctx, cands: list, max_clips: int, auto_shave: bool,
                  allow_unreviewed: bool) -> list:
    prefs = doctor_prefs(ctx.profile)
    pool = [c for c in cands if c.get("state") == "kept"]
    unreviewed = []
    if allow_unreviewed:
        unreviewed = [c for c in cands if c.get("state") == "new" and c.get("scout_score", 0) >= 8]
    finals = []
    for c in pool + unreviewed:
        peak = ctx.by_id.get(c.get("peak_id"), {}).get("s")
        protect = [[float(a), float(b)] for a, b in (c.get("protect") or [])] or None
        pieces, rep = cut_doctor(c["pieces"], ctx.words, ctx.feats, ctx.duration, fps=ctx.fps,
                                 peak_time=peak, prefs=prefs, auto_shave=auto_shave, protect=protect)
        if not pieces:
            continue
        net = pieces_net(pieces)
        if net < MIN_FINAL_NET:
            log(f"   ↪️ استبعدنا «{c['title']}» لأنها قصيرة جداً بعد التنظيف ({net:.1f} ثانية)")
            continue
        if net > HARD_MAX_NET and not c.get("long_ok"):
            log(f"   ↪️ استبعدنا «{c['title']}» لأنها أطول من {HARD_MAX_NET:.0f} ثانية ({net:.0f} ثانية)")
            continue
        critic = c.get("critic")
        feats = compute_clip_features(pieces, ctx.signals, critic=critic, category=c["category"])
        taste = predict_taste(ctx.profile, feats)
        if critic:
            critic_mean = sum(critic.get(k, 5) for k in _TEXT_RUBRIC_KEYS) / len(_TEXT_RUBRIC_KEYS)
            visual = float(critic.get("visual", 5))
        else:
            critic_mean, visual = 5.0, 5.0
        visual_bonus = visual / 10.0 if (c.get("visual_event") or c.get("needs_visual")) else 0.5
        final = (0.40 * critic_mean / 10.0 + 0.10 * visual_bonus + 0.20 * float(c.get("scout_score", 0)) / 10.0
                 + 0.15 * float(c.get("local", 0.0)) + 0.15 * taste)
        finals.append({
            "cand": c, "pieces": pieces, "doctor": rep, "features": feats, "taste": taste,
            "final": round(final, 4), "reviewed": c.get("state") == "kept",
        })
    finals.sort(key=lambda f: -f["final"])
    chosen: list = []
    for f in finals:
        sp = (f["pieces"][0][0], f["pieces"][-1][1])
        if any(_iou(sp, (g["pieces"][0][0], g["pieces"][-1][1])) > 0.3 for g in chosen):
            continue
        chosen.append(f)
        if len(chosen) >= max_clips:
            break
    chosen.sort(key=lambda f: f["pieces"][0][0])
    return chosen


def build_plan(ctx: _Ctx, run: dict, manifest: dict, chosen: list) -> list:
    """Reserve clip ids in the manifest (saved immediately, so a crash can never hand the
    same number out twice) and write plan.json."""
    plan = []
    for f in chosen:
        c = f["cand"]
        clip_id = allocate_clip_id(manifest)
        title = c["title"]
        base = f"clip_{clip_id:02d}_{sanitize_filename(title)}"
        media = c.get("media") or {}
        plan.append({
            "id": clip_id, "uid": make_clip_uid(ctx.vp.vod_id, run["run_id"], clip_id),
            "cid": c["cid"], "title": title, "category": c["category"],
            "llm_segments": c["pieces"], "segments": f["pieces"], "doctor": f["doctor"],
            "features": f["features"], "taste": f["taste"], "final": f["final"],
            "scout": c.get("scout_score"), "local": c.get("local"), "critic": c.get("critic"),
            "reviewed": f["reviewed"], "reason": c.get("why", ""), "context_note": c.get("context_note", ""),
            "cut_notes": c.get("cut_notes", ""), "voice_check": c.get("voice_check", ""),
            "models": {"scout": c.get("scout_model", ""), "critic": c.get("critic_model", "")},
            "source": c.get("source", "scout"), "visual_event": c.get("visual_event", ""),
            "visual_beats": c.get("visual_beats") or [], "protect": c.get("protect") or [],
            "hook_line": c.get("hook_line", ""), "media_key": c.get("media_key", ""),
            "preview_path": media.get("preview") if _file_ok(media.get("preview")) else "",
            "preview_window": [media.get("p0"), media.get("p1")] if _file_ok(media.get("preview")) else [],
            "video_path": os.path.join(ctx.vp.clips_dir, base + ".mp4"),
            "srt_path": os.path.join(ctx.vp.clips_dir, base + ".srt"),
            "notes_path": os.path.join(ctx.vp.clips_dir, base + "_notes.txt"),
            "status": "planned", "attempts": 0, "error": "",
        })
        c["state"] = "selected"
    save_manifest(ctx.vp, manifest)
    atomic_write_json(_run_path(ctx.vp, run["run_id"], "plan.json"), plan)
    return plan


# ============================================================================
# CapCut notes
# ============================================================================

def _twitch_time_link(vod_url: str, seconds: float) -> str:
    total = int(max(0.0, float(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    sep = "&" if "?" in vod_url else "?"
    return f"{vod_url}{sep}t={h}h{m:02d}m{s:02d}s"


def write_clip_notes(clip: dict, path: str) -> str:
    """CapCut handoff notes (UTF 8 text next to the clip): title, category, hook line, VOD
    time with a Twitch link, visual event, reason, context and cut notes. Atomic write."""
    segs = clip.get("segments") or []
    start = float(segs[0][0]) if segs else float(coerce_number(clip.get("start"), 0.0) or 0.0)
    end = float(segs[-1][1]) if segs else float(coerce_number(clip.get("end"), start) or start)
    net = float(coerce_number(clip.get("duration"), None) or sum(e - s for s, e in segs) or 0.0)
    vod_url = str(clip.get("vod_url") or "")
    clip_id = clip.get("id")
    lines = [
        f"🎬 لقطة #{int(clip_id):02d}: {clip.get('title', 'لقطة')}" if isinstance(clip_id, int) else f"🎬 {clip.get('title', 'لقطة')}",
        f"📂 التصنيف: {clip.get('category', '')}",
    ]
    if clip.get("hook_line"):
        lines.append(f"🪝 جملة الهوك للشاشة: {clip['hook_line']}")
    lines.append(f"⏱️ وقتها في البث: {sec_to_hms(start)[:8]} إلى {sec_to_hms(end)[:8]} (المدة بعد القص {net:.1f} ثانية)")
    if vod_url:
        lines.append(f"🔗 رابط اللحظة: {_twitch_time_link(vod_url, start)}")
    if clip.get("visual_event"):
        lines.append(f"👀 وش صار على الشاشة: {clip['visual_event']}")
    beats = clip.get("visual_beats") or []
    if beats:
        rel = []
        cursor = 0.0
        for s, e in segs:
            for b in beats:
                bs = float(b[0])
                if s <= bs < e:
                    rel.append(f"{sec_to_mmss(cursor + bs - s)} {b[2] if len(b) > 2 else ''}".strip())
            cursor += e - s
        if rel:
            lines.append("📍 لحظات الشاشة داخل اللقطة: " + " | ".join(rel))
    if clip.get("reason"):
        lines.append(f"💡 ليش اخترناها: {clip['reason']}")
    if clip.get("context_note"):
        lines.append(f"🧭 السياق: {clip['context_note']}")
    if clip.get("cut_notes"):
        lines.append(f"✂️ ملاحظات القص: {clip['cut_notes']}")
    if segs:
        lines.append(f"🧩 القطع ({len(segs)}): " + "، ".join(f"{sec_to_hms(s)[:8]} إلى {sec_to_hms(e)[:8]}" for s, e in segs))
    lines.append("")
    lines.append("الفيديو والسبتايتل بدون أي كتابة أو موسيقى محروقة. الهوك والمؤثرات تنضاف في CapCut.")
    atomic_write_text(path, "\n".join(lines) + "\n", encoding="utf-8")
    return path


# ============================================================================
# Rendering and drafts
# ============================================================================

def _find_by_uid(manifest: dict, uid: str) -> tuple:
    for i, c in enumerate(manifest.get("clips") or []):
        if c.get("uid") == uid:
            return i, c
    return None, None


def _ensure_preview(ctx: _Ctx, p: dict, run_id: str) -> None:
    """Draft clips need a preview window for the Studio editor. Reuse the Critic preview,
    otherwise cut one (preview only)."""
    if _file_ok(p.get("preview_path")) and p.get("preview_window"):
        return
    streams = ctx.get_streams()
    if streams is None:
        return
    fake = {"pieces": p["segments"], "peak_id": None, "cid": p["cid"],
            "media_key": p.get("media_key") or f"{run_id}_{p['cid']}"}
    media = prepare_window_media(ctx, streams, ctx.get_offset(), run_id, fake, want_critic=False)
    if media.get("status") == "ok" and _file_ok(media.get("preview")):
        p["preview_path"] = media["preview"]
        p["preview_window"] = [media.get("p0"), media.get("p1")]


def _clip_record(ctx: _Ctx, run: dict, p: dict, pieces: list, srt: dict, status: str,
                 render_info: Optional[dict]) -> dict:
    net = pieces_net(pieces)
    span = pieces[-1][1] - pieces[0][0]
    return normalize_clip({
        "uid": p["uid"], "id": p["id"], "run_id": run["run_id"], "title": p["title"], "category": p["category"],
        "segments": pieces, "llm_segments": p["llm_segments"],
        "duration": round(net, 2), "removed_sec": round(span - net, 2),
        "shaved_sec": (p.get("doctor") or {}).get("shaved_sec", 0.0),
        "end_extended_sec": (p.get("doctor") or {}).get("end_extended_sec", 0.0),
        "scores": {"scout": p.get("scout"), "critic": p.get("critic"), "local": p.get("local"),
                   "taste": p.get("taste"), "final": p.get("final")},
        "reviewed": p.get("reviewed"), "models": p.get("models", {}),
        "features": p.get("features", {}), "reason": p.get("reason", ""),
        "context_note": p.get("context_note", ""), "cut_notes": p.get("cut_notes", ""),
        "voice_check": p.get("voice_check", ""), "excerpt": _excerpt(ctx.words, pieces),
        "video_path": p["video_path"], "srt_path": p["srt_path"], "notes_path": p.get("notes_path", ""),
        "srt_stats": {k: srt.get(k) for k in ("cues", "min_cue_s", "avg_cps", "duration")},
        "render": {k: (render_info or {}).get(k) for k in ("encoder", "size_mb", "height", "fps", "refreshed", "peak_mb")}
        if render_info else {"fps": ctx.fps},
        "status": status, "created_at": now_iso(),
        "source": p.get("source", "scout"), "visual_event": p.get("visual_event", ""),
        "visual_beats": p.get("visual_beats") or [], "protect": p.get("protect") or [],
        "hook_line": p.get("hook_line", ""), "preview_path": p.get("preview_path") or "",
        "preview_window": p.get("preview_window") or [], "media_key": p.get("media_key", ""),
        "cid": p.get("cid", ""), "vod_url": ctx.vod_url,
        "doctor": {k: (p.get("doctor") or {}).get(k) for k in ("protected", "protect_extended_sec", "shaved_sec")},
    }, ctx.vp.vod_id)


def render_plan(ctx: _Ctx, run: dict, manifest: dict, plan: list, srt_bom: bool, max_height: int,
                render_mode: str = "on_approve") -> int:
    """Turn planned clips into manifest entries. "all": full quality MP4 + SRT + notes
    (status ready). "on_approve": draft with preview, initial SRT and notes (status draft)."""
    pending = [p for p in plan if p["status"] in ("planned", "failed") and p.get("attempts", 0) < MAX_CUT_ATTEMPTS]
    if not pending:
        return 0
    ensure_dir(ctx.vp.clips_dir)
    plan_path = _run_path(ctx.vp, run["run_id"], "plan.json")
    streams = None
    offset = 0.0
    if render_mode == "all":
        streams = ctx.get_streams(max_height)
        if streams is None:
            raise RuntimeError(f"تعذر استخراج رابط البث للقص: {ctx.streams_error}")
        offset = ctx.get_offset()
    done = 0
    for n, p in enumerate(pending, start=1):
        p["attempts"] = int(p.get("attempts", 0)) + 1
        t0 = time.time()
        try:
            if render_mode == "all":
                pre = snap_to_frames(p["segments"], streams.master.fps)
                log(f"✂️ [{n}/{len(pending)}] لقطة #{p['id']:02d} «{p['title']}» ({len(pre)} قطعة، {pieces_net(pre):.1f} ثانية)")
                info = render_pieces(streams, p["segments"], p["video_path"], p["srt_path"], ctx.words,
                                     time_offset=offset, max_height=max_height, srt_bom=srt_bom)
                pieces = info["segments"]
                srt = info["srt"]
                _ensure_preview(ctx, p, run["run_id"])
                clip = _clip_record(ctx, run, p, pieces, srt, "ready", info)
            else:
                _ensure_preview(ctx, p, run["run_id"])
                pieces = snap_to_frames(p["segments"], ctx.fps) if ctx.fps else [list(x) for x in p["segments"]]
                srt = generate_srt(ctx.words, pieces, p["srt_path"], fps=ctx.fps, bom=srt_bom)
                clip = _clip_record(ctx, run, p, pieces, srt, "draft", None)
                info = {}
            write_clip_notes(clip, p["notes_path"])
        except Exception as exc:
            p["status"] = "failed"
            p["error"] = short_error(exc, 300)
            atomic_write_json(plan_path, plan)
            log(f"   ❌ فشل تجهيز اللقطة #{p['id']:02d}: {p['error']}")
            continue
        if srt.get("errors"):
            log(f"   ⚠️ ملاحظات على السبتايتل: {'، '.join(srt['errors'][:3])}")
        _, existing = _find_by_uid(manifest, p["uid"])
        if existing is not None:
            manifest["clips"] = [c for c in manifest["clips"] if c.get("uid") != p["uid"]]
        manifest["clips"].append(clip)
        save_manifest(ctx.vp, manifest)
        p["status"] = "cut" if render_mode == "all" else "drafted"
        p["error"] = ""
        atomic_write_json(plan_path, plan)
        done += 1
        badge = "" if p.get("reviewed") else " ⚠️ بدون ناقد"
        eye = " 👀" if p.get("visual_event") else ""
        if render_mode == "all":
            log(f"   ✅ جاهزة خلال {time.time() - t0:.0f} ثانية ({info.get('size_mb', 0.0):.1f} MB، {srt.get('cues', 0)} سطر سبتايتل){badge}{eye}")
        else:
            prev = "مع معاينة" if clip.get("preview_path") else "بدون معاينة"
            log(f"   📝 مسودة #{p['id']:02d} «{p['title']}» {clip['duration']:.1f}ث ({prev}){badge}{eye}")
    return done


# ============================================================================
# REDO: archive liked clips, delete the rest, keep ids moving forward
# ============================================================================

def archive_and_reset(ctx: _Ctx, manifest: dict) -> dict:
    """Liked clips are moved to clips_archive/<stamp>/ (drafts are recorded without files),
    everything else is deleted through safe_delete. Ratings live in memory/feedback.json and
    are never touched; next_id never moves back, so clip ids are never reused."""
    clips = manifest.get("clips") or []
    if not clips:
        return manifest
    stamp = new_run_id()
    archive_dir = os.path.join(ctx.vp.clips_archive_dir, stamp)
    archived, deleted = [], 0
    keep_previews = {os.path.abspath(a["preview_path"]) for a in manifest.get("archived_clips") or []
                     if isinstance(a, dict) and a.get("preview_path")}
    for c in clips:
        paths = [c.get("video_path"), c.get("srt_path"), c.get("notes_path")]
        if c.get("rating") == "liked":
            ensure_dir(archive_dir)
            moved = {}
            for path in paths:
                if path and os.path.isfile(path):
                    dst = os.path.join(archive_dir, os.path.basename(path))
                    shutil.move(path, dst)
                    moved[path] = dst
            if c.get("preview_path"):
                keep_previews.add(os.path.abspath(c["preview_path"]))
            archived.append({"uid": c.get("uid"), "id": c.get("id"), "title": c.get("title"),
                             "rating": c.get("rating"), "status": c.get("status"),
                             "segments": c.get("segments"), "preview_path": c.get("preview_path", ""),
                             "video_path": moved.get(c.get("video_path"), ""),
                             "srt_path": moved.get(c.get("srt_path"), ""),
                             "notes_path": moved.get(c.get("notes_path"), ""),
                             "archived_at": now_iso()})
        else:
            for path in paths:
                if path and os.path.isfile(path):
                    try:
                        safe_delete(path, ctx.drive_root)
                        deleted += 1
                    except PermissionError as exc:
                        log(f"   ⚠️ {exc}")
    if os.path.isdir(ctx.vp.previews_dir):
        for name in os.listdir(ctx.vp.previews_dir):
            path = os.path.join(ctx.vp.previews_dir, name)
            if os.path.isfile(path) and os.path.abspath(path) not in keep_previews:
                try:
                    safe_delete(path, ctx.drive_root)
                except PermissionError as exc:
                    log(f"   ⚠️ {exc}")
    history = list(manifest.get("archived_clips") or []) + archived
    new_manifest = {"schema": manifest.get("schema", 2), "vod_id": ctx.vp.vod_id,
                    "next_id": max(int(manifest.get("next_id", 1) or 1),
                                   max((int(c.get("id") or 0) for c in clips), default=0) + 1),
                    "clips": [], "archived_clips": history}
    save_manifest(ctx.vp, new_manifest)
    log(f"♻️ إعادة كاملة: حفظنا {len(archived)} لقطة معجبتك في الأرشيف وحذفنا {deleted} ملف. "
        f"الترقيم يكمل من #{new_manifest['next_id']:02d} والتقييمات القديمة محفوظة")
    return load_manifest(ctx.vp)


# ============================================================================
# Main entry
# ============================================================================

def _default_max_clips(ctx: _Ctx) -> int:
    return int(clamp(round(ctx.active_hours * 2.5), 4, 14))


def _choose_run(ctx: _Ctx, manifest: dict, extra_clips: bool, redo_all: bool) -> tuple:
    """Returns (run_state_or_None, message). None means nothing to do."""
    runs = _list_runs(ctx.vp)
    latest = runs[-1] if runs else None
    if redo_all:
        return _new_run(ctx, "full"), "🆕 تشغيل جديد كامل"
    if latest and latest.get("status") != "complete":
        if latest.get("transcript_tag") != _transcript_tag(ctx):
            log("ℹ️ التفريغ تغير بعد آخر تشغيل، بنكمل نفس التشغيل بأرقام الأسطر المحفوظة")
        return latest, f"♻️ نكمل التشغيل السابق {latest['run_id']} من آخر نقطة محفوظة"
    if manifest.get("clips") and not extra_clips:
        drafts = sum(1 for c in manifest["clips"] if c.get("status") == "draft")
        tail = f" ({drafts} منها مسودات تنتظر تقييمك في الاستوديو)" if drafts else ""
        return None, (f"✅ عندك {len(manifest['clips'])} لقطة لهذا البث{tail}. "
                      "فعّل EXTRA_CLIPS للقطات إضافية أو REDO_ALL_CLIPS لإعادة كاملة")
    if extra_clips and latest:
        cands = _load_cands(ctx.vp, latest["run_id"])
        existing = _existing_ranges(manifest)
        leftovers = [c for c in cands if c.get("state") in ("new", "kept", "unreviewed")
                     and not any(_overlap_short(_span(c), ex) > 0.3 for ex in existing)]
        if len(leftovers) >= 3:
            run = _new_run(ctx, "extra", parent=latest["run_id"])
            parent_phrases = load_json_safe(_run_path(ctx.vp, latest["run_id"], "phrases.json"), default=None, expected_types=(list,))
            if parent_phrases:
                atomic_write_json(_run_path(ctx.vp, run["run_id"], "phrases.json"), parent_phrases, indent=None, backup=False)
            for name in ("timeline.json", "radar.json"):
                data = load_json_safe(_run_path(ctx.vp, latest["run_id"], name), default=None, expected_types=(dict,), quiet=True)
                if data:
                    atomic_write_json(_run_path(ctx.vp, run["run_id"], name), data, backup=False)
            for c in leftovers:
                if c.get("state") == "unreviewed":
                    c["state"] = "new"
                c["miss_count"] = 0
                c["parse_fail"] = 0
                c.setdefault("media_key", f"{latest['run_id']}_{c['cid']}")
            atomic_write_json(_run_path(ctx.vp, run["run_id"], "candidates.json"), leftovers, indent=None, backup=True)
            run["status"] = "critic"
            run["reused_candidates"] = len(leftovers)
            run["critic_batch"] = latest.get("critic_batch", CRITIC_BATCH_DEFAULT)
            _save_run(ctx.vp, run)
            return run, f"➕ لقطات إضافية من {len(leftovers)} مرشح متبقي من التشغيل السابق (بدون طلبات كشاف جديدة)"
    kind = "extra" if extra_clips else "full"
    return _new_run(ctx, kind, parent=latest["run_id"] if latest else None), "🆕 تشغيل جديد"


def _load_or_make_timeline(ctx: _Ctx, run: dict) -> dict:
    path = _run_path(ctx.vp, run["run_id"], "timeline.json")
    saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
    if saved and saved.get("active"):
        ctx.set_timeline(saved)
        return saved
    timeline = compute_active_timeline(ctx)
    ctx.set_timeline(timeline)
    atomic_write_json(path, timeline)
    return timeline


def run_stage3(vod_dir: str, drive_root: str, api_key: str, extra_clips: bool = False,
               redo_all: bool = False, auto_trim_silence: bool = True, max_clips: Optional[int] = None,
               allow_unreviewed: bool = False, max_height: int = 1080, srt_bom: bool = False,
               critic_video: bool = True, render_mode: str = "on_approve",
               chunk_minutes: float = 45.0, critic_batch: int = 3) -> list:
    """Run (or resume) Stage 3 for one VOD and return the manifest clips.
    v2 arguments keep their meaning. v3 arguments:
      critic_video   send each Critic candidate as a short video window (False: audio only)
      render_mode    "on_approve" writes drafts (preview, SRT, notes) and leaves the 1080p
                     render to the Studio; "all" renders every chosen clip now
      chunk_minutes  active minutes per Scout chunk (15 to 55)
      critic_batch   candidates per Critic request (1 to 6, default 3)"""
    t_start = time.time()
    render_mode = str(render_mode or "on_approve").strip().lower()
    if render_mode not in RENDER_MODES:
        log(f"⚠️ وضع الرندر «{render_mode}» غير معروف، نستخدم on_approve")
        render_mode = "on_approve"
    chunk_active_s = float(clamp(float(chunk_minutes or 45.0) * 60.0, CHUNK_ACTIVE_MIN_S, CHUNK_ACTIVE_MAX_S))
    batch_req = int(clamp(coerce_int(critic_batch, CRITIC_BATCH_DEFAULT), 1, 6))

    vp = VodPaths(vod_dir).ensure()
    log(f"🎬 المرحلة 3 للبث {vp.vod_id}")
    ctx = _Ctx(vp, drive_root)
    ctx.max_height = int(max_height)
    manifest = load_manifest(vp)
    tagger = "AST" if ctx.feats is not None and getattr(ctx.feats, "tagger", "") == "ast" else "تقديري"
    log(f"📊 مدة البث {ctx.duration / 60:.1f} دقيقة | {len(ctx.words)} كلمة | الشات {len(ctx.chat)} رسالة | "
        f"التصنيف الصوتي: {tagger}")
    g_info = ctx.golden_info
    if g_info.get("n_golden"):
        log(f"🌟 الحمض الذهبي: {g_info['n_golden']} لقطة، تأثيرها على الذوق {float(g_info.get('alpha', 0.0)) * 100:.0f}%")

    if redo_all and manifest.get("clips"):
        manifest = archive_and_reset(ctx, manifest)

    run, message = _choose_run(ctx, manifest, extra_clips, redo_all)
    log(message)
    if run is None:
        return manifest.get("clips") or []

    ctx.set_phrases(_load_run_phrases(ctx, run))
    timeline = _load_or_make_timeline(ctx, run)
    log(f"⏱️ النشاط الفعلي {format_duration_ar(timeline.get('active_s', 0))} من {format_duration_ar(ctx.duration)} "
        f"(تخطينا {format_duration_ar(timeline.get('dead_s', 0))} وقت ميت، منه {format_duration_ar(timeline.get('afk_s', 0))} AFK)")
    peaks = load_or_make_radar(ctx, run)
    target = int(max_clips) if max_clips else _default_max_clips(ctx)
    gw = GeminiGateway(api_key, ctx.memory.gemini_state)

    try:
        # Plan and scout
        if run["status"] == "scout":
            if not run.get("chunks"):
                choice = plan_budget(ctx, gw, chunk_active_s, batch_req)
                log(choice["budget"].text_ar)
                run["chunks"] = choice["chunks"]
                run["critic_batch"] = choice["batch"]
                run["pool"] = choice["pool"]
                run["mode"] = choice["mode"]
                run["chunk_active_s"] = choice["chunk_active_s"]
                timeline["chunks"] = choice["chunks"]
                timeline["chunk_active_s"] = choice["chunk_active_s"]
                atomic_write_json(_run_path(vp, run["run_id"], "timeline.json"), timeline)
                _save_run(vp, run)
                avg = sum(c["active_s"] for c in run["chunks"]) / max(1, len(run["chunks"]))
                log(f"🧩 عدد أجزاء الكشاف: {len(run['chunks'])} (متوسط النشاط في الجزء {format_duration_ar(avg)})، "
                    f"والناقد ياخذ {run['pool']} مرشح بدفعات {run['critic_batch']}")
            scout_cands = run_scout(ctx, gw, run, manifest)
            radar = radar_candidates(ctx, peaks, scout_cands, manifest)
            if radar:
                log(f"👀 {len(radar)} لحظة من الرادار ما غطاها الكشاف، رايحة للناقد مباشرة")
            ranked = merge_and_rank(ctx, scout_cands + radar, manifest, run["run_id"])
            pool_size = int(run.get("pool") or _critic_pool(ctx.active_hours))
            afford = gw.affordable("critic")
            batch = int(run.get("critic_batch") or batch_req)
            if afford and afford * batch < min(pool_size, len(ranked)):
                if batch < CRITIC_BATCH_LOW:
                    batch = CRITIC_BATCH_LOW
                pool_size = max(batch, min(pool_size, afford * batch))
                log(f"💡 حصة الناقد المتبقية {afford} طلب، صغّرنا القائمة لـ {pool_size} مرشح بدفعات {batch}")
            pool = pick_pool(ranked, pool_size)
            pool_ids = {id(c) for c in pool}
            for c in ranked:
                if id(c) not in pool_ids:
                    c["state"] = "unreviewed"
            _save_cands(ctx, run, ranked)
            run["critic_batch"] = batch
            run["status"] = "critic"
            _save_run(vp, run)
            n_radar = sum(1 for c in pool if c.get("source") == "radar")
            log(f"🧮 {len(ranked)} مرشح بعد الدمج وحذف المكرر، {len(pool)} منها للناقد"
                + (f" ({n_radar} من الرادار)" if n_radar else ""))

        cands = _load_cands(vp, run["run_id"])

        # Critic
        if run["status"] == "critic":
            if any(c.get("state") == "new" for c in cands):
                try:
                    run_critic(ctx, gw, run, cands, batch_size=int(run.get("critic_batch") or batch_req),
                               critic_video=critic_video)
                except QuotaPause as pause:
                    if not allow_unreviewed:
                        raise
                    log(f"⚠️ {pause.message_ar}")
                    log("⚠️ بنكمل بالمرشحين بدون ناقد لأنك فعّلت السماح بذلك (بتظهر بعلامة ⚠️ في الاستوديو)")
            run["status"] = "select"
            _save_run(vp, run)

        # Select and reserve ids
        if run["status"] == "select":
            if critic_video or render_mode == "all":
                ctx.get_streams(max_height)
            chosen = select_finals(ctx, cands, target, auto_trim_silence, allow_unreviewed)
            if not chosen:
                log("😶 ما فيه لقطات عدّت فلتر الناقد والذوق في هذا التشغيل")
                run["status"] = "complete"
                _save_cands(ctx, run, cands)
                _save_run(vp, run)
                return load_manifest(vp).get("clips") or []
            manifest = load_manifest(vp)
            build_plan(ctx, run, manifest, chosen)
            _save_cands(ctx, run, cands)
            run["status"] = "render"
            run["render_mode"] = render_mode
            _save_run(vp, run)
            protected = sum(1 for f in chosen if (f["doctor"] or {}).get("protected"))
            log(f"🗂️ اخترنا {len(chosen)} لقطة وحجزنا أرقامها في ملف اللقطات"
                + (f" ({protected} فيها لحظات شاشة محمية من القص)" if protected else ""))

        # Render or drafts
        if run["status"] == "render":
            plan = load_json_safe(_run_path(vp, run["run_id"], "plan.json"), default=[], expected_types=(list,)) or []
            manifest = load_manifest(vp)
            if render_mode == "on_approve":
                log("📝 وضع الرندر عند الموافقة: نجهز مسودات بمعاينة وسبتايتل وملاحظات، والرندر 1080p من الاستوديو بعد تقييمك")
            try:
                render_plan(ctx, run, manifest, plan, srt_bom, max_height, render_mode)
            except RuntimeError as exc:
                log(f"❌ {short_error(exc, 200)} | التقدم محفوظ، شغّل الخلية مرة ثانية")
            failed = [p for p in plan if p["status"] == "failed"]
            retryable = [p for p in failed if p.get("attempts", 0) < MAX_CUT_ATTEMPTS]
            planned = [p for p in plan if p["status"] == "planned"]
            if retryable or planned:
                log(f"⚠️ {len(retryable) + len(planned)} لقطة ما خلصت، شغّل الخلية مرة ثانية ونعيد محاولتها فقط")
            else:
                run["status"] = "complete"
            _save_run(vp, run)
    except QuotaPause as pause:
        _save_run(vp, run)
        log(pause.message_ar)
        gw.flush()
        return load_manifest(vp).get("clips") or []
    except GatewayError as exc:
        _save_run(vp, run)
        log(f"{exc} | التقدم محفوظ")
        gw.flush()
        return load_manifest(vp).get("clips") or []
    finally:
        gw.flush()

    manifest = load_manifest(vp)
    clips = manifest.get("clips") or []
    run_clips = [c for c in clips if c.get("run_id") == run["run_id"]]
    drafts = [c for c in run_clips if c.get("status") == "draft"]
    log(f"\n🎉 المرحلة 3 خلصت خلال {format_duration_ar(time.time() - t_start)}: {len(run_clips)} لقطة جديدة "
        f"(المجموع {len(clips)})")
    for c in run_clips:
        badge = "" if c.get("reviewed") else " ⚠️"
        eye = " 👀" if c.get("visual_event") else ""
        state = " 📝 مسودة" if c.get("status") == "draft" else ""
        log(f"   🎞️ #{c['id']:02d} «{c['title']}» [{c['category']}] {c['duration']:.1f}ث "
            f"من {sec_to_hms(c['start'])[:8]}{state}{eye}{badge}")
    if drafts:
        log("👉 افتح الاستوديو، شوف المعاينات وقيّم. بعدها اضغط «🎬 رندر كل اللقطات المقبولة (👍)» عشان نطلع نسخ 1080p للي عجبتك بس")
    log(gw.usage_report_ar())
    return clips


# === END OF stage3_clipper.py ===
