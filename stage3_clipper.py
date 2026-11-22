"""
stage3_clipper.py
Stage 3 of TwitchClipper (Colab CPU): one shot viral clip selection and rendering.

Pipeline (every step checkpointed on Drive under vods/<id>/stage3/run_<run_id>/)
  1. Phrase index with line ids (frozen per run so saved answers always map to the same lines)
  2. Local scores (taste weighted) and 🔥 hints, nothing is filtered out
  3. Request budget plan (gateway.plan), low budget mode when the day is short on quota
  4. Scout: about 90 minute chunks, one Gemini call each, answers saved per chunk
  5. Merge, dedupe, exclude existing clips, rank before the critic
  6. Critic: batches of up to 6 candidates with transcript context and an audio snippet each
  7. Cut Doctor (deterministic), final ranking with the taste model
  8. Clip ids reserved in the manifest before rendering (ids are never reused)
  9. Render (multi input FFmpeg cutter), dynamic SRT, manifest update after every clip

QuotaPause (daily quota or a long 503 storm) saves everything and returns cleanly; the next
run resumes the same run directory without repeating finished chunks or batches.

Public contract
  run_stage3(vod_dir, drive_root, api_key, extra_clips=False, redo_all=False,
             auto_trim_silence=True, max_clips=None, allow_unreviewed=False,
             max_height=1080, srt_bom=False) -> list[clip dict] (manifest clips)
"""

from __future__ import annotations

import math
import os
import shutil
import time
from typing import Any, Optional, Sequence

from pydantic import BaseModel, field_validator

from common import (
    TMP_ROOT,
    VodPaths,
    allocate_clip_id,
    atomic_write_json,
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
    short_error,
    text_hash,
)
from gemini_gateway import GatewayError, GeminiGateway, ParseFailure, QuotaPause
from media_engine import (
    audio_snippet,
    build_phrases,
    collect_words,
    cut_and_stitch,
    cut_doctor,
    generate_srt,
    ids_to_pieces,
    load_features,
    measure_av_offset,
    pieces_net,
    resolve_stream,
    snap_to_frames,
)
from taste_engine import (
    build_signals,
    compute_clip_features,
    doctor_prefs,
    load_profile,
    local_score_for_span,
    mark_hot_phrases,
    predict_taste,
    prompt_context,
    score_windows,
)


# ============================================================================
# Configuration
# ============================================================================

CHUNK_S = 90 * 60
CHUNK_S_LOW = 165 * 60
OVERLAP_S = 120
CHUNK_TOKEN_CAP = 60_000
CLIPS_PER_CHUNK = 8
CRITIC_BATCH = 6
CRITIC_BATCH_LOW = 8
MAX_CRITIC_CANDIDATES = 24
CRITIC_PRE_S = 60.0
CRITIC_POST_S = 20.0
AUDIO_PRE_S = 15.0
AUDIO_POST_S = 10.0
AUDIO_MAX_S = 150.0
MIN_CAND_NET = 4.0
MAX_CAND_NET = 90.0
MIN_FINAL_NET = 8.0
HARD_MAX_NET = 60.0
AV_OFFSET_MIN = 0.04
MAX_CUT_ATTEMPTS = 2

CATEGORIES = ("ضحك", "صراخ_وتوتر", "تفاعل_شات", "سالفة", "قيمينق", "مقلب_وغدر")
_CATEGORY_ALIASES = {
    "ضحك": "ضحك", "كوميدي": "ضحك", "مضحك": "ضحك",
    "صراخ": "صراخ_وتوتر", "توتر": "صراخ_وتوتر", "رعب": "صراخ_وتوتر", "صراخ وتوتر": "صراخ_وتوتر",
    "تفاعل شات": "تفاعل_شات", "شات": "تفاعل_شات", "تفاعل": "تفاعل_شات",
    "سالفة": "سالفة", "قصة": "سالفة", "سوالف": "سالفة",
    "قيمينق": "قيمينق", "لعب": "قيمينق", "gaming": "قيمينق",
    "مقلب": "مقلب_وغدر", "غدر": "مقلب_وغدر", "مقلب وغدر": "مقلب_وغدر", "خيانة": "مقلب_وغدر",
}
REJECT_CODES = ("كلام_عادي", "بدون_ذروة", "صراخ_لعبة_فقط", "سياق_ناقص", "نهاية_مبتورة",
                "طويلة_مملة", "موضوع_ممنوع", "مكرر")


def _norm_category(value: Any) -> str:
    text = str(value or "").strip()
    if text in CATEGORIES:
        return text
    key = text.replace("_", " ").strip()
    for alias, cat in _CATEGORY_ALIASES.items():
        if alias == key or alias in key:
            return cat
    return "ضحك"


# ============================================================================
# Structured output schemas
# ============================================================================

class IdRange(BaseModel):
    from_id: int
    to_id: int

    @field_validator("from_id", "to_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)


class Topic(BaseModel):
    from_id: int
    to_id: int
    topic: str

    @field_validator("from_id", "to_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)


class ScoutClip(BaseModel):
    title: str
    category: str
    ranges: list[IdRange]
    peak_id: int
    payoff_end_id: int
    context_note: str = ""
    removed: list[str] = []
    scout_score: int
    why: str = ""

    @field_validator("peak_id", "payoff_end_id", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> int:
        return coerce_int(v, -1)

    @field_validator("scout_score", mode="before")
    @classmethod
    def _score(cls, v: Any) -> int:
        return coerce_int(v, 0, lo=0, hi=10)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v: Any) -> str:
        return _norm_category(v)

    @field_validator("removed", mode="before")
    @classmethod
    def _removed(cls, v: Any) -> list:
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v or []]


class ScoutResponse(BaseModel):
    clips: list[ScoutClip] = []
    topics: list[Topic] = []


class Rubric(BaseModel):
    hook: int
    payoff: int
    reaction: int
    context: int
    pacing: int

    @field_validator("hook", "payoff", "reaction", "context", "pacing", mode="before")
    @classmethod
    def _score(cls, v: Any) -> int:
        return coerce_int(v, 5, lo=0, hi=10)


class CriticVerdict(BaseModel):
    candidate: int
    keep: bool
    reject: list[str] = []
    ranges: list[IdRange] = []
    end_anchor_id: Optional[int] = None
    rubric: Rubric
    voice_check: str = "her"
    long_ok: bool = False
    title: str = ""
    category: str = ""
    cut_notes: str = ""

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
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ("true", "1", "yes", "نعم", "اي", "keep")

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
        if isinstance(v, str):
            v = [v]
        return [str(x).strip() for x in v or [] if str(x).strip()]


class CriticResponse(BaseModel):
    verdicts: list[CriticVerdict] = []


# ============================================================================
# Prompts (Arabic)
# ============================================================================

LINE_FORMAT_AR = """طريقة قراءة النص:
• كل سطر يبدأ برقم مثل #412 ثم الوقت ثم كلام الستريمر.
• أسطر بين قوسين مثل (ضحك 2.4ث) أو (صراخ 1.2ث) أو (صوت لعبة عالي 1.8ث) أو (صمت 3.1ث) هي أحداث صوتية بدون كلام، ولها أرقام وتقدر تدخلها أو تحذفها.
• العلامة 🔥 تعني مؤشرات صوتية قوية حول هذا السطر، والوسم [ضحك] أو [صراخ] يعني فيه ضحك أو صراخ أثناء الكلام.
• أسطر 💬 هي عينة من شات تويتش، بدون أرقام، ما تنفع كبداية أو نهاية.
• الستريمر تبث على تويتش وتيك توك مع بعض، فممكن تقرأ أو ترد على تعليق ما يظهر في شات تويتش. هذا تفاعل حقيقي، لا تحذفه."""

SCOUT_SYSTEM_AR = """أنت ممنتج محترف لمقاطع تيك توك وريلز، متخصص في بثوث تويتش عربية بلهجة خليجية سعودية.
مهمتك تختار أقوى اللحظات من جزء من البث وتحدد قصها الذكي بأرقام الأسطر فقط.

قواعد إلزامية:
1. كل لقطة حدث واحد له ذروة واضحة تقدر تحدد سطرها في peak_id: ضحكة قوية، فجعة، قفلة مضحكة، غدر، صدمة. إذا ما فيه ذروة لا تقترح اللقطة.
2. احفظ النهاية: النطاقات لازم تمتد لين payoff_end_id، وأي سطر ضحك أو صراخ بعد الذروة إلزامي. لا تقطع وهي تضحك.
3. البداية مباشرة: ابدأ من أول سطر يحتاجه مشاهد جديد عشان يفهم، بدون تمهيد أو ترحيب أو "طيب يا جماعة".
4. القص الذكي: مسموح تحذف فقط ترحيب بالشات خارج السالفة، مواضيع جانبية، وأسطر الصمت. ممنوع تحذف أسطر الضحك أو أسطر السياق أو قراءتها وردها على التعليقات.
5. ممنوع نهائياً: بريك الصلاة، كلام الميكب، التذمر العادي، المشي الطويل بدون حدث، صوت اللعبة بدون ردة فعل منها، والكلام العادي اللي ما يضحك.
6. افهم السياق قبل ما تختار: لا تخلط بين الألعاب أو المواضيع، واكتب في context_note وش اللعبة أو الموضوع ومين الأطراف.
7. المدة الصافية بعد القص من 12 إلى 35 ثانية غالباً، والحد الأعلى 45 ثانية.
8. scout_score من 1 إلى 10 بصراحة، وإذا ما فيه شي يستاهل رجّع قائمة clips فاضية.
9. استخدم أرقام الأسطر فقط في ranges و peak_id و payoff_end_id، ولا تكتب أوقات.

""" + LINE_FORMAT_AR

CRITIC_SYSTEM_AR = """أنت الناقد النهائي والممنتج الأخير قبل النشر على تيك توك. معك النص بأرقام الأسطر وصوت كل لقطة.
اسمع الصوت فعلاً قبل الحكم: نبرة الستريمر، هل الضحك أو الصراخ منها أو من اللعبة، وهل الضحكة خلصت.

لكل مرشح احكم:
• keep: true فقط إذا اللقطة تستاهل النشر فعلاً حسب ذوق الممنتج. اللقطة العادية أو الباردة ترفضها.
• voice_check: "her" إذا ردة الفعل منها، "game" إذا الصوت العالي من اللعبة فقط، "mixed" إذا الاثنين.
• ranges: القص النهائي بأرقام الأسطر داخل النافذة المسموحة فقط. ابدأ مباشرة من الهوك، احذف الترحيب والمواضيع الجانبية والصمت، ولا تحذف أي ضحك أو سياق ضروري.
• end_anchor_id: آخر سطر لازم يدخل عشان النهاية تكتمل (نهاية الضحكة أو ردة الفعل).
• rubric من 0 إلى 10: hook قوة البداية، payoff قوة القفلة، reaction صدق ردة الفعل، context وضوح السياق، pacing سرعة الإيقاع.
• reject: أسباب الرفض من هذه فقط: كلام_عادي، بدون_ذروة، صراخ_لعبة_فقط، سياق_ناقص، نهاية_مبتورة، طويلة_مملة، موضوع_ممنوع، مكرر.
• long_ok: true فقط إذا السالفة تحتاج أكثر من 45 ثانية فعلاً.
• title: عنوان قصير جذاب بالعامية (2 إلى 6 كلمات)، category من: ضحك، صراخ_وتوتر، تفاعل_شات، سالفة، قيمينق، مقلب_وغدر.

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
        self.afk = self.transcript.get("afk") or []
        self.meta = load_json_safe(vp.meta, default={}, expected_types=(dict,), quiet=True) or {}
        self.vod_url = self.meta.get("vod_url") or f"https://www.twitch.tv/videos/{vp.vod_id}"
        self.profile = load_profile(self.memory)
        self.rules = read_text_safe(self.memory.rules, "")
        self.feedback = load_feedback(self.memory)
        self.taste_text = prompt_context(self.profile, self.rules, self.feedback)
        self.phrases: list = []
        self.by_id: dict = {}
        self.signals = None
        self.scores: dict = {}
        self.hot_ids: set = set()
        self.chat_marks: dict = {}

    def set_phrases(self, phrases: list) -> None:
        self.phrases = phrases
        self.by_id = {int(p["id"]): p for p in phrases}
        self.signals = build_signals(self.feats, self.words, phrases, self.chat, self.duration)
        self.scores = score_windows(self.signals, self.profile)
        self.hot_ids = set(mark_hot_phrases(self.scores, phrases))
        self.chat_marks = _chat_bursts(self.chat, phrases)

    @property
    def active_hours(self) -> float:
        afk_total = sum(max(0.0, float(e) - float(s)) for s, e in self.afk)
        return max(0.1, (self.duration - afk_total) / 3600.0)


def _chat_bursts(chat: Sequence[dict], phrases: Sequence[dict], min_msgs: int = 3) -> dict:
    """phrase id -> (count, samples) for phrases that have a burst of at least min_msgs
    chat messages within 5 s of their span."""
    if not chat or not phrases:
        return {}
    times = sorted((coerce_number(m.get("t"), None), str(m.get("m", ""))) for m in chat if isinstance(m, dict))
    times = [(t, m) for t, m in times if t is not None]
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
# Rendering transcript lines for prompts
# ============================================================================

def _line(ctx: _Ctx, ph: dict, marker: str = "") -> str:
    pid = int(ph["id"])
    fire = " 🔥" if pid in ctx.hot_ids else ""
    tags = "".join(f" [{t}]" for t in ph.get("tags") or [])
    text = ph.get("text", "")
    line = f"{marker}#{pid} [{sec_to_clock(ph['s'])}]{fire} {text}{tags}"
    burst = ctx.chat_marks.get(pid)
    if burst:
        line += f"\n   💬 ×{burst[0]}: " + " | ".join(burst[1])
    return line


def _is_cold(ctx: _Ctx, ph: dict) -> bool:
    if ph.get("kind") != "speech":
        return ph.get("kind") == "silence"
    if int(ph["id"]) in ctx.hot_ids or ph.get("tags"):
        return False
    if local_score_for_span(ctx.scores, ph["s"], ph["e"]) >= 0.35:
        return False
    return True


def render_lines(ctx: _Ctx, phrases: Sequence[dict], compact: bool = False,
                 ranges: Optional[Sequence[Sequence[int]]] = None) -> str:
    """Prompt text for a list of phrases. compact collapses runs of 6+ cold lines into one
    summary line (only used when a chunk is over its token cap). ranges marks the lines a
    candidate currently keeps with ▶."""
    keep_ids = set()
    for a, b in ranges or []:
        keep_ids.update(range(int(a), int(b) + 1))
    out = []
    run: list = []

    def flush_run() -> None:
        if len(run) >= 6:
            out.append(f"… ({len(run)} سطر كلام عادي بدون مؤشرات، من #{run[0]['id']} إلى #{run[-1]['id']}، "
                       f"{sec_to_clock(run[0]['s'])} إلى {sec_to_clock(run[-1]['e'])})")
        else:
            for p in run:
                out.append(_line(ctx, p, "▶ " if int(p["id"]) in keep_ids else ("  " if ranges else "")))
        run.clear()

    for ph in phrases:
        if compact and _is_cold(ctx, ph):
            run.append(ph)
            continue
        flush_run()
        out.append(_line(ctx, ph, "▶ " if int(ph["id"]) in keep_ids else ("  " if ranges else "")))
    flush_run()
    return "\n".join(out)


def _phrases_between(ctx: _Ctx, t0: float, t1: float) -> list:
    return [p for p in ctx.phrases if p["e"] > t0 and p["s"] < t1]


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
        "chunks": [], "critic_batches": [], "selected": False,
    }
    _save_run(ctx.vp, run)
    return run


def _load_run_phrases(ctx: _Ctx, run: dict) -> list:
    phrases = load_json_safe(_run_path(ctx.vp, run["run_id"], "phrases.json"), default=None, expected_types=(list,))
    if not phrases:
        phrases = build_phrases(ctx.words, ctx.feats, vod_duration=ctx.duration)
        atomic_write_json(_run_path(ctx.vp, run["run_id"], "phrases.json"), phrases, indent=None, backup=False)
    return phrases


# ============================================================================
# Chunks and scout
# ============================================================================

def _make_chunks(duration: float, chunk_s: float) -> list:
    if duration <= chunk_s * 1.15:
        return [[0.0, round(duration, 2)]]
    n = int(math.ceil((duration - OVERLAP_S) / (chunk_s - OVERLAP_S)))
    size = (duration + (n - 1) * OVERLAP_S) / n
    chunks = []
    t = 0.0
    for _ in range(n):
        t1 = min(duration, t + size)
        chunks.append([round(t, 2), round(t1, 2)])
        t = t1 - OVERLAP_S
    chunks[-1][1] = round(duration, 2)
    return chunks


def _est_tokens(text: str) -> int:
    return int(len(text) / 3.0) + 200


def _seconds_to_ids(ctx: _Ctx, s: Any, e: Any) -> Optional[dict]:
    s_val, e_val = parse_time_to_sec(s), parse_time_to_sec(e)
    if s_val is None or e_val is None or e_val <= s_val:
        return None
    ids = [p["id"] for p in ctx.phrases if p["e"] > s_val and p["s"] < e_val]
    if not ids:
        return None
    return {"from_id": min(ids), "to_id": max(ids)}


def _normalize_ranges(ctx: _Ctx, raw: Any) -> list:
    out = []
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
        if not isinstance(obj, dict):
            return obj
        clips = obj.get("clips")
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
        out.setdefault("topics", [])
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


def _scout_prompt(ctx: _Ctx, idx: int, n: int, t0: float, t1: float, body: str,
                  carry_topics: list, existing: list) -> str:
    parts = [ctx.taste_text, ""]
    parts.append(f"📍 هذا الجزء {idx + 1} من {n} من البث ({sec_to_clock(t0)} إلى {sec_to_clock(t1)}).")
    if carry_topics:
        parts.append("🧭 آخر مواضيع الجزء اللي قبله: " + " ← ".join(carry_topics))
    near = [r for r in existing if r[1] > t0 and r[0] < t1]
    if near:
        parts.append("⛔ لقطات موجودة مسبقاً لا تكررها: " + "، ".join(f"{sec_to_clock(a)} إلى {sec_to_clock(b)}" for a, b in near))
    parts.append("")
    parts.append("النص:")
    parts.append(body)
    parts.append("")
    parts.append(
        f"المطلوب: ارسم خريطة مواضيع هذا الجزء في topics، ثم اختر حتى {CLIPS_PER_CHUNK} لقطات قوية فقط في clips "
        "(الأفضل أقل وأقوى). لكل لقطة: title و category و ranges و peak_id و payoff_end_id و context_note "
        "و removed (وش حذفت بالقص الذكي ولماذا) و scout_score و why."
    )
    return "\n".join(parts)


def run_scout(ctx: _Ctx, gw: GeminiGateway, run: dict, manifest: dict) -> list:
    """Scout every pending chunk (one call each). Finished chunks are read from Drive."""
    chunks = run["chunks"]
    existing = _existing_ranges(manifest)
    carry: list = []
    pending_parse = 0
    for i, (t0, t1) in enumerate(chunks):
        path = _run_path(ctx.vp, run["run_id"], f"scout_{i:02d}.json")
        saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
        if saved and saved.get("status") in ("done", "empty"):
            carry = [t.get("topic", "") for t in (saved.get("topics") or [])][-3:]
            continue
        phrases = _phrases_between(ctx, t0, t1)
        if not any(p.get("kind") == "speech" for p in phrases):
            atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "empty", "candidates": [], "topics": []})
            log(f"⏭️ الجزء {i + 1} من {len(chunks)} ({sec_to_clock(t0)} إلى {sec_to_clock(t1)}) بدون كلام (سكوت أو AFK)")
            continue
        body = render_lines(ctx, phrases)
        if _est_tokens(body) > CHUNK_TOKEN_CAP:
            body = render_lines(ctx, phrases, compact=True)
            log(f"   🗜️ الجزء {i + 1} طويل، ضغطنا الأسطر الباردة ({_est_tokens(body) // 1000}k توكن تقريباً)")
        prompt = _scout_prompt(ctx, i, len(chunks), t0, t1, body, carry, existing)
        log(f"🔍 الكشاف: الجزء {i + 1} من {len(chunks)} ({sec_to_clock(t0)} إلى {sec_to_clock(t1)})")
        try:
            res = gw.generate("scout", [prompt], schema=ScoutResponse, system_instruction=SCOUT_SYSTEM_AR,
                              normalizer=_scout_normalizer(ctx), label=f"جزء {i + 1}/{len(chunks)}")
        except ParseFailure as exc:
            atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "pending_parse",
                                     "raw": exc.raw_text[:20000], "model": exc.model})
            pending_parse += 1
            log(f"   ⚠️ رد الجزء {i + 1} ما انقرأ، محفوظ وبنعيد طلبه في التشغيل الجاي")
            continue
        data: ScoutResponse = res.data
        lo_id = min(p["id"] for p in phrases)
        hi_id = max(p["id"] for p in phrases)
        cands = []
        for k, clip in enumerate(data.clips):
            checked = _validate_candidate(ctx, clip.ranges, clip.peak_id, clip.payoff_end_id, lo_id, hi_id)
            if checked is None:
                continue
            cands.append({
                "title": clip.title.strip()[:80] or "لقطة", "category": clip.category,
                "ranges": checked["ranges"], "pieces": checked["pieces"], "net": checked["net"],
                "peak_id": clip.peak_id, "payoff_end_id": clip.payoff_end_id,
                "context_note": clip.context_note[:300], "removed": clip.removed[:6],
                "scout_score": clip.scout_score, "why": clip.why[:300], "chunk": i,
                "scout_model": res.model,
            })
        topics = [{"from_id": t.from_id, "to_id": t.to_id, "topic": t.topic[:120]} for t in data.topics]
        atomic_write_json(path, {"chunk": i, "t0": t0, "t1": t1, "status": "done", "model": res.model,
                                 "candidates": cands, "topics": topics, "dropped": len(data.clips) - len(cands)})
        carry = [t["topic"] for t in topics][-3:]
        log(f"   ✅ {len(cands)} مرشح من الجزء {i + 1}" + (f" (استبعدنا {len(data.clips) - len(cands)} غير صالح)" if len(data.clips) > len(cands) else ""))
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


def _topic_for(topics: Sequence[dict], pid: int) -> str:
    for t in topics:
        if coerce_int(t.get("from_id"), -1) <= pid <= coerce_int(t.get("to_id"), -1):
            return str(t.get("topic", ""))
    return ""


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


def merge_and_rank(ctx: _Ctx, cands: list, manifest: dict) -> list:
    existing = _existing_ranges(manifest)
    for c in cands:
        s, e = _span(c)
        c["local"] = local_score_for_span(ctx.scores, s, e)
        prelim = compute_clip_features(c["pieces"], ctx.signals, critic=None, category=c["category"])
        c["taste_pre"] = predict_taste(ctx.profile, prelim)
        c["rank"] = round(0.5 * c["scout_score"] / 10.0 + 0.3 * c["local"] + 0.2 * c["taste_pre"], 4)
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
        c.setdefault("state", "new")
    return kept


# ============================================================================
# Critic
# ============================================================================

def _critic_window(ctx: _Ctx, c: dict) -> tuple:
    s, e = _span(c)
    phrases = _phrases_between(ctx, s - CRITIC_PRE_S, e + CRITIC_POST_S)
    if not phrases:
        return [], -1, -1
    return phrases, min(p["id"] for p in phrases), max(p["id"] for p in phrases)


def _audio_range(c: dict, duration: float) -> tuple:
    s, e = _span(c)
    a0 = max(0.0, s - AUDIO_PRE_S)
    a1 = min(duration, e + AUDIO_POST_S)
    if a1 - a0 > AUDIO_MAX_S:
        a0 = max(0.0, s - 5.0)
        a1 = min(duration, a0 + AUDIO_MAX_S)
    return a0, a1


def _critic_contents(ctx: _Ctx, batch: list, audio_path: str) -> tuple:
    contents: list = [ctx.taste_text, "",
                      f"قدامك {len(batch)} مرشحين. لكل مرشح: معلوماته، النافذة المسموحة بأرقام الأسطر، "
                      "النص (▶ يعني السطر داخل قص الكشاف الحالي)، ثم صوت اللقطة."]
    windows = {}
    for n, c in enumerate(batch, start=1):
        phrases, lo, hi = _critic_window(ctx, c)
        windows[n] = (lo, hi)
        a0, a1 = _audio_range(c, ctx.duration)
        header = (
            f"\n━━━━ مرشح {n} ━━━━\n"
            f"العنوان المقترح: {c['title']} | التصنيف: {c['category']} | تقييم الكشاف: {c['scout_score']}/10\n"
            f"السبب: {c.get('why', '')}\n"
            f"السياق: {c.get('context_note', '')} {('| الموضوع: ' + c['topic']) if c.get('topic') else ''}\n"
            f"المحذوف بالقص: {'، '.join(c.get('removed') or []) or 'لا شي'}\n"
            f"النافذة المسموحة: من #{lo} إلى #{hi} | المدة الحالية بعد القص {c['net']:.0f} ثانية\n"
            f"النص:\n{render_lines(ctx, phrases, ranges=c['ranges'])}\n"
            f"🎧 الصوت التالي من {sec_to_clock(a0)} إلى {sec_to_clock(a1)}:"
        )
        contents.append(header)
        try:
            contents.append(audio_snippet(audio_path, a0, a1))
        except Exception as exc:
            contents.append(f"(تعذر إرفاق الصوت: {short_error(exc, 80)})")
    contents.append(
        "\nالمطلوب: verdicts فيها حكم لكل مرشح برقم candidate (1 إلى "
        f"{len(batch)}) مع keep و reject و ranges و end_anchor_id و rubric و voice_check و long_ok و title و category و cut_notes."
    )
    return contents, windows


def run_critic(ctx: _Ctx, gw: GeminiGateway, run: dict, cands: list, audio_path: str,
               batch_size: int) -> None:
    """Critic every candidate not yet reviewed. Verdicts are written into the candidates
    (state kept / rejected) and each batch is saved on Drive."""
    todo = [c for c in cands if c.get("state") == "new"]
    if not todo:
        return
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    for bi, batch in enumerate(batches):
        tag = text_hash("|".join(c["cid"] for c in batch), 8)
        path = _run_path(ctx.vp, run["run_id"], f"critic_{tag}.json")
        saved = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
        if saved and saved.get("status") == "done":
            _apply_verdicts(ctx, batch, saved.get("verdicts") or [], saved.get("windows") or {}, saved.get("model", ""))
            _save_cands(ctx, run, cands)
            continue
        log(f"🎧 الناقد: الدفعة {bi + 1} من {len(batches)} ({len(batch)} مرشح مع الصوت)")
        contents, windows = _critic_contents(ctx, batch, audio_path)
        try:
            res = gw.generate("critic", contents, schema=CriticResponse, system_instruction=CRITIC_SYSTEM_AR,
                              label=f"دفعة {bi + 1}/{len(batches)}")
        except ParseFailure as exc:
            atomic_write_json(path, {"status": "pending_parse", "raw": exc.raw_text[:20000], "cids": [c["cid"] for c in batch]})
            raise QuotaPause("critic", "busy", None,
                             "⏸️ رد الناقد ما انقرأ بعد محاولتين. التقدم محفوظ، شغّل الخلية مرة ثانية وبنعيد هذي الدفعة فقط.")
        verdicts = [v.model_dump() for v in res.data.verdicts]
        windows_json = {str(k): list(v) for k, v in windows.items()}
        atomic_write_json(path, {"status": "done", "model": res.model, "cids": [c["cid"] for c in batch],
                                 "verdicts": verdicts, "windows": windows_json})
        _apply_verdicts(ctx, batch, verdicts, windows_json, res.model)
        _save_cands(ctx, run, cands)
        kept = sum(1 for c in batch if c.get("state") == "kept")
        log(f"   ✅ الناقد قبل {kept} ورفض {len(batch) - kept}")


def _apply_verdicts(ctx: _Ctx, batch: list, verdicts: list, windows: dict, model: str) -> None:
    by_num = {}
    for v in verdicts:
        num = coerce_int(v.get("candidate"), -1)
        if 1 <= num <= len(batch) and num not in by_num:
            by_num[num] = v
    for n, c in enumerate(batch, start=1):
        v = by_num.get(n)
        if v is None:
            c["state"] = "rejected"
            c["reject"] = ["بدون_حكم"]
            continue
        lo, hi = windows.get(str(n), windows.get(n, (-1, -1)))
        rubric = v.get("rubric") or {}
        c["critic"] = {k: coerce_int(rubric.get(k), 5, lo=0, hi=10) for k in ("hook", "payoff", "reaction", "context", "pacing")}
        c["voice_check"] = v.get("voice_check", "her")
        c["long_ok"] = bool(v.get("long_ok"))
        c["cut_notes"] = str(v.get("cut_notes") or "")[:300]
        c["critic_model"] = model
        c["reject"] = [r for r in v.get("reject") or [] if r]
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
        if keep:
            lo_i, hi_i = int(lo), int(hi)
            if new_ranges and lo_i >= 0:
                checked = _validate_candidate(ctx, new_ranges, c["peak_id"], anchor if anchor is not None else -1, lo_i, hi_i)
                if checked is not None:
                    c["ranges"], c["pieces"], c["net"] = checked["ranges"], checked["pieces"], checked["net"]
            elif anchor is not None and lo_i <= anchor <= hi_i and anchor > c["ranges"][-1][1] and anchor in ctx.by_id:
                c["ranges"][-1][1] = anchor
                c["pieces"] = ids_to_pieces([{"from_id": a, "to_id": b} for a, b in c["ranges"]], ctx.phrases)
                c["net"] = round(pieces_net(c["pieces"]), 2)
        c["state"] = "kept" if keep else "rejected"


def _save_cands(ctx: _Ctx, run: dict, cands: list) -> None:
    atomic_write_json(_run_path(ctx.vp, run["run_id"], "candidates.json"), cands, indent=None, backup=True)


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
        pieces, rep = cut_doctor(c["pieces"], ctx.words, ctx.feats, ctx.duration, fps=None,
                                 peak_time=peak, prefs=prefs, auto_shave=auto_shave)
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
        critic_mean = (sum(critic.values()) / len(critic)) if critic else 5.0
        final = 0.45 * critic_mean / 10.0 + 0.20 * c["scout_score"] / 10.0 + 0.20 * c.get("local", 0.0) + 0.15 * taste
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
        plan.append({
            "id": clip_id, "uid": make_clip_uid(ctx.vp.vod_id, run["run_id"], clip_id),
            "cid": c["cid"], "title": title, "category": c["category"],
            "llm_segments": c["pieces"], "segments": f["pieces"], "doctor": f["doctor"],
            "features": f["features"], "taste": f["taste"], "final": f["final"],
            "scout": c.get("scout_score"), "local": c.get("local"), "critic": c.get("critic"),
            "reviewed": f["reviewed"], "reason": c.get("why", ""), "context_note": c.get("context_note", ""),
            "cut_notes": c.get("cut_notes", ""), "voice_check": c.get("voice_check", ""),
            "models": {"scout": c.get("scout_model", ""), "critic": c.get("critic_model", "")},
            "video_path": os.path.join(ctx.vp.clips_dir, base + ".mp4"),
            "srt_path": os.path.join(ctx.vp.clips_dir, base + ".srt"),
            "status": "planned", "attempts": 0, "error": "",
        })
        c["state"] = "selected"
    save_manifest(ctx.vp, manifest)
    atomic_write_json(_run_path(ctx.vp, run["run_id"], "plan.json"), plan)
    return plan


# ============================================================================
# Rendering
# ============================================================================

def _av_offset(ctx: _Ctx, stream) -> float:
    path = os.path.join(ctx.vp.stage3_dir, "av_offset.json")
    cached = load_json_safe(path, default=None, expected_types=(dict,), quiet=True)
    if cached and "offset" in cached:
        return float(cached["offset"] or 0.0)
    log("🎚️ فحص تزامن الصوت بين ملف الصوت ورابط الفيديو (مرة وحدة للبث)...")
    try:
        offset = measure_av_offset(stream, ctx.vp.audio, ctx.duration)
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


def render_plan(ctx: _Ctx, run: dict, manifest: dict, plan: list, srt_bom: bool, max_height: int) -> int:
    pending = [p for p in plan if p["status"] in ("planned", "failed") and p.get("attempts", 0) < MAX_CUT_ATTEMPTS]
    if not pending:
        return 0
    ensure_dir(ctx.vp.clips_dir)
    stream = resolve_stream(ctx.vod_url, max_height=max_height)
    offset = _av_offset(ctx, stream)
    done = 0
    for n, p in enumerate(pending, start=1):
        p["attempts"] = int(p.get("attempts", 0)) + 1
        pieces = snap_to_frames(p["segments"], stream.fps)
        net = pieces_net(pieces)
        log(f"✂️ [{n}/{len(pending)}] لقطة #{p['id']:02d} «{p['title']}» ({len(pieces)} قطعة، {net:.1f} ثانية)")
        t0 = time.time()
        try:
            info = cut_and_stitch(stream, pieces, p["video_path"], time_offset=offset, max_height=max_height)
            srt = generate_srt(ctx.words, pieces, p["srt_path"], fps=stream.fps, bom=srt_bom)
        except Exception as exc:
            p["status"] = "failed"
            p["error"] = short_error(exc, 300)
            atomic_write_json(_run_path(ctx.vp, run["run_id"], "plan.json"), plan)
            log(f"   ❌ فشل قص اللقطة #{p['id']:02d}: {p['error']}")
            continue
        if srt["errors"]:
            log(f"   ⚠️ ملاحظات على السبتايتل: {'، '.join(srt['errors'][:3])}")
        span = pieces[-1][1] - pieces[0][0]
        clip = normalize_clip({
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
            "video_path": p["video_path"], "srt_path": p["srt_path"],
            "srt_stats": {k: srt[k] for k in ("cues", "min_cue_s", "avg_cps", "avg_wps", "bands")},
            "avg_wps": srt.get("avg_wps", 0.0),
            "render": {k: info.get(k) for k in ("encoder", "size_mb", "height", "fps", "refreshed", "peak_mb")},
            "status": "ready", "created_at": now_iso(),
        }, ctx.vp.vod_id)
        _, existing = _find_by_uid(manifest, p["uid"])
        if existing is not None:
            manifest["clips"] = [c for c in manifest["clips"] if c.get("uid") != p["uid"]]
        manifest["clips"].append(clip)
        save_manifest(ctx.vp, manifest)
        p["status"] = "cut"
        p["error"] = ""
        atomic_write_json(_run_path(ctx.vp, run["run_id"], "plan.json"), plan)
        done += 1
        badge = "" if p.get("reviewed") else " ⚠️ بدون ناقد"
        log(f"   ✅ جاهزة خلال {time.time() - t0:.0f} ثانية ({info['size_mb']:.1f} MB، {srt['cues']} سطر سبتايتل){badge}")
    return done


def _find_by_uid(manifest: dict, uid: str) -> tuple:
    for i, c in enumerate(manifest.get("clips") or []):
        if c.get("uid") == uid:
            return i, c
    return None, None


# ============================================================================
# REDO: archive liked clips, delete the rest, keep ids moving forward
# ============================================================================

def archive_and_reset(ctx: _Ctx, manifest: dict) -> dict:
    clips = manifest.get("clips") or []
    if not clips:
        return manifest
    stamp = new_run_id()
    archive_dir = os.path.join(ctx.vp.clips_archive_dir, stamp)
    archived, deleted = [], 0
    for c in clips:
        paths = [c.get("video_path"), c.get("srt_path")]
        if c.get("rating") == "liked":
            ensure_dir(archive_dir)
            moved = {}
            for path in paths:
                if path and os.path.isfile(path):
                    dst = os.path.join(archive_dir, os.path.basename(path))
                    shutil.move(path, dst)
                    moved[path] = dst
            archived.append({"uid": c.get("uid"), "id": c.get("id"), "title": c.get("title"),
                             "rating": c.get("rating"), "video_path": moved.get(c.get("video_path"), ""),
                             "srt_path": moved.get(c.get("srt_path"), "")})
        else:
            for path in paths:
                if path and os.path.isfile(path):
                    safe_delete(path, ctx.drive_root)
                    deleted += 1
    history = list(manifest.get("archived_clips") or []) + archived
    new_manifest = {"schema": manifest.get("schema", 2), "vod_id": ctx.vp.vod_id,
                    "next_id": manifest.get("next_id", 1), "clips": [], "archived_clips": history}
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
        return None, (f"✅ عندك {len(manifest['clips'])} لقطة جاهزة لهذا البث. "
                      "فعّل EXTRA_CLIPS للقطات إضافية أو REDO_ALL_CLIPS لإعادة كاملة")
    if extra_clips and latest:
        cands = load_json_safe(_run_path(ctx.vp, latest["run_id"], "candidates.json"), default=[], expected_types=(list,), quiet=True) or []
        existing = _existing_ranges(manifest)
        leftovers = [c for c in cands if c.get("state") in ("new", "kept")
                     and not any(_overlap_short(_span(c), ex) > 0.3 for ex in existing)]
        if len(leftovers) >= 3:
            run = _new_run(ctx, "extra", parent=latest["run_id"])
            parent_phrases = load_json_safe(_run_path(ctx.vp, latest["run_id"], "phrases.json"), default=None, expected_types=(list,))
            if parent_phrases:
                atomic_write_json(_run_path(ctx.vp, run["run_id"], "phrases.json"), parent_phrases, indent=None, backup=False)
            atomic_write_json(_run_path(ctx.vp, run["run_id"], "candidates.json"), leftovers, indent=None, backup=True)
            run["status"] = "critic"
            run["reused_candidates"] = len(leftovers)
            _save_run(ctx.vp, run)
            return run, f"➕ لقطات إضافية من {len(leftovers)} مرشح متبقي من التشغيل السابق (بدون طلبات كشاف جديدة)"
    kind = "extra" if extra_clips else "full"
    return _new_run(ctx, kind, parent=latest["run_id"] if latest else None), "🆕 تشغيل جديد"


def run_stage3(vod_dir: str, drive_root: str, api_key: str, extra_clips: bool = False,
               redo_all: bool = False, auto_trim_silence: bool = True, max_clips: Optional[int] = None,
               allow_unreviewed: bool = False, max_height: int = 1080, srt_bom: bool = False) -> list:
    t_start = time.time()
    vp = VodPaths(vod_dir).ensure()
    log(f"🎬 المرحلة 3 للبث {vp.vod_id}")
    ctx = _Ctx(vp, drive_root)
    manifest = load_manifest(vp)
    log(f"📊 مدة البث {ctx.duration / 60:.1f} دقيقة (نشطة {ctx.active_hours:.1f} ساعة) | "
        f"{len(ctx.words)} كلمة | الشات {len(ctx.chat)} رسالة | التصنيف الصوتي: "
        f"{'AST' if ctx.feats is not None and ctx.feats.tagger == 'ast' else 'تقديري'}")

    if redo_all and manifest.get("clips"):
        manifest = archive_and_reset(ctx, manifest)

    run, message = _choose_run(ctx, manifest, extra_clips, redo_all)
    log(message)
    if run is None:
        return manifest.get("clips") or []

    ctx.set_phrases(_load_run_phrases(ctx, run))
    target = int(max_clips) if max_clips else _default_max_clips(ctx)
    gw = GeminiGateway(api_key, ctx.memory.gemini_state)

    try:
        # Scout
        if run["status"] == "scout":
            if not run.get("chunks"):
                chunks = _make_chunks(ctx.duration, CHUNK_S)
                need_critic = max(1, math.ceil(min(MAX_CRITIC_CANDIDATES, len(chunks) * 6) / CRITIC_BATCH))
                budget = gw.plan({"scout": len(chunks), "critic": need_critic})
                if not budget.ok and len(chunks) > 1:
                    low_chunks = _make_chunks(ctx.duration, CHUNK_S_LOW)
                    if len(low_chunks) < len(chunks):
                        chunks = low_chunks
                        run["mode"] = "low_budget"
                        log(f"💡 الحصة قليلة اليوم، شغّلنا وضع التوفير: {len(chunks)} أجزاء أطول بدل الأجزاء العادية")
                        budget = gw.plan({"scout": len(chunks), "critic": max(1, math.ceil(MAX_CRITIC_CANDIDATES / CRITIC_BATCH_LOW))})
                log(budget.text_ar)
                run["chunks"] = chunks
                _save_run(vp, run)
            cands = run_scout(ctx, gw, run, manifest)
            ranked = merge_and_rank(ctx, cands, manifest)
            limit = MAX_CRITIC_CANDIDATES
            batch = CRITIC_BATCH_LOW if run.get("mode") == "low_budget" else CRITIC_BATCH
            afford = gw.affordable("critic")
            if afford and afford * batch < min(limit, len(ranked)):
                batch = CRITIC_BATCH_LOW
                limit = max(batch, min(limit, afford * batch))
            ranked = ranked[:limit]
            _save_cands(ctx, run, ranked)
            run["critic_batch"] = batch
            run["status"] = "critic"
            _save_run(vp, run)
            log(f"🧮 {len(ranked)} مرشح للناقد بعد الدمج وحذف المكرر")

        cands = load_json_safe(_run_path(vp, run["run_id"], "candidates.json"), default=[], expected_types=(list,)) or []

        # Critic
        if run["status"] == "critic":
            if any(c.get("state") == "new" for c in cands):
                local_audio = os.path.join(ensure_dir(os.path.join(TMP_ROOT, "stage3")), f"{vp.vod_id}.mp3")
                if not os.path.isfile(local_audio) or os.path.getsize(local_audio) != os.path.getsize(vp.audio):
                    shutil.copyfile(vp.audio, local_audio)
                try:
                    run_critic(ctx, gw, run, cands, local_audio, int(run.get("critic_batch", CRITIC_BATCH)))
                except QuotaPause as pause:
                    if not allow_unreviewed:
                        raise
                    log(f"⚠️ {pause.message_ar}")
                    log("⚠️ بنكمل بالمرشحين بدون ناقد لأنك فعّلت السماح بذلك (بتظهر بعلامة ⚠️ في الاستوديو)")
            run["status"] = "select"
            _save_run(vp, run)

        # Select and reserve ids
        if run["status"] == "select":
            chosen = select_finals(ctx, cands, target, auto_trim_silence, allow_unreviewed)
            if not chosen:
                log("😶 ما فيه لقطات عدّت فلتر الناقد والذوق في هذا التشغيل")
                run["status"] = "complete"
                _save_cands(ctx, run, cands)
                _save_run(vp, run)
                return manifest.get("clips") or []
            build_plan(ctx, run, manifest, chosen)
            _save_cands(ctx, run, cands)
            run["status"] = "render"
            _save_run(vp, run)
            log(f"🗂️ اخترنا {len(chosen)} لقطة وحجزنا أرقامها في ملف اللقطات قبل القص")

        # Render
        if run["status"] == "render":
            plan = load_json_safe(_run_path(vp, run["run_id"], "plan.json"), default=[], expected_types=(list,)) or []
            manifest = load_manifest(vp)
            render_plan(ctx, run, manifest, plan, srt_bom, max_height)
            failed = [p for p in plan if p["status"] == "failed"]
            retryable = [p for p in failed if p.get("attempts", 0) < MAX_CUT_ATTEMPTS]
            if retryable:
                log(f"⚠️ {len(retryable)} لقطة فشل قصها، شغّل الخلية مرة ثانية ونعيد محاولتها فقط")
            else:
                run["status"] = "complete"
            _save_run(vp, run)
    except QuotaPause as pause:
        _save_run(vp, run)
        log(pause.message_ar)
        return load_manifest(vp).get("clips") or []
    except GatewayError as exc:
        _save_run(vp, run)
        log(f"{exc} | التقدم محفوظ")
        return load_manifest(vp).get("clips") or []

    manifest = load_manifest(vp)
    clips = manifest.get("clips") or []
    run_clips = [c for c in clips if c.get("run_id") == run["run_id"]]
    log(f"\n🎉 المرحلة 3 خلصت خلال {format_duration_ar(time.time() - t_start)}: {len(run_clips)} لقطة جديدة "
        f"(المجموع {len(clips)})")
    for c in run_clips:
        badge = "" if c.get("reviewed") else " ⚠️"
        log(f"   🎞️ #{c['id']:02d} «{c['title']}» [{c['category']}] {c['duration']:.1f}ث "
            f"من {sec_to_hms(c['start'])}{badge}")
    log(gw.usage_report_ar())
    return clips


# === END OF stage3_clipper.py ===
