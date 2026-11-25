"""
taste_engine.py
Learned editor taste for the TwitchClipper pipeline: per second signals, local window
scoring, quantitative clip features, a regularized taste model trained on Studio ratings,
the Reaction Radar, Golden priors, and the Arabic prompt context that carries rules,
preferences and examples to Gemini. Version 3.

Main contracts
  build_signals(feats, words, phrases, chat, duration) -> Signals
  score_windows(signals, profile) -> {"windows", "hot", "hot_phrase_ids", "p99"}
  local_score_for_span(scores, s, e) -> float in 0..1
  compute_clip_features(pieces, signals, critic=None, category=None) -> dict
  load_profile(memory_paths) / save_profile(memory_paths, profile)
  predict_taste(profile, features) -> float in 0..1
  train_profile(feedback_entries, prev_profile=None) -> profile
  prompt_context(profile, rules_text, feedback_entries, ..., golden_text="") -> str (Arabic)
  backfill_legacy(feedback_entries, drive_root) -> (entries, filled_count)
  text_cues(text) -> {"excite", "story", "mundane", "apology", "chaos"}
  radar_peaks(signals, per_hour, min_score, min_gap_s, exclude, active_spans) -> [{"t", "score", "why"}]
  mark_radar_phrases(peaks, phrases, radius_s) -> phrase ids to mark with 👀
  apply_golden_priors(profile, golden_stats, fade_n=35) -> profile copy with Golden priors
  parse_note_score(note) -> float | None
  chat_health(chat, duration) -> float in 0..1

Design notes
  * Quiet storytelling is protected: the story bonus term T*S*(1-L) rewards dense speech
    with story cues even when loudness is low, and nothing here filters lines out.
  * Missing Twitch chat never counts against a moment: chat weight scales with chat health.
  * The taste model is logistic on 0..1 features, anchored to hand set prior weights with an
    L2 penalty (lambda), so a handful of ratings nudges it instead of overturning it.
  * v3 Reaction Radar: signal V marks laughter or screams that come with almost no verbal
    setup in the previous 8 s (something happened on screen), boosted by apology or chaos
    words near it and by chat asking "وش صار". It needs no Gemini request.
  * v3 Golden priors: duration, ending and category preferences from the editor's finished
    edits, blended in with a weight that fades linearly to zero at fade_n rated clips.
    Applied per run on a copy; save_profile always stores the trained values, never the blend.
  * Old feedback entries and profiles load as they are: the two new visual features read 0
    in old entries and take their prior weight in old profiles.
"""

from __future__ import annotations

import copy
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from common import (
    MemoryPaths,
    VodPaths,
    atomic_write_json,
    clamp,
    coerce_number,
    load_json_safe,
    load_manifest,
    log,
    normalize_arabic,
    normalize_digits,
    now_iso,
    short_error,
    spans_total,
)
from media_engine import Features, collect_words, load_features, merge_pieces, pieces_net


# ============================================================================
# Lexicons (normalized with common.normalize_arabic at import time)
# ============================================================================

_EXCITE_RAW = (
    "ههه", "هههه", "والله", "اقسم بالله", "قسم بالله", "ورب الكعبة", "يا ويلي", "يا ويلي يا امي",
    "يمه", "يما", "مستحيل", "شوفوا", "لا لا", "يا جماعة", "وش ذا", "وش هذا", "يا الله",
    "ما اصدق", "يا ربي", "اوه", "واو", "فجعت", "فجعني", "خوفتني", "خفت", "ابقر", "يا شيخ",
    "مو معقول", "يا حبيبي", "الحقوا", "ساعدوني", "ذبحني", "غدرت", "خانتني",
)
_STORY_RAW = (
    "مرة", "تخيلوا", "اسمعوا", "سالفة", "قصة", "صار لي", "صارت لي", "تذكرون", "اقولكم",
    "خلوني اقولكم", "قبل كم يوم", "امس", "البارح", "مرة من المرات", "تدرون وش صار", "اللي صار",
)
_MUNDANE_RAW = (
    "اذان", "الاذان", "صلاة", "الصلاة", "اصلي", "بريك", "بريك صلاة", "ميكب", "مكياج", "رموش",
    "روج", "حمام", "تعبانة", "نعسانة", "راسي يعورني", "جوعانة",
)
_APOLOGY_RAW = (
    "سوري", "سوري سوري", "اسف", "آسف", "اسفه", "آسفة", "ما قصدت", "ماقصدت", "والله ما قصدت",
    "مو قصدي", "مب قصدي", "بالغلط", "غلطة", "غلطتي", "سامحني", "سامحيني", "سامحوني", "اعتذر",
    "sorry",
)
_CHAOS_RAW = (
    "وش سويت", "وش سويتي", "ايش سويت", "شسويت", "اهرب", "اهربي", "اهربوا", "لا لا لا",
    "يا ويلي", "الحقوا", "الحقي", "الحقني", "طاح", "طحت", "طاحت", "مات", "ماتت", "مت",
    "دفيته", "دفيتها", "دفعته", "دفعتها", "دزيته", "دزيتها", "وش صار", "غدر", "غدرته", "غدرتها",
    "قتلته", "قتلتها", "ذبحته", "ذبحتها", "انقلع", "وين راح", "وين راحت",
)
_CHAT_WHAT_RAW = (
    "وش صار", "ايش صار", "شصار", "وش صاير", "شنو صار", "وش سوت", "ايش سوت", "وش سويتي",
    "ههههه", "لووول", "lol", "lmao",
)
# Interjections that are reactions, not verbal setup (excluded when counting setup words)
_INTERJECTION_RAW = (
    "لا", "يا", "اه", "اوه", "واو", "وي", "ياي", "ها", "هاه", "اي", "ايه", "يمه", "يما",
    "اوو", "اووه", "يع", "اف",
)


def _norm_list(raw: Sequence[str]) -> tuple:
    out = []
    for item in raw:
        n = normalize_arabic(item)
        if n and n not in out:
            out.append(n)
    return tuple(out)


EXCITE_CUES = _norm_list(_EXCITE_RAW)
STORY_CUES = _norm_list(_STORY_RAW)
MUNDANE_CUES = _norm_list(_MUNDANE_RAW)
APOLOGY_CUES = _norm_list(_APOLOGY_RAW)
CHAOS_CUES = _norm_list(_CHAOS_RAW)
CHAT_WHAT_CUES = _norm_list(_CHAT_WHAT_RAW)
INTERJECTIONS = frozenset(_norm_list(_INTERJECTION_RAW))
_LAUGH_TOKEN = normalize_arabic("هههه")  # collapses to "هه"
_WORD_PUNCT = "؟?!.،,؛;:…\"'«»()[]{} "


def _token_variants(token: str) -> set:
    """A token plus its form without common Arabic proclitics (و، ف، ب، ال)."""
    out = {token}
    for prefix in ("وال", "فال", "بال", "ال", "و", "ف", "ب"):
        if token.startswith(prefix) and len(token) - len(prefix) >= 2:
            out.add(token[len(prefix):])
    return out


def _count_cues(norm_text: str, tokens_variants: list, cues: tuple) -> int:
    count = 0
    for cue in cues:
        if " " in cue:
            count += norm_text.count(cue)
        else:
            count += sum(1 for variants in tokens_variants if cue in variants)
    return count


def _is_laugh_token(token: str) -> bool:
    return _LAUGH_TOKEN in token and set(token) <= {"ه", "ا"}


def is_setup_word(word: str) -> bool:
    """True for a word that carries meaning (verbal setup), False for laughter tokens,
    interjections and pure punctuation."""
    norm = normalize_arabic(word or "").strip(_WORD_PUNCT)
    if not norm or _is_laugh_token(norm):
        return False
    return norm not in INTERJECTIONS


def text_cues(text: str) -> dict:
    """Counts of excitement, story, mundane, apology and chaos cues in a piece of transcript text."""
    norm = normalize_arabic(text or "")
    if not norm:
        return {"excite": 0, "story": 0, "mundane": 0, "apology": 0, "chaos": 0}
    padded = f" {norm} "
    tokens = norm.split()
    variants = [_token_variants(t) for t in tokens]
    excite = _count_cues(padded, variants, EXCITE_CUES)
    excite += sum(1 for t in tokens if _LAUGH_TOKEN in t and set(t) <= {"ه", "ا"})
    return {
        "excite": int(excite),
        "story": int(_count_cues(padded, variants, STORY_CUES)),
        "mundane": int(_count_cues(padded, variants, MUNDANE_CUES)),
        "apology": int(_count_cues(padded, variants, APOLOGY_CUES)),
        "chaos": int(_count_cues(padded, variants, CHAOS_CUES)),
    }


def chat_what_cues(text: str) -> int:
    """Chat messages asking what just happened ("وش صار", "؟؟") or exploding in laughter."""
    raw = str(text or "")
    count = 1 if ("؟؟" in raw or "??" in raw) else 0
    norm = normalize_arabic(raw)
    if not norm:
        return count
    padded = f" {norm} "
    tokens = norm.split()
    variants = [_token_variants(t) for t in tokens]
    return count + _count_cues(padded, variants, CHAT_WHAT_CUES)


# ============================================================================
# Feature catalogue and defaults
# ============================================================================

CATEGORY_KEYS = {
    "ضحك": "cat_laugh",
    "صراخ_وتوتر": "cat_scream",
    "تفاعل_شات": "cat_chat",
    "سالفة": "cat_story",
    "قيمينق": "cat_gaming",
    "مقلب_وغدر": "cat_prank",
}

FEATURE_NAMES = (
    "duration_z", "removed_ratio", "n_pieces_z", "reaction_peak", "laugh_s", "game_only_frac",
    "loud_p90", "speech_wps", "chat_peak", "excite_cue", "story_cue", "mundane_cue",
    "ends_on_reaction", "starts_with_speech", "silence_frac", "critic_mean",
    "visual_score", "visual_suspect",
) + tuple(CATEGORY_KEYS.values())

DEFAULT_PRIOR = {
    "duration_z": -0.30, "removed_ratio": 0.20, "n_pieces_z": 0.0, "reaction_peak": 0.90,
    "laugh_s": 0.60, "game_only_frac": -0.80, "loud_p90": 0.20, "speech_wps": 0.20,
    "chat_peak": 0.20, "excite_cue": 0.40, "story_cue": 0.40, "mundane_cue": -0.90,
    "ends_on_reaction": 0.70, "starts_with_speech": 0.30, "silence_frac": -0.70,
    "critic_mean": 1.00, "visual_score": 0.60, "visual_suspect": 0.40, "cat_laugh": 0.0, "cat_scream": 0.0, "cat_chat": 0.0,
    "cat_story": 0.0, "cat_gaming": 0.0, "cat_prank": 0.0,
}
DEFAULT_BIAS = -1.2

FEATURE_LABELS_AR = {
    "duration_z": "طول اللقطة",
    "removed_ratio": "كثافة القص",
    "n_pieces_z": "عدد القطع",
    "reaction_peak": "قوة ذروة ردة الفعل",
    "laugh_s": "مدة الضحك",
    "game_only_frac": "صوت اللعبة بدون ردة فعل منها",
    "loud_p90": "علو الصوت",
    "speech_wps": "سرعة الكلام",
    "chat_peak": "تفاعل الشات",
    "excite_cue": "كلمات الحماس",
    "story_cue": "وجود سالفة",
    "mundane_cue": "كلام عادي (صلاة، ميكب، تذمر)",
    "ends_on_reaction": "النهاية على ردة فعل",
    "starts_with_speech": "بداية مباشرة بدون فراغ",
    "silence_frac": "سكوت داخل اللقطة",
    "critic_mean": "تقييم الناقد",
    "visual_score": "قوة اللحظة البصرية (حسب الناقد)",
    "visual_suspect": "ضحك أو اعتذار بدون كلام قبله (شي صار في الشاشة)",
    "cat_laugh": "تصنيف ضحك",
    "cat_scream": "تصنيف صراخ وتوتر",
    "cat_chat": "تصنيف تفاعل شات",
    "cat_story": "تصنيف سالفة",
    "cat_gaming": "تصنيف قيمينق",
    "cat_prank": "تصنيف مقلب وغدر",
}

DEFAULT_LOCAL_WEIGHTS = {
    "w_r": 0.30, "w_l": 0.10, "w_s": 0.10, "w_c": 0.15, "w_x": 0.15, "w_t": 0.10,
    "w_q": 0.10, "w_g": 0.15, "w_m": 0.15, "w_sil": 0.10, "w_v": 0.10,
}
# local weight -> (clip feature, sign). sign -1 means "more negative learned weight -> bigger penalty".
_LOCAL_MAP = {
    "w_r": ("reaction_peak", 1), "w_l": ("loud_p90", 1), "w_s": ("speech_wps", 1),
    "w_c": ("chat_peak", 1), "w_x": ("excite_cue", 1), "w_t": ("story_cue", 1),
    "w_q": ("story_cue", 1), "w_g": ("game_only_frac", -1), "w_m": ("mundane_cue", -1),
    "w_sil": ("silence_frac", -1), "w_v": ("visual_suspect", 1),
}

DEFAULT_PREFS = {
    "target_s": [12.0, 35.0],
    "end_pad_s": 0.30,
    "start_pad_s": 0.12,
    "max_removed_ratio": 0.45,
}

DEFAULT_LAMBDA = 4.0
_POSITIVE_HINTS = ("حلو", "رهيب", "ممتاز", "مضحك", "مضبوط", "جامد", "عجبني")
WINDOW_S = 45.0
STEP_S = 5.0
HOT_PER_HOUR = 8
SETUP_LOOKBACK_S = 8
SETUP_FULL_WORDS = 10.0
CUE_REACH_S = 8
CHAT_LAG_S = 20
RADAR_REACTION_MIN = 0.3

_MOMENT_TO_CAT = {
    "فشل_بصري": "cat_laugh",
    "غدر_ومقلب": "cat_prank",
    "ضحك_هستيري": "cat_laugh",
    "سالفة": "cat_story",
    "تفاعل_شات": "cat_chat",
    "رعب_وفجعة": "cat_scream",
    "لعب_قوي": "cat_gaming",
}
_VISUAL_MOMENTS = ("فشل_بصري", "غدر_ومقلب")
_REACTION_ENDINGS = ("ذروة_ضحك", "ردة_فعل")
GOLDEN_FADE_N = 35


def default_profile() -> dict:
    return {
        "version": 1,
        "n": 0,
        "lambda": DEFAULT_LAMBDA,
        "bias": DEFAULT_BIAS,
        "weights": dict(DEFAULT_PRIOR),
        "prior": dict(DEFAULT_PRIOR),
        "mean": {},
        "std": {},
        "local_weights": dict(DEFAULT_LOCAL_WEIGHTS),
        "prefs": {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_PREFS.items()},
        "tag_counts": {},
        "updated_at": "",
    }


def load_profile(memory_paths: MemoryPaths) -> dict:
    """Taste profile with every key present (defaults fill anything missing)."""
    prof = default_profile()
    raw = load_json_safe(memory_paths.taste, default={}, expected_types=(dict,), quiet=True) or {}
    for key in ("version", "n", "lambda", "bias", "updated_at"):
        if key in raw:
            prof[key] = raw[key]
    for key in ("weights", "prior", "mean", "std", "local_weights", "prefs", "tag_counts"):
        if isinstance(raw.get(key), dict):
            prof[key].update(raw[key])
    for name in FEATURE_NAMES:
        prof["weights"].setdefault(name, DEFAULT_PRIOR.get(name, 0.0))
        prof["prior"].setdefault(name, DEFAULT_PRIOR.get(name, 0.0))
    target = prof["prefs"].get("target_s")
    if not (isinstance(target, (list, tuple)) and len(target) == 2):
        prof["prefs"]["target_s"] = list(DEFAULT_PREFS["target_s"])
    return prof


def save_profile(memory_paths: MemoryPaths, profile: dict) -> str:
    """Save the trained profile. If Golden priors were blended into this copy, the trained
    values kept in profile["golden"]["base"] are written instead, so blends never compound."""
    memory_paths.ensure()
    data = copy.deepcopy(profile)
    golden = data.pop("golden", None)
    if isinstance(golden, dict) and isinstance(golden.get("base"), dict):
        for key, value in golden["base"].items():
            data[key] = value
    return atomic_write_json(memory_paths.taste, data, backup=True)


# ============================================================================
# Per second signals
# ============================================================================

def chat_health(chat: Optional[Sequence[dict]], duration: float) -> float:
    """0..1. About 300 messages per hour or more counts as a healthy chat."""
    if not chat or not duration or duration <= 0:
        return 0.0
    per_hour = len(chat) / (duration / 3600.0)
    return float(clamp(per_hour / 300.0, 0.0, 1.0))


def _box(arr: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width == 1 or not len(arr):
        return arr.astype(np.float32)
    kernel = np.ones(width, dtype=np.float32)
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not len(mask):
        return mask.astype(bool)
    return _box(mask.astype(np.float32), 2 * radius + 1) > 0


def _rolling_median_blocks(arr: np.ndarray, block: int, reach: int) -> np.ndarray:
    n = len(arr)
    if n == 0:
        return arr.copy()
    nb = int(math.ceil(n / block))
    med = np.array([np.median(arr[i * block:(i + 1) * block]) for i in range(nb)], dtype=np.float32)
    smooth = np.array([np.median(med[max(0, i - reach):i + reach + 1]) for i in range(nb)], dtype=np.float32)
    return np.repeat(smooth, block)[:n]


@dataclass
class Signals:
    """Per second (1 Hz) signals for one VOD, all in 0..1 except x_raw and words_per_s."""
    duration: float
    R: np.ndarray
    L: np.ndarray
    S: np.ndarray
    C: np.ndarray
    X: np.ndarray
    x_raw: np.ndarray
    T: np.ndarray
    M: np.ndarray
    G: np.ndarray
    SIL: np.ndarray
    words_per_s: np.ndarray
    laugh: np.ndarray
    chat_h: float
    words: list = field(default_factory=list)
    word_starts: list = field(default_factory=list)
    feats: Optional[Features] = None
    V: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    A: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    apology: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    chaos: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    chat_q: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    setup_words: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    @property
    def n(self) -> int:
        return len(self.R)

    def idx_range(self, s: float, e: float) -> tuple:
        a = int(clamp(math.floor(s), 0, max(0, self.n - 1)))
        b = int(clamp(math.ceil(e), a + 1, self.n))
        return a, b


def _max_filter(arr: np.ndarray, before: int, after: int) -> np.ndarray:
    """out[t] = max(arr[t - before : t + after + 1]) with edges clipped."""
    n = len(arr)
    if n == 0:
        return arr.astype(np.float32)
    out = arr.astype(np.float32).copy()
    for k in range(1, max(before, after) + 1):
        if k >= n:
            break
        if k <= after:
            out[:n - k] = np.maximum(out[:n - k], arr[k:])
        if k <= before:
            out[k:] = np.maximum(out[k:], arr[:n - k])
    return out


def _window_sum_before(arr: np.ndarray, start_back: int, end_back: int) -> np.ndarray:
    """out[t] = sum(arr[t - start_back : t - end_back]) (the seconds just before t)."""
    n = len(arr)
    cs = np.concatenate([[0.0], np.cumsum(arr.astype(np.float64))])
    idx = np.arange(n)
    a = np.clip(idx - start_back, 0, n)
    b = np.clip(idx - end_back, 0, n)
    return np.maximum(0.0, cs[b] - cs[a]).astype(np.float32)


def build_signals(
    feats: Optional[Features],
    words: Sequence[dict],
    phrases: Optional[Sequence[dict]] = None,
    chat: Optional[Sequence[dict]] = None,
    duration: Optional[float] = None,
) -> Signals:
    """Build the 1 Hz signal set used by window scoring and clip features."""
    words = list(words or [])
    dur = float(duration or (feats.duration if feats is not None else 0.0) or (words[-1]["e"] if words else 0.0))
    n = max(1, int(math.ceil(dur)))

    words_per_s = np.zeros(n, dtype=np.float32)
    for w in words:
        i = int((w["s"] + w["e"]) / 2.0)
        if 0 <= i < n:
            words_per_s[i] += 1.0
    has_words = words_per_s > 0
    near_words_1 = _dilate(has_words, 1)
    near_words_2 = _dilate(has_words, 2)

    def ev(name: str) -> np.ndarray:
        if feats is None:
            return np.zeros(n, dtype=np.float32)
        arr = feats.event(name)
        if arr is None:
            return np.zeros(n, dtype=np.float32)
        out = np.zeros(n, dtype=np.float32)
        m = min(n, len(arr))
        out[:m] = arr[:m]
        return out

    laugh = ev("laugh")
    scream = np.maximum(ev("scream"), ev("shout"))
    game_sfx = ev("game_sfx")
    gate = np.where(near_words_1 | (laugh > 0.3), 1.0, 0.3).astype(np.float32)
    R = np.clip(np.maximum(laugh, scream * gate), 0.0, 1.0)

    if feats is not None and len(feats.rms_db):
        per = int(round(1.0 / feats.hop10))
        total = n * per
        rms = np.full(total, float(np.min(feats.rms_db)), dtype=np.float32)
        flo = np.full(total, float(np.min(feats.floor_db)), dtype=np.float32)
        sp = np.zeros(total, dtype=np.float32)
        m = min(total, len(feats.rms_db))
        rms[:m] = feats.rms_db[:m]
        flo[:m] = feats.floor_db[:m]
        sp[:m] = feats.speech[:m]
        rms1 = rms.reshape(n, per).mean(axis=1)
        above1 = (rms - flo).reshape(n, per).mean(axis=1)
        speech1 = sp.reshape(n, per).mean(axis=1)
        base = _rolling_median_blocks(rms1, 30, 5)
        L = np.clip((rms1 - base) / 12.0, 0.0, 1.0)
        G = ((game_sfx >= 0.4) | ((above1 >= 15.0) & ~near_words_2 & (laugh < 0.2))).astype(np.float32)
        SIL = ((speech1 < 0.05) & (above1 <= 6.0) & ~has_words).astype(np.float32)
    else:
        L = np.zeros(n, dtype=np.float32)
        G = np.zeros(n, dtype=np.float32)
        SIL = (~near_words_1).astype(np.float32)

    S = np.clip(_box(words_per_s, 5) / 5.0 / 3.0, 0.0, 1.0)

    setup_per_s = np.zeros(n, dtype=np.float32)
    for w in words:
        i = int((w["s"] + w["e"]) / 2.0)
        if 0 <= i < n and is_setup_word(w.get("w", "")):
            setup_per_s[i] += 1.0
    setup_words = _window_sum_before(setup_per_s, SETUP_LOOKBACK_S, 1)

    h = chat_health(chat, dur)
    C = np.zeros(n, dtype=np.float32)
    chat_q = np.zeros(n, dtype=np.float32)
    if chat and h > 0:
        counts = np.zeros(n, dtype=np.float32)
        what = np.zeros(n, dtype=np.float32)
        for msg in chat:
            t = coerce_number(msg.get("t") if isinstance(msg, dict) else None, None)
            if t is not None and 0 <= t < n:
                counts[int(t)] += 1.0
                what[int(t)] += float(chat_what_cues(msg.get("m", "")))
        chat_q = np.clip(_box(what, 10) / 3.0, 0.0, 1.0) * float(clamp(0.5 + h, 0.0, 1.0))
        c10 = _box(counts, 10)
        base_c = _rolling_median_blocks(c10, 60, 5)
        C = np.clip((c10 - base_c) / (base_c + 3.0), 0.0, 1.0)

    x_raw = np.zeros(n, dtype=np.float32)
    story = np.zeros(n, dtype=np.float32)
    mundane = np.zeros(n, dtype=np.float32)
    apology_raw = np.zeros(n, dtype=np.float32)
    chaos_raw = np.zeros(n, dtype=np.float32)
    for ph in phrases or []:
        if ph.get("kind") != "speech":
            continue
        cues = text_cues(ph.get("text", ""))
        i = int(clamp((ph["s"] + ph["e"]) / 2.0, 0, n - 1))
        x_raw[i] += cues["excite"]
        story[i] += cues["story"]
        mundane[i] += cues["mundane"]
        apology_raw[i] += cues["apology"]
        chaos_raw[i] += cues["chaos"]
    X = np.clip(_box(x_raw, 7) / 3.0, 0.0, 1.0)
    T = np.clip(_box(story, 41) / 2.0, 0.0, 1.0)
    M = np.clip(_box(mundane, 41) / 2.0, 0.0, 1.0)
    apology = np.clip(_box(apology_raw, 5), 0.0, 1.0)
    chaos = np.clip(_box(chaos_raw, 5), 0.0, 1.0)
    A = np.maximum(apology, chaos)

    # Visual suspicion: a reaction with little verbal setup before it, boosted by apology or
    # chaos words within CUE_REACH_S and by chat asking what happened in the next CHAT_LAG_S.
    base = R * np.clip(1.0 - setup_words / SETUP_FULL_WORDS, 0.0, 1.0)
    a_near = _max_filter(A, CUE_REACH_S, CUE_REACH_S)
    q_near = _max_filter(chat_q, 0, CHAT_LAG_S)
    reacting = (R >= RADAR_REACTION_MIN).astype(np.float32)
    V = np.clip(base + reacting * (0.30 * a_near + 0.20 * q_near), 0.0, 1.0)
    V = np.maximum(V, 0.35 * apology).astype(np.float32)

    return Signals(
        duration=dur, R=R, L=L, S=S, C=C, X=X, x_raw=x_raw, T=T, M=M, G=G, SIL=SIL,
        words_per_s=words_per_s, laugh=laugh, chat_h=h, words=words,
        word_starts=[w["s"] for w in words], feats=feats,
        V=V, A=A.astype(np.float32), apology=apology.astype(np.float32), chaos=chaos.astype(np.float32),
        chat_q=chat_q.astype(np.float32), setup_words=setup_words,
    )


# ============================================================================
# Window scoring
# ============================================================================

def _p90(arr: np.ndarray) -> float:
    return float(np.percentile(arr, 90)) if len(arr) else 0.0


def score_windows(signals: Signals, profile: Optional[dict] = None,
                  window_s: float = WINDOW_S, step_s: float = STEP_S) -> dict:
    """45 s windows every 5 s. Returns windows (t0, t1, score, norm), the hot windows after
    non maximum suppression (about 8 per hour) and the phrase ids to mark with 🔥 around
    each hot peak (filled by mark_hot_phrases)."""
    lw = dict(DEFAULT_LOCAL_WEIGHTS)
    if profile and isinstance(profile.get("local_weights"), dict):
        lw.update({k: float(v) for k, v in profile["local_weights"].items() if k in lw})
    n = signals.n
    win = max(1, int(window_s))
    step = max(1, int(step_s))
    h = signals.chat_h
    windows = []
    for t0 in range(0, max(1, n - win + 1), step):
        a, b = t0, min(n, t0 + win)
        R = signals.R[a:b]
        L = signals.L[a:b]
        S = signals.S[a:b]
        X = float(signals.X[a:b].max()) if b > a else 0.0
        T = float(signals.T[a:b].mean()) if b > a else 0.0
        M = float(signals.M[a:b].mean()) if b > a else 0.0
        quiet_story = T * float(S.mean()) * (1.0 - float(L.mean()))
        score = (
            lw["w_r"] * _p90(R)
            + lw["w_l"] * _p90(L)
            + lw["w_s"] * float(S.mean())
            + lw["w_c"] * h * _p90(signals.C[a:b])
            + lw["w_x"] * X
            + lw["w_t"] * T
            + lw["w_q"] * quiet_story
            + lw["w_v"] * (_p90(signals.V[a:b]) if len(signals.V) >= b else 0.0)
            - lw["w_g"] * float(signals.G[a:b].mean())
            - lw["w_m"] * M
            - lw["w_sil"] * float(signals.SIL[a:b].mean())
        )
        windows.append({"t0": float(a), "t1": float(b), "score": round(float(score), 4)})
    if not windows:
        return {"windows": [], "hot": [], "hot_phrase_ids": [], "p99": 0.0}

    scores = np.array([w["score"] for w in windows], dtype=np.float32)
    lo = float(np.percentile(scores, 5))
    hi = float(np.percentile(scores, 99))
    span = max(1e-6, hi - lo)
    for w in windows:
        w["norm"] = round(float(clamp((w["score"] - lo) / span, 0.0, 1.0)), 4)

    limit = max(1, int(math.ceil(signals.duration / 3600.0 * HOT_PER_HOUR)))
    hot = []
    for w in sorted(windows, key=lambda x: -x["score"]):
        if w["norm"] < 0.35:
            break
        if any(abs(w["t0"] - hw["t0"]) < window_s for hw in hot):
            continue
        a, b = int(w["t0"]), int(w["t1"])
        composite = 0.5 * signals.R[a:b] + 0.2 * signals.L[a:b] + 0.3 * signals.X[a:b]
        peak = a + int(np.argmax(composite)) if b > a else a
        hot.append({**w, "peak": float(peak)})
        if len(hot) >= limit:
            break
    hot.sort(key=lambda x: x["t0"])
    return {"windows": windows, "hot": hot, "hot_phrase_ids": [], "p99": round(hi, 4)}


def mark_hot_phrases(scores: dict, phrases: Sequence[dict], radius_s: float = 4.0) -> list:
    """Phrase ids within radius_s of each hot peak (these get a 🔥 in prompts)."""
    ids = set()
    for hw in scores.get("hot", []):
        peak = float(hw.get("peak", (hw["t0"] + hw["t1"]) / 2.0))
        for ph in phrases:
            if ph["e"] >= peak - radius_s and ph["s"] <= peak + radius_s:
                ids.add(int(ph["id"]))
    out = sorted(ids)
    scores["hot_phrase_ids"] = out
    return out


def _radar_why(signals: Signals, t: int) -> str:
    parts = []
    setup = float(signals.setup_words[t]) if len(signals.setup_words) > t else 0.0
    laugh = float(signals.laugh[t]) if len(signals.laugh) > t else 0.0
    react = float(signals.R[t]) if len(signals.R) > t else 0.0
    if react >= RADAR_REACTION_MIN:
        what = "ضحك" if laugh >= max(0.3, react * 0.8) else "صراخ"
        if setup <= 2:
            parts.append(f"{what} بدون كلام قبله")
        elif setup <= 6:
            parts.append(f"{what} بعد كلام قليل")
        else:
            parts.append(what)
    lo, hi = max(0, t - CUE_REACH_S), t + CUE_REACH_S + 1
    if len(signals.apology) and float(signals.apology[lo:hi].max()) > 0:
        parts.append("اعتذار")
    if len(signals.chaos) and float(signals.chaos[lo:hi].max()) > 0:
        parts.append("فوضى")
    if len(signals.chat_q) and float(signals.chat_q[t:t + CHAT_LAG_S + 1].max()) >= 0.3:
        parts.append("الشات يسأل وش صار")
    return " + ".join(parts) or "لحظة مشبوهة بصرياً"


def radar_peaks(
    signals: Signals,
    per_hour: float = 3.0,
    min_score: float = 0.45,
    min_gap_s: float = 60.0,
    exclude: Sequence[Sequence[float]] = (),
    active_spans: Optional[Sequence[Sequence[float]]] = None,
) -> list:
    """Reaction Radar (no Gemini request): the strongest moments where a reaction came with
    almost no verbal setup, so something probably happened on screen.
      per_hour     peaks allowed per active hour (at least 1 when any peak qualifies)
      min_score    minimum V
      min_gap_s    minimum distance between two peaks
      exclude      spans already covered by Scout candidates (peaks inside are skipped)
      active_spans only peaks inside these spans count (AFK and dead time removed)
    Returns [{"t": seconds, "score": 0..1, "why": Arabic reason}] sorted by time."""
    V = signals.V
    n = len(V)
    if n == 0:
        return []
    mask = np.ones(n, dtype=bool)
    if active_spans is not None:
        mask[:] = False
        for s, e in active_spans:
            a, b = int(clamp(math.floor(float(s)), 0, n)), int(clamp(math.ceil(float(e)), 0, n))
            mask[a:b] = True
        active_s = spans_total(active_spans)
    else:
        active_s = signals.duration
    for s, e in exclude or ():
        a, b = int(clamp(math.floor(float(s)), 0, n)), int(clamp(math.ceil(float(e)), 0, n))
        mask[a:b] = False
    scored = np.where(mask, V, 0.0)
    budget = max(1, int(math.ceil(max(0.0, float(active_s)) / 3600.0 * float(per_hour))))
    order = np.argsort(-scored, kind="stable")
    peaks: list = []
    for idx in order:
        val = float(scored[idx])
        if val < min_score:
            break
        t = int(idx)
        if any(abs(t - p["t"]) < min_gap_s for p in peaks):
            continue
        peaks.append({"t": float(t), "score": round(val, 3), "why": _radar_why(signals, t)})
        if len(peaks) >= budget:
            break
    peaks.sort(key=lambda p: p["t"])
    return peaks


def mark_radar_phrases(peaks: Sequence[dict], phrases: Sequence[dict], radius_s: float = 6.0) -> list:
    """Phrase ids near each radar peak (these get a 👀 in Scout prompts). The window reaches
    further back than forward because the visual setup comes before the reaction."""
    ids = set()
    for pk in peaks or []:
        t = float(pk.get("t", 0.0))
        for ph in phrases:
            if ph["e"] >= t - radius_s * 1.5 and ph["s"] <= t + radius_s:
                ids.add(int(ph["id"]))
    return sorted(ids)


def local_score_for_span(scores: dict, s: float, e: float) -> float:
    """Best normalized window score overlapping [s, e] (0 when nothing overlaps)."""
    best = 0.0
    for w in scores.get("windows", []):
        if w["t1"] > s and w["t0"] < e:
            best = max(best, float(w.get("norm", 0.0)))
    return round(best, 4)


# ============================================================================
# Clip features and the taste model
# ============================================================================

def _seconds_in(pieces: Sequence[Sequence[float]], signals: Signals) -> np.ndarray:
    idx = []
    for s, e in pieces:
        a, b = signals.idx_range(float(s), float(e))
        idx.extend(range(a, b))
    return np.array(sorted(set(idx)), dtype=np.int64)


def _category_onehot(category: Optional[str]) -> dict:
    out = {key: 0.0 for key in CATEGORY_KEYS.values()}
    if category:
        key = CATEGORY_KEYS.get(str(category).strip())
        if key:
            out[key] = 1.0
    return out


def _critic_mean(critic: Optional[dict]) -> float:
    if not isinstance(critic, dict):
        return 0.5
    vals = [coerce_number(critic.get(k), None) for k in ("hook", "payoff", "reaction", "context", "pacing")]
    vals = [v for v in vals if v is not None]
    if not vals:
        return 0.5
    return float(clamp(sum(vals) / len(vals) / 10.0, 0.0, 1.0))


def _critic_visual(critic: Optional[dict]) -> float:
    if not isinstance(critic, dict):
        return 0.5
    val = coerce_number(critic.get("visual"), None)
    if val is None:
        return 0.5
    return float(clamp(val / 10.0, 0.0, 1.0))


def compute_clip_features(
    pieces: Sequence[Sequence[float]],
    signals: Signals,
    critic: Optional[dict] = None,
    category: Optional[str] = None,
) -> dict:
    """Quantitative features of a final clip (all roughly 0..1, duration_z and n_pieces_z centered)."""
    pcs = merge_pieces(pieces, 0.0)
    feats_out = {name: 0.0 for name in FEATURE_NAMES}
    if not pcs:
        return feats_out
    net = pieces_net(pcs)
    span = pcs[-1][1] - pcs[0][0]
    secs = _seconds_in(pcs, signals)
    feats = signals.feats

    def take(arr: np.ndarray) -> np.ndarray:
        return arr[secs] if len(secs) else np.zeros(0, dtype=np.float32)

    R = take(signals.R)
    n_words = 0
    for w in signals.words:
        mid = (w["s"] + w["e"]) / 2.0
        if any(s <= mid < e for s, e in pcs):
            n_words += 1

    last_s, last_e = pcs[-1]
    a, b = signals.idx_range(max(last_s, last_e - 1.5), last_e)
    ends_on_reaction = 1.0 if (b > a and float(signals.R[a:b].max()) >= 0.4) else 0.0

    first_s = pcs[0][0]
    starts_with_speech = 0.0
    j = next((k for k, w in enumerate(signals.words) if w["s"] >= first_s - 0.05), None)
    if j is not None and signals.words[j]["s"] <= first_s + 1.5:
        if feats is None or float(feats.above_floor(first_s, first_s + 1.5).max()) >= 6.0:
            starts_with_speech = 1.0

    if feats is not None:
        quiet_frames = 0
        total_frames = 0
        for s, e in pcs:
            ia, ib = feats.range10(s, e)
            above = feats.rms_db[ia:ib].astype(np.float32) - feats.floor_db[ia:ib].astype(np.float32)
            sp = feats.speech[ia:ib]
            quiet_frames += int(((sp == 0) & (above <= 6.0)).sum())
            total_frames += max(0, ib - ia)
        silence_frac = quiet_frames / total_frames if total_frames else 0.0
    else:
        silence_frac = float(take(signals.SIL).mean()) if len(secs) else 0.0

    laugh_secs = float((take(signals.laugh) >= 0.35).sum())
    excite_total = float(take(signals.x_raw).sum())

    feats_out.update({
        "duration_z": round((net - 22.0) / 10.0, 4),
        "removed_ratio": round(float(clamp(1.0 - net / span, 0.0, 1.0)) if span > 0 else 0.0, 4),
        "n_pieces_z": round((len(pcs) - 2) / 2.0, 4),
        "reaction_peak": round(float(R.max()) if len(R) else 0.0, 4),
        "laugh_s": round(float(clamp(laugh_secs / 6.0, 0.0, 1.0)), 4),
        "game_only_frac": round(float(take(signals.G).mean()) if len(secs) else 0.0, 4),
        "loud_p90": round(_p90(take(signals.L)), 4),
        "speech_wps": round(float(clamp(n_words / max(net, 1e-6) / 3.0, 0.0, 1.0)), 4),
        "chat_peak": round(float(take(signals.C).max() * signals.chat_h) if len(secs) else 0.0, 4),
        "excite_cue": round(float(clamp(excite_total / max(1.0, net / 10.0) / 2.0, 0.0, 1.0)), 4),
        "story_cue": round(float(take(signals.T).mean()) if len(secs) else 0.0, 4),
        "mundane_cue": round(float(take(signals.M).mean()) if len(secs) else 0.0, 4),
        "ends_on_reaction": ends_on_reaction,
        "starts_with_speech": starts_with_speech,
        "silence_frac": round(float(clamp(silence_frac, 0.0, 1.0)), 4),
        "critic_mean": round(_critic_mean(critic), 4),
        "visual_score": round(_critic_visual(critic), 4),
        "visual_suspect": round(float(take(signals.V).max())
                                if len(secs) and len(signals.V) == signals.n else 0.0, 4),
    })
    feats_out.update(_category_onehot(category))
    return feats_out


def _vector(features: dict) -> np.ndarray:
    return np.array([float(coerce_number(features.get(name), 0.0) or 0.0) for name in FEATURE_NAMES], dtype=np.float64)


def predict_taste(profile: dict, features: dict) -> float:
    """Probability (0..1) that the editor likes a clip with these features."""
    w = np.array([float(profile["weights"].get(n, DEFAULT_PRIOR.get(n, 0.0))) for n in FEATURE_NAMES])
    z = float(profile.get("bias", DEFAULT_BIAS)) + float(np.dot(w, _vector(features)))
    return round(1.0 / (1.0 + math.exp(-clamp(z, -30.0, 30.0))), 4)


_NOTE_SCORE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[\\/]\s*(\d+(?:\.\d+)?)")


def parse_note_score(note: Any) -> Optional[float]:
    """Read a 0..10 score the editor typed in a note: "6\\10", "6/10", "10\\5" (= 5 of 10)."""
    text = normalize_digits(str(note or ""))
    for m in _NOTE_SCORE_RE.finditer(text):
        a, b = float(m.group(1)), float(m.group(2))
        if b == 10 and 0 <= a <= 10:
            return a
        if a == 10 and 0 <= b < 10:
            return b
    return None


def _label(entry: dict) -> Optional[float]:
    score = coerce_number(entry.get("score10"), None)
    if score is None:
        score = parse_note_score(entry.get("note"))
    rating = entry.get("rating")
    if score is not None:
        y = clamp(score / 10.0, 0.0, 1.0)
        if rating == "liked":
            return max(y, 0.5)
        if rating == "disliked":
            return min(y, 0.5)
        return y
    if rating == "liked":
        return 1.0
    if rating == "disliked":
        return 0.0
    return None


def _fit_logistic(X: np.ndarray, y: np.ndarray, sw: np.ndarray, prior_w: np.ndarray, prior_b: float,
                  lam: float, iters: int = 400, lr: float = 0.3) -> tuple:
    """Weighted logistic regression anchored to prior weights (L2 toward the prior)."""
    w = prior_w.copy()
    b = float(prior_b)
    total = float(sw.sum()) or 1.0
    for _ in range(iters):
        z = np.clip(X @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = (p - y) * sw
        grad_w = X.T @ err / total + lam * (w - prior_w) / total
        grad_b = float(err.sum()) / total + 0.25 * lam * (b - prior_b) / total
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def train_profile(feedback_entries: Sequence[dict], prev_profile: Optional[dict] = None,
                  lam: float = DEFAULT_LAMBDA) -> dict:
    """Retrain the taste profile from rated feedback (instant, local, no Gemini call).
    Updates weights, local window weights, length and padding preferences, and tag counts."""
    prof = default_profile()
    if prev_profile:
        prof["prior"].update(prev_profile.get("prior") or {})
        prof["lambda"] = float(prev_profile.get("lambda", lam))
    lam = float(prof["lambda"])
    prior_w = np.array([float(prof["prior"].get(n, 0.0)) for n in FEATURE_NAMES])

    rows, ys, weights = [], [], []
    liked_durations = []
    end_deltas, start_deltas = [], []
    tag_counts: dict = {}
    for e in feedback_entries or []:
        if not isinstance(e, dict):
            continue
        y = _label(e)
        feats = e.get("features") or {}
        if y is not None and feats:
            rows.append(_vector(feats))
            ys.append(y)
            weights.append(0.5 if e.get("features_source") == "coarse" else 1.0)
        if e.get("rating") == "liked":
            dur = coerce_number(e.get("duration"), None)
            if dur:
                liked_durations.append(float(dur))
        if e.get("rating") == "disliked":
            for tag in e.get("tags") or []:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
            note = str(e.get("note") or "")
            for part in note.split("|"):
                part = part.strip()
                if (3 <= len(part) <= 40 and not any(ch.isdigit() for ch in part)
                        and not any(hint in part for hint in _POSITIVE_HINTS)):
                    tag_counts[part] = tag_counts.get(part, 0) + 1
        for trim in e.get("manual_trims") or []:
            ed = coerce_number(trim.get("end_delta"), None)
            sd = coerce_number(trim.get("start_delta"), None)
            if ed is not None and ed != 0:
                end_deltas.append(float(ed))
            if sd is not None and sd != 0:
                start_deltas.append(float(sd))

    if rows:
        X = np.vstack(rows)
        w, b = _fit_logistic(X, np.array(ys), np.array(weights), prior_w, DEFAULT_BIAS, lam)
        prof["weights"] = {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, w)}
        prof["bias"] = round(float(b), 4)
        prof["mean"] = {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, X.mean(axis=0))}
        prof["std"] = {n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, X.std(axis=0))}
    prof["n"] = len(rows)

    lw = dict(DEFAULT_LOCAL_WEIGHTS)
    for key, (feat, sign) in _LOCAL_MAP.items():
        delta = (float(prof["weights"].get(feat, 0.0)) - float(prof["prior"].get(feat, 0.0))) * sign
        lw[key] = round(DEFAULT_LOCAL_WEIGHTS[key] * (1.0 + 0.5 * math.tanh(delta)), 4)
    prof["local_weights"] = lw

    prefs = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_PREFS.items()}
    if len(liked_durations) >= 5:
        lo = float(np.percentile(liked_durations, 25))
        hi = float(np.percentile(liked_durations, 75))
        prefs["target_s"] = [round(clamp(lo, 8.0, 20.0), 1), round(clamp(hi, 20.0, 45.0), 1)]
    if len(end_deltas) >= 3:
        prefs["end_pad_s"] = round(clamp(DEFAULT_PREFS["end_pad_s"] + float(np.median(end_deltas)), 0.10, 3.0), 2)
    if len(start_deltas) >= 3:
        prefs["start_pad_s"] = round(clamp(DEFAULT_PREFS["start_pad_s"] - float(np.median(start_deltas)), 0.05, 2.0), 2)
    prof["prefs"] = prefs
    prof["tag_counts"] = dict(sorted(tag_counts.items(), key=lambda kv: -kv[1]))
    prof["updated_at"] = now_iso()
    return prof


def apply_golden_priors(profile: dict, golden_stats: Optional[dict], fade_n: int = GOLDEN_FADE_N) -> dict:
    """Blend Golden Clips statistics into a COPY of the taste profile for this run.

    alpha = 1 - rated_clips / fade_n (0 when at least fade_n clips are rated or fewer than 3
    golden clips exist). With alpha > 0 the copy gets:
      * target duration pulled toward the golden p25..p75
      * max removed ratio pulled toward what the jump cut density of the edits implies
      * ends_on_reaction weight moved by the share of golden edits ending on laughter or reaction
      * category weights moved by how often each moment type appears in the edits
      * visual weights raised by the share of visual moments (fails, betrayals, pushes)
      * starts_with_speech raised when the edits hook in the first second
    The trained values are kept in copy["golden"]["base"] so save_profile never stores the blend.
    golden_stats keys: n, duration_s [p25, p50, p75], hook_latency_s, ending_types {},
    cuts_per_min, moment_types {}."""
    prof = copy.deepcopy(profile or default_profile())
    old = prof.get("golden")
    if isinstance(old, dict) and isinstance(old.get("base"), dict):
        for key, value in old["base"].items():
            prof[key] = copy.deepcopy(value)
    stats = golden_stats if isinstance(golden_stats, dict) else {}
    n_gold = int(coerce_number(stats.get("n"), 0) or 0)
    rated = int(coerce_number(prof.get("n"), 0) or 0)
    fade = max(1, int(fade_n or GOLDEN_FADE_N))
    alpha = float(clamp(1.0 - rated / float(fade), 0.0, 1.0)) if n_gold >= 3 else 0.0
    info: dict = {"alpha": round(alpha, 3), "n_golden": n_gold, "rated": rated, "fade_n": fade, "changes": []}
    if alpha <= 0.0:
        prof["golden"] = info
        return prof
    info["base"] = {
        "weights": copy.deepcopy(prof.get("weights") or {}),
        "prefs": copy.deepcopy(prof.get("prefs") or {}),
        "local_weights": copy.deepcopy(prof.get("local_weights") or {}),
    }
    prefs = prof.setdefault("prefs", {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULT_PREFS.items()})
    weights = prof.setdefault("weights", dict(DEFAULT_PRIOR))
    lw = prof.setdefault("local_weights", dict(DEFAULT_LOCAL_WEIGHTS))

    def bump(name: str, delta: float) -> None:
        if abs(delta) < 1e-4:
            return
        weights[name] = round(float(weights.get(name, DEFAULT_PRIOR.get(name, 0.0))) + float(delta), 4)

    dur = stats.get("duration_s")
    if isinstance(dur, (list, tuple)) and len(dur) >= 3:
        p25, p75 = coerce_number(dur[0], None), coerce_number(dur[2], None)
        if p25 and p75 and p75 > p25:
            cur = prefs.get("target_s") or DEFAULT_PREFS["target_s"]
            g_lo, g_hi = clamp(float(p25), 8.0, 20.0), clamp(float(p75), 20.0, 45.0)
            new = [round((1 - alpha) * float(cur[0]) + alpha * g_lo, 1),
                   round((1 - alpha) * float(cur[1]) + alpha * g_hi, 1)]
            prefs["target_s"] = new
            info["changes"].append(f"المدة المفضلة صارت من {new[0]:.0f} إلى {new[1]:.0f} ثانية")

    cpm = coerce_number(stats.get("cuts_per_min"), None)
    if cpm is not None and cpm >= 0:
        g_ratio = clamp(0.30 + 0.04 * float(cpm), 0.30, 0.60)
        cur_ratio = float(prefs.get("max_removed_ratio", DEFAULT_PREFS["max_removed_ratio"]))
        prefs["max_removed_ratio"] = round((1 - alpha) * cur_ratio + alpha * g_ratio, 3)

    endings = stats.get("ending_types") if isinstance(stats.get("ending_types"), dict) else {}
    total_end = sum(float(coerce_number(v, 0) or 0) for v in endings.values())
    if total_end > 0:
        share = sum(float(coerce_number(endings.get(k), 0) or 0) for k in _REACTION_ENDINGS) / total_end
        bump("ends_on_reaction", alpha * 0.8 * (share - 0.5))
        if share >= 0.5:
            info["changes"].append(f"{share * 100:.0f}% من لقطاتك الذهبية تقفل على ضحك أو ردة فعل")

    moments = stats.get("moment_types") if isinstance(stats.get("moment_types"), dict) else {}
    total_m = sum(float(coerce_number(v, 0) or 0) for v in moments.values())
    if total_m > 0:
        cat_share = {key: 0.0 for key in CATEGORY_KEYS.values()}
        for mtype, count in moments.items():
            key = _MOMENT_TO_CAT.get(str(mtype))
            if key:
                cat_share[key] += float(coerce_number(count, 0) or 0) / total_m
        even = 1.0 / len(cat_share)
        for key, share in cat_share.items():
            bump(key, alpha * 0.6 * (share - even))
        visual_share = sum(float(coerce_number(moments.get(k), 0) or 0) for k in _VISUAL_MOMENTS) / total_m
        if visual_share > 0:
            bump("visual_score", alpha * 0.6 * visual_share)
            bump("visual_suspect", alpha * 0.4 * visual_share)
            lw["w_v"] = round(float(lw.get("w_v", DEFAULT_LOCAL_WEIGHTS["w_v"])) * (1.0 + 0.5 * alpha * visual_share), 4)
            info["changes"].append(f"{visual_share * 100:.0f}% من لقطاتك الذهبية لحظات بصرية (طيحة، دفة، غدر)")

    hook = coerce_number(stats.get("hook_latency_s"), None)
    if hook is not None and hook >= 0:
        bump("starts_with_speech", alpha * 0.3 * (1.0 - min(1.0, float(hook) / 2.0)))

    prof["golden"] = info
    return prof


def doctor_prefs(profile: dict) -> dict:
    """Subset of taste prefs that media_engine.cut_doctor understands."""
    prefs = profile.get("prefs") or {}
    target = prefs.get("target_s") or DEFAULT_PREFS["target_s"]
    return {
        "end_pad_s": float(prefs.get("end_pad_s", DEFAULT_PREFS["end_pad_s"])),
        "start_pad_s": float(prefs.get("start_pad_s", DEFAULT_PREFS["start_pad_s"])),
        "target_s": (float(target[0]), float(target[1])),
    }


# ============================================================================
# Prompt context (Arabic)
# ============================================================================

def _short(text: Any, limit: int) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    return t if len(t) <= limit else t[: max(0, limit - 1)].rstrip() + "…"


def _pick_examples(entries: list, rating: str, limit: int) -> list:
    pool = [e for e in entries if e.get("rating") == rating]
    pool.sort(key=lambda e: str(e.get("updated_at") or ""), reverse=True)
    chosen, seen_cats = [], set()
    for e in pool:
        cat = e.get("category") or ""
        if cat not in seen_cats:
            chosen.append(e)
            seen_cats.add(cat)
        if len(chosen) >= limit:
            return chosen
    for e in pool:
        if e not in chosen:
            chosen.append(e)
        if len(chosen) >= limit:
            break
    return chosen


def _example_line(e: dict, mark: str) -> str:
    dur = coerce_number(e.get("duration"), None)
    dur_txt = f"، {dur:.0f}ث" if dur else ""
    score = coerce_number(e.get("score10"), None)
    if score is None:
        score = parse_note_score(e.get("note"))
    score_txt = f"، تقييمه {score:g} من 10" if score is not None else ""
    parts = [f"{mark} «{_short(e.get('title'), 50)}» ({e.get('category') or 'منوع'}{dur_txt}{score_txt})"]
    excerpt = _short(e.get("excerpt"), 160)
    if excerpt:
        parts.append(f"   الكلام: {excerpt}")
    note = _short(e.get("note"), 220)
    tags = [t for t in e.get("tags") or [] if t and t not in note]
    if tags:
        note = (" | ".join(tags) + (" | " + note if note else "")).strip()
    if note:
        parts.append(f"   رأي الممنتج: {note}")
    return "\n".join(parts)


def prompt_context(profile: dict, rules_text: str, feedback_entries: Sequence[dict],
                   max_liked: int = 4, max_disliked: int = 6, rules_limit: int = 3500,
                   golden_text: str = "") -> str:
    """Arabic block for scout and critic prompts: the editor's rules, the Golden DNA from his
    finished edits (golden_text, already built by golden_engine), quantitative preferences,
    most common rejection reasons, strongest learned signals, and examples."""
    entries = [e for e in feedback_entries or [] if isinstance(e, dict)]
    lines = []
    rules = str(rules_text or "").strip()
    if rules:
        lines.append("📜 قواعد الممنتج (التزم فيها حرفياً):")
        lines.append(rules[:rules_limit])
    golden = str(golden_text or "").strip()
    if golden:
        if lines:
            lines.append("")
        lines.append(golden)

    prefs = profile.get("prefs") or DEFAULT_PREFS
    target = prefs.get("target_s") or DEFAULT_PREFS["target_s"]
    lines.append("")
    lines.append("📏 تفضيلات رقمية تعلمناها من تقييماته:")
    lines.append(f"   • المدة الصافية المفضلة من {float(target[0]):.0f} إلى {float(target[1]):.0f} ثانية")
    lines.append(f"   • نسبة المحذوف بالقص الذكي لا تتجاوز {float(prefs.get('max_removed_ratio', 0.45)) * 100:.0f}% من مدى اللقطة")
    lines.append("   • النهاية لازم تكون بعد ما تخلص الضحكة أو ردة الفعل، مو أثناءها")
    lines.append("   • لو الضحكة على شي صار في الشاشة، اللقطة تبدأ قبل الحركة نفسها (الدفة أو الطيحة) مو بعدها")

    rated = [e for e in entries if e.get("rating") in ("liked", "disliked")]
    disliked = [e for e in rated if e.get("rating") == "disliked"]
    tag_counts = profile.get("tag_counts") or {}
    if tag_counts and disliked:
        lines.append("")
        lines.append(f"🚫 أكثر أسباب الرفض ({len(disliked)} لقطة مرفوضة من {len(rated)}):")
        for tag, count in list(tag_counts.items())[:5]:
            lines.append(f"   • {tag}: {count} مرة")

    weights = profile.get("weights") or {}
    prior = profile.get("prior") or DEFAULT_PRIOR
    if profile.get("n", 0) >= 5:
        shifts = sorted(
            ((n, float(weights.get(n, 0.0)) - float(prior.get(n, 0.0))) for n in FEATURE_NAMES if not n.startswith("cat_")),
            key=lambda kv: -abs(kv[1]),
        )
        strong = [(n, d) for n, d in shifts if abs(d) >= 0.05][:4]
        if strong:
            lines.append("")
            lines.append("📈 أقوى اللي تغير في ذوقه حسب التقييمات:")
            for name, delta in strong:
                word = "يحب أكثر" if delta > 0 else "يكره أكثر"
                lines.append(f"   • {word}: {FEATURE_LABELS_AR.get(name, name)}")

    liked_ex = _pick_examples(entries, "liked", max_liked)
    disliked_ex = _pick_examples(entries, "disliked", max_disliked)
    if liked_ex:
        lines.append("")
        lines.append("✅ أمثلة لقطات عجبته:")
        lines.extend(_example_line(e, "✅") for e in liked_ex)
    if disliked_ex:
        lines.append("")
        lines.append("❌ أمثلة لقطات رفضها (لا تكرر نفس الغلط):")
        lines.extend(_example_line(e, "❌") for e in disliked_ex)
    return "\n".join(lines).strip()


def taste_summary_ar(profile: dict) -> str:
    """Short Arabic summary for the Studio taste tab."""
    prefs = profile.get("prefs") or DEFAULT_PREFS
    target = prefs.get("target_s") or DEFAULT_PREFS["target_s"]
    lines = [
        f"🧠 الذوق الرقمي متدرب على {profile.get('n', 0)} لقطة مقيّمة (آخر تحديث: {profile.get('updated_at') or 'لم يتدرب بعد'})",
        f"📏 المدة المفضلة: من {float(target[0]):.0f} إلى {float(target[1]):.0f} ثانية",
        f"✂️ مسافة النهاية بعد آخر كلمة: {float(prefs.get('end_pad_s', 0.3)):.2f} ثانية | قبل أول كلمة: {float(prefs.get('start_pad_s', 0.12)):.2f} ثانية",
    ]
    weights = profile.get("weights") or {}
    ranked = sorted(((n, float(weights.get(n, 0.0))) for n in FEATURE_NAMES if not n.startswith("cat_")),
                    key=lambda kv: -kv[1])
    pos = [FEATURE_LABELS_AR[n] for n, v in ranked[:3] if v > 0]
    neg = [FEATURE_LABELS_AR[n] for n, v in ranked[::-1][:3] if v < 0]
    if pos:
        lines.append("👍 أكثر شي يرفع اللقطة: " + "، ".join(pos))
    if neg:
        lines.append("👎 أكثر شي ينزلها: " + "، ".join(neg))
    golden = profile.get("golden") if isinstance(profile.get("golden"), dict) else {}
    if golden.get("n_golden"):
        alpha = float(golden.get("alpha") or 0.0)
        if alpha > 0:
            lines.append(f"🌟 اللقطات الذهبية ({golden['n_golden']} مقطع) مأثرة بنسبة {alpha * 100:.0f}% "
                         f"وتخف لين توصل {golden.get('fade_n', GOLDEN_FADE_N)} لقطة مقيّمة")
            for change in golden.get("changes") or []:
                lines.append(f"   • {change}")
        else:
            lines.append(f"🌟 تقييماتك صارت كافية، اللقطات الذهبية ({golden['n_golden']} مقطع) تأثر بالنص فقط")
    return "\n".join(lines)


# ============================================================================
# Legacy feedback backfill
# ============================================================================

def _coarse_features(entry: dict) -> dict:
    """Approximate features from the old metrics block when the clip media is gone."""
    m = entry.get("metrics") or {}
    dur = float(coerce_number(entry.get("duration"), 20.0) or 20.0)
    removed = float(coerce_number(entry.get("removed_sec"), 0.0) or 0.0)
    max_audio = float(coerce_number(m.get("max_audio"), 0.4) or 0.4)
    avg_audio = float(coerce_number(m.get("avg_audio"), 0.35) or 0.35)
    wps = float(coerce_number(entry.get("avg_wps"), 2.0) or 2.0)
    cues = text_cues(entry.get("excerpt") or "")
    excerpt_norm = normalize_arabic(entry.get("excerpt") or "")
    laugh_tokens = sum(1 for t in excerpt_norm.split() if _LAUGH_TOKEN in t)
    out = {name: 0.0 for name in FEATURE_NAMES}
    out.update({
        "duration_z": round((dur - 22.0) / 10.0, 4),
        "removed_ratio": round(clamp(removed / (dur + removed), 0.0, 1.0) if dur + removed > 0 else 0.0, 4),
        "n_pieces_z": 0.0,
        "reaction_peak": round(clamp((max_audio - 0.4) / 0.8, 0.0, 1.0), 4),
        "laugh_s": round(clamp(laugh_tokens / 3.0, 0.0, 1.0), 4),
        "game_only_frac": 0.0,
        "loud_p90": round(clamp(avg_audio, 0.0, 1.0), 4),
        "speech_wps": round(clamp(wps / 3.0, 0.0, 1.0), 4),
        "chat_peak": 0.0,
        "excite_cue": round(clamp(cues["excite"] / max(1.0, dur / 10.0) / 2.0, 0.0, 1.0), 4),
        "story_cue": round(clamp(cues["story"] / 2.0, 0.0, 1.0), 4),
        "mundane_cue": round(clamp(cues["mundane"] / 2.0, 0.0, 1.0), 4),
        "ends_on_reaction": 0.5,
        "starts_with_speech": 0.5,
        "silence_frac": 0.2,
        "critic_mean": 0.5,
    })
    out.update(_category_onehot(entry.get("category")))
    return out


def _legacy_key(entry: dict) -> str:
    key = str(entry.get("key") or "").strip()
    if key:
        return key
    uid = str(entry.get("uid") or "")
    return uid[len("legacy:"):] if uid.startswith("legacy:") else ""


def backfill_legacy(feedback_entries: Sequence[dict], drive_root: str,
                    allow_build_features: bool = False) -> tuple:
    """Give features to feedback entries that lack them.
    Exact path: the clip still exists in its VOD manifest (matched by legacy key, then by
    clip number plus title), so features are computed from its real segments.
    Coarse path: features approximated from the stored metrics and excerpt.
    Returns (new_entries, filled_count). Entries that already have features are untouched."""
    entries = [dict(e) for e in feedback_entries or [] if isinstance(e, dict)]
    cache: dict = {}
    filled = 0
    for e in entries:
        if e.get("features"):
            continue
        vod_id = str(e.get("vod_id") or "").strip()
        exact = None
        if vod_id:
            if vod_id not in cache:
                cache[vod_id] = _load_vod_context(drive_root, vod_id, allow_build_features)
            ctx = cache[vod_id]
            if ctx is not None:
                clip = _match_clip(ctx["manifest"], e)
                if clip is not None and clip.get("segments"):
                    try:
                        exact = compute_clip_features(clip["segments"], ctx["signals"],
                                                      critic=(clip.get("scores") or {}).get("critic"),
                                                      category=e.get("category") or clip.get("category"))
                        e["segments"] = clip["segments"]
                    except Exception as exc:
                        log(f"⚠️ تعذر حساب خصائص اللقطة {e.get('title', '')}: {short_error(exc)}")
                        exact = None
        if exact is not None:
            e["features"] = exact
            e["features_source"] = "backfill"
        else:
            e["features"] = _coarse_features(e)
            e["features_source"] = "coarse"
        filled += 1
    if filled:
        exact_n = sum(1 for e in entries if e.get("features_source") == "backfill")
        log(f"🧩 تم تجهيز خصائص {filled} تقييم قديم ({exact_n} بدقة كاملة من ملفات البث)")
    return entries, filled


def _load_vod_context(drive_root: str, vod_id: str, allow_build: bool) -> Optional[dict]:
    vp = VodPaths.from_ids(drive_root, vod_id)
    if not os.path.isdir(vp.vod_dir) or not os.path.isfile(vp.manifest):
        return None
    try:
        manifest = load_manifest(vp)
        transcript = load_json_safe(vp.transcript, default={}, expected_types=(dict,), quiet=True) or {}
        words = collect_words(transcript)
        feats = load_features(vp, words=words, allow_build=allow_build, save_built=allow_build)
        if feats is None and not words:
            return None
        chat = load_json_safe(vp.chat, default=[], expected_types=(list,), quiet=True) or []
        duration = feats.duration if feats is not None else (words[-1]["e"] if words else 0.0)
        from media_engine import build_phrases  # local import keeps module import light
        phrases = build_phrases(words, feats, vod_duration=duration)
        signals = build_signals(feats, words, phrases, chat, duration)
        return {"manifest": manifest, "signals": signals}
    except Exception as exc:
        log(f"⚠️ تعذر تحميل بيانات البث {vod_id} للتقييمات القديمة: {short_error(exc)}")
        return None


def _match_clip(manifest: dict, entry: dict) -> Optional[dict]:
    key = _legacy_key(entry)
    clips = manifest.get("clips") or []
    if key:
        for c in clips:
            if c.get("legacy_key") == key:
                return c
    cid = coerce_number(entry.get("clip_id"), None)
    title = normalize_arabic(entry.get("title") or "")
    for c in clips:
        if cid is not None and coerce_number(c.get("id"), None) == float(cid):
            if title and normalize_arabic(c.get("title") or "") == title:
                return c
    return None


# === END OF taste_engine.py ===
