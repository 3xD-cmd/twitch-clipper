import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
import numpy as np
from google import genai
from google.genai import types

SCOUT_MODELS = [
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

CRITIC_MODELS = [
    "gemini-3.8-pro",
    "gemini-pro-latest",
    "gemini-3.8-flash",
    "gemini-flash-latest",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

EXCLUDED_MODEL_KEYWORDS = ("tts", "image", "audio", "embedding", "live", "veo", "imagen", "lyria", "robotics")


def log(msg: str):
    print(msg, flush=True)


def sec_to_hms(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def sec_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def sanitize_filename(name: str, max_len: int = 40) -> str:
    cleaned = re.sub(r'[\\/*?:"<>|\r\n]+', "", name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return (cleaned[:max_len].strip("_")) or "clip"


def load_memory_context(drive_root: str) -> str:
    mem_dir = os.path.join(drive_root, "memory")
    rules_path = os.path.join(mem_dir, "my_editing_rules.md")
    history_path = os.path.join(mem_dir, "feedback_history.json")

    parts = []
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            rules = f.read().strip()
        if rules:
            parts.append(f"### قواعد ذوق الممنتج المتعلَّمة (التزم بها بدقة صارمة):\n{rules}")

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        liked = [h for h in history if h.get("rating") == "liked"][-8:]
        disliked = [h for h in history if h.get("rating") == "disliked"][-8:]
        if liked:
            lines = [
                f"- 👍 {x.get('title')} ({x.get('category', '')} | {x.get('duration', '')}ث) — ليش عجبتني: {x.get('note', 'ممتازة')}"
                for x in liked
            ]
            parts.append("### أمثلة لقطات أعجبت الممنتج مؤخراً (جب بمستواها):\n" + "\n".join(lines))
        if disliked:
            lines = [
                f"- 👎 {x.get('title')} ({x.get('category', '')} | {x.get('duration', '')}ث) — سبب الرفض: {x.get('note', 'ما تنفع')}"
                for x in disliked
            ]
            parts.append("### أمثلة لقطات رفضها الممنتج (ممنوع تكرار مثلها):\n" + "\n".join(lines))

    return "\n\n".join(parts)


def build_stream_digest(
    transcript: dict,
    hype_timeline: list,
    chat_messages: list,
    start_sec: float = 0.0,
    end_sec: float = None,
) -> str:
    hype_by_bucket = {int(item["start"] // 10): item for item in hype_timeline}

    chat_by_bucket = {}
    for msg in chat_messages:
        t = msg["t"]
        if t < start_sec or (end_sec is not None and t > end_sec):
            continue
        bucket = int(t // 10)
        chat_by_bucket.setdefault(bucket, []).append(msg["m"])

    lines = []
    last_chat_bucket = -1
    prev_end = None

    for seg in transcript.get("segments", []):
        st = seg["start"]
        en = seg["end"]
        if en < start_sec:
            continue
        if end_sec is not None and st > end_sec:
            break

        # إظهار فترات الصمت بوضوح للذكاء الاصطناعي عشان يقصّها
        if prev_end is not None and (st - prev_end) >= 1.5:
            gap = st - prev_end
            mid_bucket = int(((prev_end + st) / 2.0) // 10)
            mid_audio = hype_by_bucket.get(mid_bucket, {}).get("audio", 0)
            if mid_audio >= 0.65:
                lines.append(f"   [🔊 صوت/ضحك بدون كلام {gap:.1f}ث من {prev_end:.1f}s إلى {st:.1f}s]")
            else:
                lines.append(f"   [🔇 صمت {gap:.1f}ث من {prev_end:.1f}s إلى {st:.1f}s]")

        bucket = int(st // 10)
        h_info = hype_by_bucket.get(bucket, {})
        tags = []
        audio_lvl = h_info.get("audio", 0)
        if audio_lvl >= 0.85:
            tags.append("🔥صراخ/ضحك")
        elif audio_lvl >= 0.65:
            tags.append("⚡حماس/توتر")
        if h_info.get("chat_msgs", 0) >= 3:
            tags.append(f"💬شات({h_info['chat_msgs']})")

        tag_str = f" [{' '.join(tags)}]" if tags else ""
        lines.append(f"[{st:.1f}s - {en:.1f}s]{tag_str} {seg['text']}")

        if bucket != last_chat_bucket and bucket in chat_by_bucket and len(chat_by_bucket[bucket]) >= 2:
            sample = " | ".join(chat_by_bucket[bucket][:4])
            lines.append(f"   >> الشات: {sample}")
            last_chat_bucket = bucket

        prev_end = en

    return "\n".join(lines)


def _discover_extra_models(client, base_list: list, prefer_pro: bool = False) -> list:
    pros, flashes = [], []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            low = name.lower()
            if any(bad in low for bad in EXCLUDED_MODEL_KEYWORDS):
                continue
            if name in base_list:
                continue
            if "pro" in low and "gemini" in low:
                pros.append(name)
            elif "flash" in low and "gemini" in low:
                flashes.append(name)
    except Exception:
        pass
    ordered = (pros[:2] + flashes[:2]) if prefer_pro else (flashes[:3] + pros[:1])
    return ordered


def _request_once(
    client,
    model_name: str,
    prompt: str,
    json_mode: bool = True,
    thinking_budget: int = 0,
) -> str:
    cfg_kwargs = {"temperature": 0.3}
    if json_mode:
        cfg_kwargs["response_mime_type"] = "application/json"

    if thinking_budget and thinking_budget > 0:
        try:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)
        except Exception:
            pass

    try:
        resp = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        return (resp.text or "").strip()
    except Exception as e:
        # لو الموديل ما يدعم ThinkingConfig نعيد نفس المحاولة بدونها تلقائياً
        if "thinking" in str(e).lower() and "thinking_config" in cfg_kwargs:
            cfg_kwargs.pop("thinking_config", None)
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(**cfg_kwargs),
            )
            return (resp.text or "").strip()
        raise


def call_gemini_text(
    api_key: str,
    prompt: str,
    json_mode: bool = False,
    timeout_sec: int = 110,
    prefer_pro: bool = False,
    thinking_budget: int = 0,
) -> str:
    client = genai.Client(api_key=api_key)
    base_list = list(CRITIC_MODELS if prefer_pro else SCOUT_MODELS)
    candidates = base_list + _discover_extra_models(client, base_list, prefer_pro=prefer_pro)
    last_err = None

    for model_name in candidates:
        t0 = time.time()
        mode_tag = " + تفكير عميق" if thinking_budget > 0 else ""
        log(f"🧠 جاري الاتصال بـ ({model_name}{mode_tag})...")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_request_once, client, model_name, prompt, json_mode, thinking_budget)
        try:
            raw = future.result(timeout=timeout_sec)
            pool.shutdown(wait=False)
            log(f"✅ استجاب {model_name} خلال {time.time() - t0:.1f} ثانية")
            return raw
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False)
            log(f"⏱️ الموديل {model_name} تجاوز {timeout_sec} ثانية — نجرب البديل...")
            last_err = TimeoutError(f"{model_name} timed out")
        except Exception as e:
            pool.shutdown(wait=False)
            short_err = str(e).splitlines()[0][:180]
            log(f"⚠️ الموديل {model_name} ما استجاب ({short_err}) — نجرب البديل...")
            last_err = e
            if "429" in str(e) or "503" in str(e):
                time.sleep(3)

    raise RuntimeError(f"❌ فشلت جميع موديلات Gemini: {last_err}")


def call_gemini_json(
    api_key: str,
    prompt: str,
    timeout_sec: int = 110,
    prefer_pro: bool = False,
    thinking_budget: int = 0,
) -> dict:
    raw = call_gemini_text(
        api_key,
        prompt,
        json_mode=True,
        timeout_sec=timeout_sec,
        prefer_pro=prefer_pro,
        thinking_budget=thinking_budget,
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


def parse_and_clean_segments(item: dict, max_duration: float) -> list:
    """استخراج وتنظيف قائمة المقاطع الفرعية للقص الذكي (Jump Cuts)."""
    raw_segs = item.get("segments")
    parsed = []

    if isinstance(raw_segs, list) and raw_segs:
        for seg in raw_segs:
            try:
                if isinstance(seg, dict):
                    s, e = float(seg["start"]), float(seg["end"])
                elif isinstance(seg, (list, tuple)) and len(seg) >= 2:
                    s, e = float(seg[0]), float(seg[1])
                else:
                    continue
                s = max(0.0, min(max_duration, round(s, 2)))
                e = max(0.0, min(max_duration, round(e, 2)))
                if e - s >= 1.0:
                    parsed.append([s, e])
            except (KeyError, TypeError, ValueError):
                continue

    if not parsed and "start" in item and "end" in item:
        try:
            s = max(0.0, min(max_duration, round(float(item["start"]), 2)))
            e = max(0.0, min(max_duration, round(float(item["end"]), 2)))
            if e - s >= 1.0:
                parsed.append([s, e])
        except (TypeError, ValueError):
            pass

    if not parsed:
        return []

    parsed.sort(key=lambda x: x[0])

    # دمج أي مقطعين بينهم أقل من نصف ثانية
    merged = [parsed[0]]
    for s, e in parsed[1:]:
        if s <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    # منع القفزات المجنونة (أكبر من 45 ثانية) داخل لقطة وحدة عشان ما يخلط سالفتين
    clusters = [[merged[0]]]
    for seg in merged[1:]:
        if seg[0] - clusters[-1][-1][1] <= 45.0:
            clusters[-1].append(seg)
        else:
            clusters.append([seg])

    best_cluster = max(clusters, key=lambda cl: sum(e - s for s, e in cl))
    return [(round(s, 2), round(e, 2)) for s, e in best_cluster]


def compute_clip_metrics(segments: list, hype_timeline: list) -> dict:
    windows = []
    for st, en in segments:
        windows.extend(w for w in hype_timeline if w["end"] > st and w["start"] < en)
    if not windows:
        return {"avg_audio": 0.0, "max_audio": 0.0, "chat_msgs": 0, "avg_hype": 0.0}
    return {
        "avg_audio": round(sum(w["audio"] for w in windows) / len(windows), 3),
        "max_audio": round(max(w["audio"] for w in windows), 3),
        "chat_msgs": int(sum(w["chat_msgs"] for w in windows)),
        "avg_hype": round(sum(w["hype"] for w in windows) / len(windows), 3),
    }


def extract_clip_excerpt(transcript: dict, segments: list, max_chars: int = 500) -> str:
    texts = []
    for st, en in segments:
        for seg in transcript.get("segments", []):
            if seg["end"] >= st and seg["start"] <= en and seg.get("text"):
                texts.append(seg["text"].strip())
    joined = " ".join(texts)
    return (joined[:max_chars] + "...") if len(joined) > max_chars else joined


def _scout_window(
    api_key: str,
    digest: str,
    window_start: float,
    window_end: float,
    memory_ctx: str,
    existing_clips: list,
) -> list:
    exclude_text = ""
    relevant_existing = [
        c for c in (existing_clips or [])
        if c["end"] >= window_start - 30 and c["start"] <= window_end + 30
    ]
    if relevant_existing:
        ranges = [f"- من {c['start']}s إلى {c['end']}s ({c['title']})" for c in relevant_existing]
        exclude_text = (
            "\n### لقطات مقصوصة مسبقاً (ممنوع تكرارها أو التداخل معها):\n"
            + "\n".join(ranges)
        )

    prompt = f"""أنت مخرج وممنتج تيك توك وريلز محترف جداً متخصص في بثوث تويتش العربية.
أمامك فصل من بث تويتش (من {sec_to_hms(window_start)} إلى {sec_to_hms(window_end)}) مع علامات الصوت العالي (🔥/⚡)، الشات (💬)، وفترات الصمت (🔇).

### قواعد اختيار اللقطات والقصّ الداخلي الذكي (Jump Cuts):
1. **رشّح فقط اللقطات الفيرال الحقيقية** (ضحك حقيقي، طقطقة عفوية مع الشات، ردة فعل أو توتر مضحك، سالفة رهيبة ومكثفة). **ممنوع ترشيح الكلام العادي، الفضفضة الجانبية، بريكات الصلاة، مواضيع الميكب، أو اللعب الصامت.**
2. **القصّ المتقطع الذكي (`segments`) — أهم ميزة:**
   - إذا اللقطة ممتازة ومشدودة من بدايتها لنهايتها، حطها في مقطع واحد داخل `segments`.
   - **إذا اللقطة رهيبة بس في نصها تشتتت الستريمر** (رحّبت بشخص في الشات، دخلت في سالفة ما لها دخل، أو سكتت فترة طويلة)، **احذف الجزء الغير ضروري من النص** وحط فقط الأجزاء الممتازة كقائمة مقاطع متتالية داخل `segments`!
   - مثال: لو السالفة من 100s إلى 160s بس من 125s إلى 138s رحّبت بشخص بالشات أو سكتت، اكتب:
     `"segments": [{{"start": 100.0, "end": 125.0}}, {{"start": 138.0, "end": 160.0}}]`
3. **الرتم المشدود:** حاول تكون المدة الصافية للقطة (بعد حذف الحشو) بين 15 و 55 ثانية قدر الإمكان (إلا لو سالفة أسطورية فعلاً بدون أي ملل).
4. **موضوع واحد لكل لقطة**، وبداية/نهاية نظيفة بدون قطع وسط الجملة أو قبل القفلة.
5. إذا ما فيه شي يستاهل النشر في هذا الفصل، رجّع `"clips": []` بدون مجاملة!
{memory_ctx}
{exclude_text}

أعد النتيجة بصيغة JSON فقط بهذا الهيكل الدقيق:
{{
  "clips": [
    {{
      "title": "عنوان عربي قصير وجذاب (2 إلى 5 كلمات)",
      "segments": [
        {{"start": 100.2, "end": 124.8}},
        {{"start": 135.5, "end": 158.0}}
      ],
      "category": "تفاعل_شات أو ضحك أو صراخ_وتوتر أو سالفة أو قيمينق",
      "virality_score": 9,
      "reason": "ليش اللقطة أسطورية ومناسبة لتيك توك",
      "cut_notes": "اكتب هنا لو حذفت شي من النص (مثل: حذف ترحيب بالشات/صمت) أو اكتب 'لقطة كاملة متصلة'"
    }}
  ]
}}

=== نص هذا الفصل من البث ===
{digest}
"""
    result = call_gemini_json(api_key, prompt, prefer_pro=False, thinking_budget=1024)
    return result.get("clips", [])


def _critic_review_batch(
    api_key: str,
    candidates: list,
    transcript: dict,
    hype_timeline: list,
    chat_messages: list,
    memory_ctx: str,
) -> list:
    """المرحلة الثانية: مخرج ناقد صارم يفلتر المرشحين ويضبط قصّات الـ Jump Cuts بالثانية."""
    dossier_parts = []
    for idx, cand in enumerate(candidates, start=1):
        box_s = max(0.0, cand["start"] - 8.0)
        box_e = cand["end"] + 8.0
        local_digest = build_stream_digest(transcript, hype_timeline, chat_messages, box_s, box_e)
        segs_str = ", ".join(f"[{s}s → {e}s]" for s, e in cand["segments"])
        dossier_parts.append(
            f"--- مرشح رقم {idx} ---\n"
            f"العنوان المقترح: {cand['title']} | التصنيف: {cand['category']} | تقييم الكشاف: {cand['virality_score']}/10\n"
            f"المقاطع المقترحة: {segs_str} (المدة الصافية: {cand['duration']}ث)\n"
            f"ملاحظة الكشاف: {cand['reason']} | قصّات مقترحة: {cand.get('cut_notes', '')}\n"
            f"النص الكامل مع هامش قبل وبعد للتدقيق:\n{local_digest}\n"
        )

    dossier_text = "\n".join(dossier_parts)

    prompt = f"""أنت رئيس تحرير ومخرج تيك توك صارم جداً (ما تجامل أبداً).
عندك {len(candidates)} لقطة مرشحة من بث تويتش، ومع كل لقطة نصها الكامل بتوقيت الثواني وعلامات الصمت (🔇).

مهمتك على كل مرشح:
1. **القرار (`keep`):**
   - احفظ (`true`) **فقط** اللقطات اللي فعلاً تضحّك أو فيها تفاعل شات قوي أو توتر/ردة فعل رهيبة أو سالفة كاملة ممتعة تجيب مشاهدات لجمهور جديد.
   - ارفض (`false`) أي لقطة فيها كلام عادي، فضفضة جانبية، ملل، طول زايد بدون قيمة، أو صراخ لعبة بدون تفاعل ممتع، أو أي شي يخالف قواعد ذوق الممنتج تحت!
2. **تنظيف وتحسين القصّ (`segments` — Jump Cuts):**
   - لكل لقطة قبلتها، راجع السطور بالثانية!
   - تأكد إن البداية تبدأ بخطاف قوي (بدون تمهيد ممل) وإن النهاية تقفل على الضحكة أو القفلة بدون بتر.
   - **احذف من داخل اللقطة** أي ترحيب بشخص في الشات، أي تشتت لموضوع ثاني، أو أي فترة صمت/حشو في النص، عبر تقسيم `segments` إلى مقاطع متتالية نظيفة!
   - اجعل المدة الصافية مشدودة وممتعة (يفضّل أقل من 55–60 ثانية).
{memory_ctx}

أعد النتيجة بصيغة JSON فقط بهذا الهيكل الدقيق:
{{
  "reviewed": [
    {{
      "candidate_num": 1,
      "keep": true,
      "title": "عنوان عربي قوي وجذاب (2 إلى 5 كلمات)",
      "segments": [
        {{"start": 102.0, "end": 124.5}},
        {{"start": 133.2, "end": 152.0}}
      ],
      "category": "تفاعل_شات أو ضحك أو صراخ_وتوتر أو سالفة أو قيمينق",
      "virality_score": 9,
      "reason": "ليش تستاهل النشر في تيك توك",
      "cut_notes": "شرح مختصر للقصّات الداخلية اللي سويتها أو 'لقطة كاملة متصلة'"
    }}
  ]
}}

=== ملف المرشحين للمراجعة الصارمة ===
{dossier_text}
"""
    res = call_gemini_json(api_key, prompt, timeout_sec=130, prefer_pro=True, thinking_budget=2048)
    return res.get("reviewed", [])


def select_clips(
    api_key: str,
    vod_dir: str,
    drive_root: str,
    existing_clips: list = None,
) -> list:
    with open(os.path.join(vod_dir, "transcript.json"), "r", encoding="utf-8") as f:
        transcript = json.load(f)
    with open(os.path.join(vod_dir, "hype_timeline.json"), "r", encoding="utf-8") as f:
        hype_timeline = json.load(f)
    with open(os.path.join(vod_dir, "chat.json"), "r", encoding="utf-8") as f:
        chat_messages = json.load(f)

    duration = float(transcript.get("duration", 0))
    segs_count = len(transcript.get("segments", []))
    log(f"📄 جاري قراءة التفريغ ({segs_count} جملة / {duration / 60:.1f} دقيقة)...")

    memory_ctx = load_memory_context(drive_root)
    if memory_ctx:
        log("🧠 تم تحميل قواعد ذوقك الدائمة وتقييماتك السابقة — الكشاف والناقد بيطبقونها حرفياً!")

    # فصول 20 دقيقة مع تداخل دقيقة ونصف لتركيز عالي جداً على كل ذبّة
    if duration <= 1500:
        windows = [(0.0, duration)]
    else:
        chunk_len = 1200.0
        overlap = 90.0
        windows = []
        cur = 0.0
        while cur < duration:
            end_w = min(duration, cur + chunk_len)
            windows.append((cur, end_w))
            if end_w >= duration:
                break
            cur = end_w - overlap
        log(f"📚 تم تقسيم البث إلى {len(windows)} فصول قصيرة (20 دقيقة) لتركيز أقصى على اللقطات القوية!")

    existing_ranges = [(c["start"], c["end"]) for c in (existing_clips or [])]
    running_existing = list(existing_clips or [])
    candidates = []

    log("\n=== 🔎 المرحلة 1/2: مسح البث وترشيح اللقطات القوية ===")
    for idx, (w_start, w_end) in enumerate(windows, start=1):
        if len(windows) > 1:
            log(f"\n🔎 فحص الفصل {idx}/{len(windows)} ({sec_to_hms(w_start)} → {sec_to_hms(w_end)})...")
        digest = build_stream_digest(transcript, hype_timeline, chat_messages, w_start, w_end)
        if len(digest.strip()) < 80:
            continue

        try:
            raw_clips = _scout_window(
                api_key=api_key,
                digest=digest,
                window_start=w_start,
                window_end=w_end,
                memory_ctx=memory_ctx,
                existing_clips=running_existing,
            )
        except Exception as e:
            log(f"⚠️ تعذر فحص الفصل {idx}: {e} — نكمل باقي الفصول")
            continue

        for item in raw_clips:
            segs = parse_and_clean_segments(item, duration)
            if not segs:
                continue
            net_dur = round(sum(e - s for s, e in segs), 2)
            if net_dur < 7.0:
                continue
            st, en = segs[0][0], segs[-1][1]
            overlaps = any(not (en <= a or st >= b) for a, b in existing_ranges)
            if overlaps:
                continue

            existing_ranges.append((st, en))
            cand = {
                "title": str(item.get("title", "لقطة")).strip(),
                "segments": segs,
                "start": st,
                "end": en,
                "duration": net_dur,
                "category": str(item.get("category", "منوع")).strip(),
                "virality_score": int(item.get("virality_score", 8)),
                "reason": str(item.get("reason", "")).strip(),
                "cut_notes": str(item.get("cut_notes", "لقطة كاملة متصلة")).strip(),
            }
            candidates.append(cand)
            running_existing.append(cand)

        if idx < len(windows):
            time.sleep(2)

    if not candidates:
        return []

    log(f"\n=== ⚖️ المرحلة 2/2: المخرج الناقد يفلتر {len(candidates)} مرشح ويضبط الـ Jump Cuts ===")
    final_clips = []
    batch_size = 10
    kept_ranges = [(c["start"], c["end"]) for c in (existing_clips or [])]

    for b_start in range(0, len(candidates), batch_size):
        batch = candidates[b_start : b_start + batch_size]
        log(f"\n🎯 مراجعة دفعة ({b_start + 1} إلى {b_start + len(batch)}) من أصل {len(candidates)}...")
        try:
            reviewed = _critic_review_batch(
                api_key=api_key,
                candidates=batch,
                transcript=transcript,
                hype_timeline=hype_timeline,
                chat_messages=chat_messages,
                memory_ctx=memory_ctx,
            )
        except Exception as e:
            log(f"⚠️ تعذر المخرج الناقد لهذه الدفعة ({e}) — بنعتمد المرشحين الأعلى تقييماً")
            reviewed = [
                {
                    "candidate_num": i + 1,
                    "keep": c["virality_score"] >= 8,
                    **c,
                }
                for i, c in enumerate(batch)
            ]

        for rev in reviewed:
            if not rev.get("keep", False):
                c_idx = int(rev.get("candidate_num", 0)) - 1
                rejected_title = batch[c_idx]["title"] if 0 <= c_idx < len(batch) else rev.get("title", "")
                log(f"   🗑️ استبعد الناقد: {rejected_title} (ما ترقى للمستوى المطلوب)")
                continue

            segs = parse_and_clean_segments(rev, duration)
            if not segs:
                c_idx = int(rev.get("candidate_num", 0)) - 1
                if 0 <= c_idx < len(batch):
                    segs = batch[c_idx]["segments"]
                else:
                    continue

            net_dur = round(sum(e - s for s, e in segs), 2)
            if net_dur < 7.0:
                continue

            st, en = segs[0][0], segs[-1][1]
            if any(not (en <= a or st >= b) for a, b in kept_ranges):
                continue

            kept_ranges.append((st, en))
            metrics = compute_clip_metrics(segs, hype_timeline)
            final_clips.append({
                "title": str(rev.get("title", "لقطة")).strip(),
                "segments": [[s, e] for s, e in segs],
                "start": st,
                "end": en,
                "duration": net_dur,
                "category": str(rev.get("category", "منوع")).strip(),
                "virality_score": int(rev.get("virality_score", 8)),
                "reason": str(rev.get("reason", "")).strip(),
                "cut_notes": str(rev.get("cut_notes", "لقطة كاملة متصلة")).strip(),
                "excerpt": extract_clip_excerpt(transcript, segs),
                "metrics": metrics,
                "rating": "pending",
                "note": "",
            })

    final_clips.sort(key=lambda x: x["start"])
    log(f"\n🏆 النتيجة النهائية بعد فلترة الناقد: {len(final_clips)} لقطة صافية من أصل {len(candidates)} مرشح!")
    return final_clips


def _extract_audio_rms(audio_path: str, start_sec: float, end_sec: float, step_sec: float = 0.1):
    dur = max(0.5, end_sec - start_sec)
    sr = 4000
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-ss", f"{start_sec:.3f}",
        "-t", f"{dur:.3f}",
        "-i", audio_path,
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0 or not res.stdout:
        return None
    samples = np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    frame = int(sr * step_sec)
    n_frames = len(samples) // frame
    if n_frames < 5:
        return None
    trimmed = samples[: n_frames * frame].reshape(n_frames, frame)
    return np.sqrt(np.mean(trimmed ** 2, axis=1))


def shave_dead_air_segments(
    segments: list,
    transcript: dict,
    audio_path: str,
    min_silence_sec: float = 1.15,
    keep_after_word: float = 0.18,
    keep_before_word: float = 0.14,
) -> tuple:
    """
    حذّاف صمت ذكي يحذف فقط السكون الحقيقي (بدون كلام + صوت منخفض جداً)
    ويحافظ 100% على الضحك والصراخ وردات الفعل اللي بدون كلام.
    """
    if not audio_path or not os.path.exists(audio_path):
        return segments, 0.0

    # جمع توقيت كل الكلمات مرة وحدة
    all_words = []
    for seg in transcript.get("segments", []):
        if seg.get("words"):
            for w in seg["words"]:
                all_words.append((float(w["s"]), float(w["e"])))
        else:
            all_words.append((float(seg["start"]), float(seg["end"])))
    all_words.sort(key=lambda x: x[0])

    refined = []
    total_shaved = 0.0
    step = 0.1

    for seg_s, seg_e in segments:
        seg_dur = seg_e - seg_s
        if seg_dur < 4.0:
            refined.append((seg_s, seg_e))
            continue

        rms = _extract_audio_rms(audio_path, seg_s, seg_e, step_sec=step)
        if rms is None or len(rms) < 15:
            refined.append((seg_s, seg_e))
            continue

        ref_level = float(np.percentile(rms, 75))
        if ref_level < 0.005:
            refined.append((seg_s, seg_e))
            continue
        sil_thr = max(0.004, ref_level * 0.16)

        local_words = [
            (max(seg_s, ws), min(seg_e, we))
            for ws, we in all_words
            if we > seg_s and ws < seg_e
        ]
        if not local_words:
            refined.append((seg_s, seg_e))
            continue

        # البحث عن فجوات الكلام الداخلية والخارجية
        cuts = []

        # 1. تقليم السكوت الميت في بداية اللقطة (> 0.7ث) عشان الخطاف يضرب من أول ثانية
        lead_gap = local_words[0][0] - seg_s
        if lead_gap >= 0.75:
            end_idx = min(len(rms), int((lead_gap - keep_before_word) / step))
            if end_idx >= 5 and np.max(rms[:end_idx]) < sil_thr * 1.4:
                cuts.append((seg_s, local_words[0][0] - keep_before_word))

        # 2. تقليم السكوت الميت الداخلي (>= 1.15ث)
        for i in range(len(local_words) - 1):
            g_start = local_words[i][1]
            g_end = local_words[i + 1][0]
            if g_end - g_start < min_silence_sec:
                continue

            c_start = g_start + keep_after_word
            c_end = g_end - keep_before_word
            if c_end - c_start < 0.7:
                continue

            i0 = max(0, int((c_start - seg_s) / step))
            i1 = min(len(rms), int((c_end - seg_s) / step))
            if i1 <= i0:
                continue

            # نتأكد إن الفجوة صامتة فعلاً (ما فيها ضحكة أو صرخة!)
            quiet_mask = rms[i0:i1] < sil_thr
            if np.mean(quiet_mask) >= 0.85:
                cuts.append((round(c_start, 2), round(c_end, 2)))

        # 3. تقليم السكوت الميت في آخر اللقطة (> 0.9ث)
        tail_gap = seg_e - local_words[-1][1]
        if tail_gap >= 0.95:
            start_idx = max(0, int(((local_words[-1][1] + keep_after_word) - seg_s) / step))
            if start_idx < len(rms) - 5 and np.max(rms[start_idx:]) < sil_thr * 1.4:
                cuts.append((local_words[-1][1] + keep_after_word, seg_e))

        if not cuts:
            refined.append((seg_s, seg_e))
            continue

        # تحويل الحذوفات إلى مقاطع صافية
        cur_s = seg_s
        local_pieces = []
        for cs, ce in sorted(cuts):
            if cs > cur_s + 0.35:
                local_pieces.append((round(cur_s, 2), round(cs, 2)))
            total_shaved += max(0.0, ce - max(cur_s, cs))
            cur_s = max(cur_s, ce)
        if seg_e > cur_s + 0.35:
            local_pieces.append((round(cur_s, 2), round(seg_e, 2)))

        if local_pieces:
            refined.extend(local_pieces)
        else:
            refined.append((seg_s, seg_e))

    # دمج أي مقطعين صغار المسافة بينهم أقل من 0.35ث
    final_segs = []
    for s, e in refined:
        if e - s < 0.45:
            continue
        if final_segs and s <= final_segs[-1][1] + 0.35:
            final_segs[-1] = (final_segs[-1][0], max(final_segs[-1][1], e))
        else:
            final_segs.append((s, e))

    if not final_segs:
        return segments, 0.0
    return final_segs, round(total_shaved, 2)


def _compute_local_wps(words: list, idx: int, window_sec: float = 2.5) -> float:
    center = (words[idx]["s"] + words[idx]["e"]) / 2.0
    w_start = center - window_sec
    w_end = center + window_sec
    nearby = [w for w in words if w["e"] >= w_start and w["s"] <= w_end]
    if len(nearby) <= 1:
        return 1.8
    span = max(0.6, nearby[-1]["e"] - nearby[0]["s"])
    return len(nearby) / span


def generate_stitched_srt(
    transcript: dict,
    segments: list,
    out_srt: str,
) -> tuple:
    """
    توليد سبتايتل ديناميكي متزامن بالملي ثانية مع المقاطع المدمجة (Jump Cuts)
    ويتأقلم تلقائياً مع سرعة كلام الستريمر.
    """
    mapped_words = []
    timeline_offset = 0.0

    for seg_idx, (seg_s, seg_e) in enumerate(segments):
        piece_dur = seg_e - seg_s
        first_in_piece = True

        for tseg in transcript.get("segments", []):
            if tseg["end"] < seg_s or tseg["start"] > seg_e:
                continue

            seg_words = tseg.get("words") or []
            if seg_words:
                for w in seg_words:
                    mid = (w["s"] + w["e"]) / 2.0
                    if seg_s <= mid <= seg_e:
                        txt = w["w"].strip()
                        if txt:
                            mapped_words.append({
                                "w": txt,
                                "s": round(timeline_offset + max(0.0, w["s"] - seg_s), 3),
                                "e": round(timeline_offset + min(piece_dur, w["e"] - seg_s), 3),
                                "new_cut": first_in_piece and seg_idx > 0,
                            })
                            first_in_piece = False
            else:
                tokens = tseg["text"].split()
                if tokens:
                    s0 = max(seg_s, tseg["start"])
                    e0 = min(seg_e, tseg["end"])
                    step = (e0 - s0) / len(tokens)
                    for idx, tok in enumerate(tokens):
                        mapped_words.append({
                            "w": tok,
                            "s": round(timeline_offset + max(0.0, (s0 + idx * step) - seg_s), 3),
                            "e": round(timeline_offset + min(piece_dur, (s0 + (idx + 1) * step) - seg_s), 3),
                            "new_cut": first_in_piece and seg_idx > 0,
                        })
                        first_in_piece = False

        timeline_offset += piece_dur

    total_dur = timeline_offset
    entries = []
    buf = []
    wps_samples = []

    def flush():
        if not buf:
            return
        text = " ".join(x["w"] for x in buf).strip()
        st = buf[0]["s"]
        en = max(buf[-1]["e"], st + 0.22)
        en = min(en, total_dur)
        if text:
            entries.append((st, en, text))
        buf.clear()

    for i, w in enumerate(mapped_words):
        tok = w["w"]
        if not tok:
            continue

        # فصل السطر فوراً عند أي Jump Cut عشان النص يمشي مضبوط مع القطع
        if w.get("new_cut") and buf:
            flush()

        wps = _compute_local_wps(mapped_words, i)
        wps_samples.append(wps)

        if wps >= 3.0:
            max_words, max_chars, pause_thr = 2, 16, 0.28
        elif wps >= 2.1:
            max_words, max_chars, pause_thr = 4, 26, 0.40
        else:
            max_words, max_chars, pause_thr = 6, 42, 0.58

        if buf:
            gap = w["s"] - buf[-1]["e"]
            projected_len = len(" ".join(x["w"] for x in buf)) + 1 + len(tok)
            if gap > pause_thr or len(buf) >= max_words or projected_len > max_chars:
                flush()

        buf.append(w)

        if tok.endswith(("؟", "!", ".", "?")):
            flush()
        elif tok.endswith("،") and len(buf) >= 2:
            flush()
    flush()

    srt_blocks = []
    for idx, (st, en, text) in enumerate(entries, start=1):
        srt_blocks.append(f"{idx}\n{sec_to_srt_time(st)} --> {sec_to_srt_time(en)}\n{text}")

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""))

    avg_wps = round(sum(wps_samples) / len(wps_samples), 2) if wps_samples else 0.0
    return len(entries), avg_wps


def resolve_stream_urls(vod_url: str) -> list:
    log("🔗 جاري استخراج رابط البث المباشر للقص السريع...")
    cmd = [
        "yt-dlp",
        "-f", "best[height<=1080]/bestvideo[height<=1080]+bestaudio/best",
        "-g",
        "--no-playlist",
        "--no-warnings",
        vod_url,
    ]
    out = subprocess.check_output(cmd, text=True, timeout=60).strip()
    urls = [line.strip() for line in out.splitlines() if line.strip()]
    if not urls:
        raise RuntimeError("تعذر استخراج رابط البث من تويتش")
    return urls


def cut_and_stitch_video(stream_urls: list, segments: list, out_mp4: str):
    """قصّ ودمج مقطع واحد أو عدة مقاطع (Jump Cuts) بسلاسة تامة وبدون طقطقة صوت."""
    local_tmp = os.path.join("/tmp", os.path.basename(out_mp4))
    if os.path.exists(local_tmp):
        os.remove(local_tmp)

    encoders = [
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21", "-pix_fmt", "yuv420p"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
    ]

    last_err = ""
    success = False

    if len(segments) == 1:
        start, end = segments[0]
        duration = max(1.0, end - start)
        input_args = []
        for u in stream_urls[:2]:
            input_args += ["-ss", f"{start:.3f}", "-i", u]

        for enc in encoders:
            cmd = (
                ["ffmpeg", "-y", "-nostdin", "-v", "error"]
                + input_args
                + ["-t", f"{duration:.3f}"]
                + enc
                + ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", local_tmp]
            )
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=240)
            if res.returncode == 0 and os.path.exists(local_tmp) and os.path.getsize(local_tmp) > 10_000:
                success = True
                break
            last_err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"exit code {res.returncode}"
    else:
        box_start = max(0.0, segments[0][0] - 0.5)
        box_end = segments[-1][1] + 0.5
        box_dur = (box_end - box_start) + 2.0

        input_args = []
        for u in stream_urls[:2]:
            input_args += ["-ss", f"{box_start:.3f}", "-t", f"{box_dur:.3f}", "-i", u]

        v_in = "0:v"
        a_in = "1:a" if len(stream_urls) > 1 else "0:a"

        filter_lines = []
        concat_inputs = []
        for i, (s, e) in enumerate(segments):
            rs = max(0.0, s - box_start)
            re_t = max(rs + 0.1, e - box_start)
            seg_d = re_t - rs
            fade_d = min(0.015, seg_d / 6.0)
            fade_out_st = max(0.0, seg_d - fade_d)

            filter_lines.append(f"[{v_in}]trim=start={rs:.3f}:end={re_t:.3f},setpts=PTS-STARTPTS[v{i}]")
            filter_lines.append(
                f"[{a_in}]atrim=start={rs:.3f}:end={re_t:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:st=0:d={fade_d:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}[a{i}]"
            )
            concat_inputs.append(f"[v{i}][a{i}]")

        filter_lines.append(f"{''.join(concat_inputs)}concat=n={len(segments)}:v=1:a=1[vout][aout]")
        filter_complex = ";".join(filter_lines)

        for enc in encoders:
            cmd = (
                ["ffmpeg", "-y", "-nostdin", "-v", "error"]
                + input_args
                + ["-filter_complex", filter_complex, "-map", "[vout]", "-map", "[aout]"]
                + enc
                + ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", local_tmp]
            )
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            if res.returncode == 0 and os.path.exists(local_tmp) and os.path.getsize(local_tmp) > 10_000:
                success = True
                break
            last_err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"exit code {res.returncode}"

    if not success:
        raise RuntimeError(f"فشل FFmpeg أثناء القص والدمج: {last_err}")

    if os.path.exists(out_mp4):
        os.remove(out_mp4)
    shutil.copy2(local_tmp, out_mp4)
    os.remove(local_tmp)


def _cleanup_broken_parts(clips_dir: str):
    for fname in os.listdir(clips_dir):
        if fname.endswith(".part"):
            try:
                os.remove(os.path.join(clips_dir, fname))
            except OSError:
                pass


def run_stage3(
    vod_dir: str,
    drive_root: str,
    api_key: str,
    extra_clips: bool = False,
    redo_all: bool = False,
    auto_trim_silence: bool = True,
):
    meta_path = os.path.join(vod_dir, "meta.json")
    transcript_path = os.path.join(vod_dir, "transcript.json")
    audio_path = os.path.join(vod_dir, "audio.mp3")
    manifest_path = os.path.join(vod_dir, "clips_manifest.json")
    clips_dir = os.path.join(vod_dir, "clips")

    if redo_all:
        if os.path.exists(manifest_path):
            os.remove(manifest_path)
        if os.path.isdir(clips_dir):
            shutil.rmtree(clips_dir, ignore_errors=True)
        log("🔄 تم تفعيل إعادة القص من الصفر — تم مسح اللقطات القديمة (مع الحفاظ على تفريغ البث وقواعد ذوقك)")

    os.makedirs(clips_dir, exist_ok=True)
    _cleanup_broken_parts(clips_dir)

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    vod_url = meta["vod_url"]

    existing_clips = []
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            existing_clips = json.load(f)

    if existing_clips and not extra_clips:
        log(f"ℹ️ يوجد بالفعل {len(existing_clips)} لقطة جاهزة لهذا البث.")
        log("   - لو تبي لقطات إضافية: فعّل EXTRA_CLIPS.")
        log("   - لو تبي تعيد قص البث بالمحرك الجديد وقواعد ذوقك: فعّل REDO_ALL_CLIPS.")
        return existing_clips

    new_clips = select_clips(
        api_key=api_key,
        vod_dir=vod_dir,
        drive_root=drive_root,
        existing_clips=existing_clips,
    )

    if not new_clips:
        log("⚠️ ما لقى المخرج الناقد لقطات جديدة تستاهل النشر.")
        return existing_clips

    stream_urls = resolve_stream_urls(vod_url)
    start_index = len(existing_clips) + 1
    log(f"\n🎬 جاري تنفيذ القصّ والدمج الذكي (Jump Cuts) لـ {len(new_clips)} لقطة...\n")

    for offset, clip in enumerate(new_clips):
        clip_num = start_index + offset
        safe_title = sanitize_filename(clip["title"])
        base_name = f"clip_{clip_num:02d}_{safe_title}"
        mp4_path = os.path.join(clips_dir, f"{base_name}.mp4")
        srt_path = os.path.join(clips_dir, f"{base_name}.srt")

        raw_segs = [(float(s), float(e)) for s, e in clip["segments"]]
        raw_span = round(raw_segs[-1][1] - raw_segs[0][0], 2)

        if auto_trim_silence:
            final_segs, shaved_sil = shave_dead_air_segments(raw_segs, transcript, audio_path)
        else:
            final_segs, shaved_sil = raw_segs, 0.0

        net_dur = round(sum(e - s for s, e in final_segs), 2)
        removed_total = round(max(0.0, raw_span - net_dur), 2)

        segs_preview = " + ".join(f"[{sec_to_hms(s)}→{sec_to_hms(e)}]" for s, e in final_segs)
        log(
            f"✂️ [{clip_num:02d}] {clip['title']} "
            f"(المدة الصافية: {net_dur}ث | {len(final_segs)} جزء مدمج | حذف {removed_total}ث حشو وصمت | تقييم {clip['virality_score']}/10)"
        )
        log(f"    💡 السبب: {clip['reason']}")
        log(f"    🎞️ القصّات: {segs_preview}")

        t0 = time.time()
        try:
            cut_and_stitch_video(stream_urls, final_segs, mp4_path)
            sub_count, avg_wps = generate_stitched_srt(transcript, final_segs, srt_path)
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)

            clip["id"] = clip_num
            clip["segments"] = [[s, e] for s, e in final_segs]
            clip["start"] = final_segs[0][0]
            clip["end"] = final_segs[-1][1]
            clip["duration"] = net_dur
            clip["removed_sec"] = removed_total
            clip["shaved_silence_sec"] = shaved_sil
            clip["video_path"] = mp4_path
            clip["srt_path"] = srt_path
            clip["avg_wps"] = avg_wps
            existing_clips.append(clip)

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(existing_clips, f, ensure_ascii=False, indent=2)

            log(
                f"    ✅ جاهز خلال {time.time() - t0:.1f}ث! ({size_mb:.1f} MB | "
                f"سبتايتل متزامن {sub_count} سطر | سرعة {avg_wps} كلمة/ث)\n"
            )
        except Exception as e:
            log(f"    ❌ فشل قص اللقطة رقم {clip_num}: {e}\n")

    log(f"🎉 خلصنا! افتح الاستوديو (الخلية 4) وشوف الفرق الجبار! المسار:\n   {clips_dir}")
    return existing_clips


# === END OF stage3_clipper.py ===
