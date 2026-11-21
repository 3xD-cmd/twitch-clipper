import concurrent.futures
import json
import os
import re
import subprocess
import time
from google import genai
from google.genai import types

PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]


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
            parts.append(f"### قواعد ذوق الممنتج (مهم جداً الالتزام بها):\n{rules}")

    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        liked = [h for h in history if h.get("rating") == "liked"][-6:]
        disliked = [h for h in history if h.get("rating") == "disliked"][-6:]
        if liked:
            lines = [f"- 👍 {x.get('title')} ({x.get('category', '')}) — ملاحظة: {x.get('note', 'ممتازة')}" for x in liked]
            parts.append("### أمثلة لقطات أعجبت الممنتج مؤخراً:\n" + "\n".join(lines))
        if disliked:
            lines = [f"- 👎 {x.get('title')} ({x.get('category', '')}) — سبب الرفض: {x.get('note', 'ما تنفع')}" for x in disliked]
            parts.append("### أمثلة لقطات رفضها الممنتج (تجنب مثلها):\n" + "\n".join(lines))

    return "\n\n".join(parts)


def build_stream_digest(transcript: dict, hype_timeline: list, chat_messages: list) -> str:
    hype_by_bucket = {}
    for item in hype_timeline:
        bucket = int(item["start"] // 10)
        hype_by_bucket[bucket] = item

    chat_by_bucket = {}
    for msg in chat_messages:
        bucket = int(msg["t"] // 10)
        chat_by_bucket.setdefault(bucket, []).append(msg["m"])

    lines = []
    last_chat_bucket = -1

    for seg in transcript.get("segments", []):
        st = seg["start"]
        bucket = int(st // 10)
        h_info = hype_by_bucket.get(bucket, {})
        tags = []
        if h_info.get("audio", 0) >= 0.75:
            tags.append("🔥صوت_عالي")
        if h_info.get("chat_msgs", 0) >= 3:
            tags.append(f"💬شات({h_info['chat_msgs']})")

        tag_str = f" [{' '.join(tags)}]" if tags else ""
        lines.append(f"[{st:.1f}s - {seg['end']:.1f}s]{tag_str} {seg['text']}")

        if bucket != last_chat_bucket and bucket in chat_by_bucket and len(chat_by_bucket[bucket]) >= 2:
            sample = " | ".join(chat_by_bucket[bucket][:4])
            lines.append(f"   >> الشات: {sample}")
            last_chat_bucket = bucket

    return "\n".join(lines)


def _discover_extra_flash_models(client) -> list:
    found = []
    try:
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "flash" in name.lower() and ("generateContent" in actions or not actions):
                if name not in PREFERRED_MODELS:
                    found.append(name)
    except Exception:
        pass
    return found[:3]


def _request_once(client, model_name: str, prompt: str) -> str:
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    return (resp.text or "").strip()


def call_gemini_json(api_key: str, prompt: str, timeout_sec: int = 60) -> dict:
    client = genai.Client(api_key=api_key)
    candidates = list(PREFERRED_MODELS) + _discover_extra_flash_models(client)
    last_err = None

    for model_name in candidates:
        t0 = time.time()
        log(f"🧠 جاري استشارة الذكاء الاصطناعي ({model_name}) — مهلة الانتظار {timeout_sec} ثانية...")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_request_once, client, model_name, prompt)
        try:
            raw = future.result(timeout=timeout_sec)
            pool.shutdown(wait=False)
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)
            log(f"✅ تم اختيار اللقطات بنجاح عبر {model_name} في {time.time() - t0:.1f} ثانية")
            return parsed
        except concurrent.futures.TimeoutError:
            pool.shutdown(wait=False)
            log(f"⏱️ الموديل {model_name} تجاوز {timeout_sec} ثانية بدون رد — نجرب البديل...")
            last_err = TimeoutError(f"{model_name} timed out")
        except Exception as e:
            pool.shutdown(wait=False)
            short_err = str(e).splitlines()[0][:180]
            log(f"⚠️ الموديل {model_name} ما استجاب ({short_err}) — نجرب البديل...")
            last_err = e

    raise RuntimeError(f"❌ فشلت جميع موديلات Gemini: {last_err}")


def compute_clip_metrics(start: float, end: float, hype_timeline: list) -> dict:
    windows = [w for w in hype_timeline if w["end"] > start and w["start"] < end]
    if not windows:
        return {"avg_audio": 0.0, "max_audio": 0.0, "chat_msgs": 0, "avg_hype": 0.0}
    return {
        "avg_audio": round(sum(w["audio"] for w in windows) / len(windows), 3),
        "max_audio": round(max(w["audio"] for w in windows), 3),
        "chat_msgs": int(sum(w["chat_msgs"] for w in windows)),
        "avg_hype": round(sum(w["hype"] for w in windows) / len(windows), 3),
    }


def select_clips(
    api_key: str,
    vod_dir: str,
    drive_root: str,
    num_clips: int = 4,
    min_sec: int = 18,
    max_sec: int = 60,
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

    digest = build_stream_digest(transcript, hype_timeline, chat_messages)
    memory_ctx = load_memory_context(drive_root)

    exclude_text = ""
    if existing_clips:
        ranges = [f"- من {c['start']}s إلى {c['end']}s ({c['title']})" for c in existing_clips]
        exclude_text = (
            "\n### تنبيه مهم — لقطات تم قصّها مسبقاً (ممنوع تكرارها أو التداخل معها):\n"
            + "\n".join(ranges)
        )

    prompt = f"""أنت ممنتج ومخرج محتوى محترف متخصص في تحويل بثوث تويتش العربية (قيمينق وسوالف) إلى مقاطع TikTok و Reels فيرال تجيب مشاهدات عالية.

أمامك نص بث كامل مع التوقيت بالثواني، وعلامات ارتفاع الصوت (🔥صوت_عالي) وتفاعل الشات (💬شات).
مدة البث المتاحة: {duration:.1f} ثانية.

المطلوب:
اختر أفضل {num_clips} لقطات مستقلة تصلح للقص والنشر في TikTok / Reels.

شروط الاختيار والقصّ الصارمة:
1. مدة كل لقطة تكون بين {min_sec} ثانية و {max_sec} ثانية (مرونة بسيطة مسموحة فقط لإكمال الجملة).
2. ممنوع تبدأ اللقطة من نص جملة أو تنهيها قبل ما تخلص القفلة/الضحكة/ردة الفعل.
3. ركّز على: الذبّات القوية، الضحك، المواقف المحرجة، الطقطقة مع الشات، حماس أو عصبية القيمينق، السوالف اللي لها خطاف (Hook) في أول 3 ثوانٍ.
4. تجنب المقاطع الميتة أو السكوت الطويل أو الكلام التقني الممل عن إعدادات البث (إلا لو صار فيه موقف مضحك).
5. ممنوع تداخل الأوقات بين اللقطات المختارة.
{memory_ctx}
{exclude_text}

أعد النتيجة بصيغة JSON فقط بهذا الهيكل الدقيق:
{{
  "clips": [
    {{
      "title": "عنوان عربي قصير ومشوق (2 إلى 5 كلمات)",
      "start": 12.5,
      "end": 48.0,
      "category": "ضحك أو سالفة أو قيمينق أو تفاعل_شات أو موقف_محرج",
      "virality_score": 8,
      "reason": "سبب اختيار اللقطة في سطر واحد"
    }}
  ]
}}

=== نص البث ===
{digest}
"""

    result = call_gemini_json(api_key, prompt)
    raw_clips = result.get("clips", [])
    existing_ranges = [(c["start"], c["end"]) for c in (existing_clips or [])]

    valid_clips = []
    for item in raw_clips:
        try:
            st = max(0.0, round(float(item["start"]), 2))
            en = min(duration, round(float(item["end"]), 2))
        except (KeyError, TypeError, ValueError):
            continue

        if en - st < 8.0:
            continue

        overlaps = any(not (en <= a or st >= b) for a, b in existing_ranges)
        if overlaps:
            continue

        existing_ranges.append((st, en))
        metrics = compute_clip_metrics(st, en, hype_timeline)
        valid_clips.append({
            "title": str(item.get("title", "لقطة")).strip(),
            "start": st,
            "end": en,
            "duration": round(en - st, 2),
            "category": str(item.get("category", "منوع")).strip(),
            "virality_score": int(item.get("virality_score", 7)),
            "reason": str(item.get("reason", "")).strip(),
            "metrics": metrics,
            "rating": "pending",
            "note": "",
        })

    return valid_clips


def generate_clip_srt(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    out_srt: str,
    max_words: int = 3,
    max_chars: int = 24,
    pause_threshold: float = 0.45,
) -> int:
    clip_dur = clip_end - clip_start
    words = []

    for seg in transcript.get("segments", []):
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue

        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                mid = (w["s"] + w["e"]) / 2.0
                if clip_start <= mid <= clip_end:
                    words.append({
                        "w": w["w"].strip(),
                        "s": max(0.0, w["s"] - clip_start),
                        "e": min(clip_dur, w["e"] - clip_start),
                    })
        else:
            tokens = seg["text"].split()
            if tokens:
                s0 = max(clip_start, seg["start"])
                e0 = min(clip_end, seg["end"])
                step = (e0 - s0) / len(tokens)
                for idx, tok in enumerate(tokens):
                    words.append({
                        "w": tok,
                        "s": max(0.0, (s0 + idx * step) - clip_start),
                        "e": min(clip_dur, (s0 + (idx + 1) * step) - clip_start),
                    })

    entries = []
    buf = []

    def flush():
        if not buf:
            return
        text = " ".join(x["w"] for x in buf).strip()
        st = buf[0]["s"]
        en = max(buf[-1]["e"], st + 0.25)
        en = min(en, clip_dur)
        if text:
            entries.append((st, en, text))
        buf.clear()

    for w in words:
        if not w["w"]:
            continue
        if buf:
            gap = w["s"] - buf[-1]["e"]
            projected_len = len(" ".join(x["w"] for x in buf)) + 1 + len(w["w"])
            if len(buf) >= max_words or projected_len > max_chars or gap > pause_threshold:
                flush()
        buf.append(w)
    flush()

    srt_blocks = []
    for idx, (st, en, text) in enumerate(entries, start=1):
        srt_blocks.append(f"{idx}\n{sec_to_srt_time(st)} --> {sec_to_srt_time(en)}\n{text}")

    with open(out_srt, "w", encoding="utf-8") as f:
        f.write("\n\n".join(srt_blocks) + ("\n" if srt_blocks else ""))

    return len(entries)


def cut_clip_video(vod_url: str, start: float, end: float, out_mp4: str):
    if os.path.exists(out_mp4):
        os.remove(out_mp4)

    section = f"*{sec_to_hms(start)}-{sec_to_hms(end)}"
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
        "--merge-output-format", "mp4",
        "--remux-video", "mp4",
        "--download-sections", section,
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--no-warnings",
        "-q",
        "-o", out_mp4,
        vod_url,
    ]
    subprocess.run(cmd, check=True, timeout=240)


def run_stage3(
    vod_dir: str,
    drive_root: str,
    api_key: str,
    num_clips: int = 4,
    min_sec: int = 18,
    max_sec: int = 60,
    words_per_line: int = 3,
    extra_clips: bool = False,
):
    meta_path = os.path.join(vod_dir, "meta.json")
    transcript_path = os.path.join(vod_dir, "transcript.json")
    manifest_path = os.path.join(vod_dir, "clips_manifest.json")
    clips_dir = os.path.join(vod_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

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
        log(f"ℹ️ يوجد بالفعل {len(existing_clips)} لقطة مقصوصة لهذا البث.")
        log("   إذا تبي لقطات زيادة من نفس البث، فعّل خيار EXTRA_CLIPS = True وشغّل الخلية.")
        return existing_clips

    new_clips = select_clips(
        api_key=api_key,
        vod_dir=vod_dir,
        drive_root=drive_root,
        num_clips=num_clips,
        min_sec=min_sec,
        max_sec=max_sec,
        existing_clips=existing_clips,
    )

    if not new_clips:
        log("⚠️ ما لقى الذكاء الاصطناعي لقطات جديدة مطابقة للشروط في هذا الجزء.")
        return existing_clips

    start_index = len(existing_clips) + 1
    log(f"\n🎬 تم اختيار {len(new_clips)} لقطة! جاري القصّ الجراحي وتوليد ملفات الـ .srt المنفصلة...\n")

    for offset, clip in enumerate(new_clips):
        clip_num = start_index + offset
        safe_title = sanitize_filename(clip["title"])
        base_name = f"clip_{clip_num:02d}_{safe_title}"
        mp4_path = os.path.join(clips_dir, f"{base_name}.mp4")
        srt_path = os.path.join(clips_dir, f"{base_name}.srt")

        log(
            f"✂️ [{clip_num:02d}] {clip['title']} "
            f"({sec_to_hms(clip['start'])} → {sec_to_hms(clip['end'])} | {clip['duration']}ث | تقييم {clip['virality_score']}/10)"
        )
        log(f"    💡 السبب: {clip['reason']}")
        log("    ⏳ جاري تحميل وقصّ الفيديو من تويتش (تاخذ 15-40 ثانية)...")

        try:
            cut_clip_video(vod_url, clip["start"], clip["end"], mp4_path)
            sub_count = generate_clip_srt(
                transcript,
                clip["start"],
                clip["end"],
                srt_path,
                max_words=words_per_line,
            )
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            clip["id"] = clip_num
            clip["video_path"] = mp4_path
            clip["srt_path"] = srt_path
            existing_clips.append(clip)
            log(f"    ✅ جاهز! الفيديو ({size_mb:.1f} MB) + ملف سبتايتل منفصل ({sub_count} سطر)\n")
        except Exception as e:
            log(f"    ❌ فشل قص اللقطة رقم {clip_num}: {e}\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(existing_clips, f, ensure_ascii=False, indent=2)

    log(f"🎉 خلصنا! كل اللقطات وملفات الـ .srt محفوظة في Google Drive داخل:\n   {clips_dir}")
    return existing_clips
