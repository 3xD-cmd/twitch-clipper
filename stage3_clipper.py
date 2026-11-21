import concurrent.futures
import json
import os
import re
import shutil
import subprocess
import time
from google import genai
from google.genai import types

PREFERRED_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-latest",
    "gemini-flash-lite-latest",
]

EXCLUDED_MODEL_KEYWORDS = ("tts", "image", "audio", "embedding", "live", "veo", "imagen", "lyria")


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
            parts.append(f"### قواعد ذوق الممنتج المتعلَّمة (التزم بها بدقة عالية):\n{rules}")

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
            parts.append("### أمثلة لقطات أعجبت الممنتج مؤخراً (جب مثلها):\n" + "\n".join(lines))
        if disliked:
            lines = [
                f"- 👎 {x.get('title')} ({x.get('category', '')} | {x.get('duration', '')}ث) — سبب الرفض: {x.get('note', 'ما تنفع')}"
                for x in disliked
            ]
            parts.append("### أمثلة لقطات رفضها الممنتج (تجنّب تكرار مثلها):\n" + "\n".join(lines))

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

    for seg in transcript.get("segments", []):
        st = seg["start"]
        en = seg["end"]
        if en < start_sec:
            continue
        if end_sec is not None and st > end_sec:
            break

        bucket = int(st // 10)
        h_info = hype_by_bucket.get(bucket, {})
        tags = []
        audio_lvl = h_info.get("audio", 0)
        if audio_lvl >= 0.85:
            tags.append("🔥صراخ_أو_ضحك_عالي")
        elif audio_lvl >= 0.65:
            tags.append("⚡نبرة_حماس_أو_توتر")
        if h_info.get("chat_msgs", 0) >= 3:
            tags.append(f"💬شات({h_info['chat_msgs']})")

        tag_str = f" [{' '.join(tags)}]" if tags else ""
        lines.append(f"[{st:.1f}s - {en:.1f}s]{tag_str} {seg['text']}")

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
            low = name.lower()
            if "flash" not in low:
                continue
            if any(bad in low for bad in EXCLUDED_MODEL_KEYWORDS):
                continue
            if name not in PREFERRED_MODELS:
                found.append(name)
    except Exception:
        pass
    return found[:3]


def _request_once(client, model_name: str, prompt: str, json_mode: bool = True) -> str:
    cfg = types.GenerateContentConfig(temperature=0.35)
    if json_mode:
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.35,
        )
    resp = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=cfg,
    )
    return (resp.text or "").strip()


def call_gemini_text(api_key: str, prompt: str, json_mode: bool = False, timeout_sec: int = 90) -> str:
    client = genai.Client(api_key=api_key)
    candidates = list(PREFERRED_MODELS) + _discover_extra_flash_models(client)
    last_err = None

    for model_name in candidates:
        t0 = time.time()
        log(f"🧠 جاري الاتصال بـ ({model_name})...")
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = pool.submit(_request_once, client, model_name, prompt, json_mode)
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


def call_gemini_json(api_key: str, prompt: str, timeout_sec: int = 90) -> dict:
    raw = call_gemini_text(api_key, prompt, json_mode=True, timeout_sec=timeout_sec)
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


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


def extract_clip_excerpt(transcript: dict, start: float, end: float, max_chars: int = 450) -> str:
    texts = [
        seg["text"].strip()
        for seg in transcript.get("segments", [])
        if seg["end"] >= start and seg["start"] <= end and seg.get("text")
    ]
    joined = " ".join(texts)
    return (joined[:max_chars] + "...") if len(joined) > max_chars else joined


def _select_clips_for_window(
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
            "\n### لقطات تم قصّها مسبقاً (ممنوع تكرارها أو التداخل معها — هات لقطات جديدة مختلفة فقط):\n"
            + "\n".join(ranges)
        )

    prompt = f"""أنت مخرج وممنتج محتوى محترف متخصص في بثوث تويتش العربية (قيمينق وسوالف) لتحويلها إلى مقاطع TikTok و Reels فيرال.

أمامك نص جزء من البث (من {sec_to_hms(window_start)} إلى {sec_to_hms(window_end)}) مع التوقيت بالثواني، وعلامات نبرة الصوت (🔥صراخ_أو_ضحك_عالي، ⚡نبرة_حماس_أو_توتر) وتفاعل الشات (💬شات).

### مهمتك وحريتك الكاملة:
1. **لا يوجد عدد محدد للقطات ولا مدة محددة:** أنت حر تماماً! استخرج فقط اللقطات التي تستاهل فعلاً النشر — الجودة والضحك والسالفة القوية أهم من العدد، وممنوع الحشو نهائياً (إذا الجزء هذا هادي وما فيه شي يستاهل، رجّع قائمة فاضية).
2. **السياق الكامل (Full Context):** خذ اللقطة كاملة من بداية الموقف/السالفة حتى نهاية القفلة أو الضحكة أو ردة الفعل:
   - لو لحظة ضحك أو صراخ أو خوف سريعة: ممكن 12 إلى 30 ثانية.
   - لو سالفة حلوة أو موقف متصاعد: عادي جداً تمتد لدقيقة أو دقيقتين أو أكثر طالما ممتعة.
   - ممنوع تبدأ من نص جملة أو تنهيها قبل ما تخلص ردة الفعل!
3. **أسلوب الستريمر الخاص:**
   - ركّز على: **الضحك، الصراخ، التوتر والخوف، العفوية، الطقطقة مع الشات، أو السوالف الحلوة**.
   - **موضوع التشتت:** الستريمر تتشتت بسرعة وتدخل أكثر من موضوع في بعض؛ لو فتحت سالفة وتشتتت ثواني بسيطة على الشات أو اللعبة ثم رجعت كملت سالفتها، لا تقطع السالفة قبل ما تكمل! أما لو انتقلت من سالفة حلوة لموضوع ثاني مختلف ورهيب، فافصل كل سالفة بسياقها الكامل.
4. **تجنّب:** اللعب الصامت، أو كلام اللعبة الإنجليزي فقط بدون تفاعل ممتع من الستريمر.
5. ممنوع تداخل الأوقات بين اللقطات.
{memory_ctx}
{exclude_text}

أعد النتيجة بصيغة JSON فقط بهذا الهيكل الدقيق:
{{
  "clips": [
    {{
      "title": "عنوان عربي قصير وجذاب (2 إلى 5 كلمات)",
      "start": 12.5,
      "end": 58.0,
      "category": "ضحك أو صراخ_وتوتر أو سالفة أو تفاعل_شات أو قيمينق",
      "virality_score": 9,
      "reason": "ليش اللقطة رهيبة ووش السياق والقفلة فيها"
    }}
  ]
}}

=== نص هذا الجزء من البث ===
{digest}
"""
    result = call_gemini_json(api_key, prompt)
    return result.get("clips", [])


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
        log("🧠 تم تحميل ذاكرة ذوقك وقواعد التقييم السابقة بنجاح!")

    if duration <= 2700:
        windows = [(0.0, duration)]
    else:
        chunk_len = 2100.0
        overlap = 120.0
        windows = []
        cur = 0.0
        while cur < duration:
            end_w = min(duration, cur + chunk_len)
            windows.append((cur, end_w))
            if end_w >= duration:
                break
            cur = end_w - overlap
        log(f"📚 البث طويل — تم تقسيمه ذكياً إلى {len(windows)} فصول متداخلة عشان ما تفوت ولا لقطة!")

    existing_ranges = [(c["start"], c["end"]) for c in (existing_clips or [])]
    running_existing = list(existing_clips or [])
    valid_clips = []

    for idx, (w_start, w_end) in enumerate(windows, start=1):
        if len(windows) > 1:
            log(f"\n🔎 تحليل الفصل {idx}/{len(windows)} ({sec_to_hms(w_start)} → {sec_to_hms(w_end)})...")
        digest = build_stream_digest(transcript, hype_timeline, chat_messages, w_start, w_end)
        if len(digest.strip()) < 80:
            continue

        try:
            raw_clips = _select_clips_for_window(
                api_key=api_key,
                digest=digest,
                window_start=w_start,
                window_end=w_end,
                memory_ctx=memory_ctx,
                existing_clips=running_existing,
            )
        except Exception as e:
            log(f"⚠️ تعذر تحليل الفصل {idx}: {e} — بنكمل باقي الفصول")
            continue

        for item in raw_clips:
            try:
                st = max(0.0, round(float(item["start"]), 2))
                en = min(duration, round(float(item["end"]), 2))
            except (KeyError, TypeError, ValueError):
                continue

            if en - st < 6.0:
                continue

            overlaps = any(not (en <= a or st >= b) for a, b in existing_ranges)
            if overlaps:
                continue

            existing_ranges.append((st, en))
            metrics = compute_clip_metrics(st, en, hype_timeline)
            clip_obj = {
                "title": str(item.get("title", "لقطة")).strip(),
                "start": st,
                "end": en,
                "duration": round(en - st, 2),
                "category": str(item.get("category", "منوع")).strip(),
                "virality_score": int(item.get("virality_score", 8)),
                "reason": str(item.get("reason", "")).strip(),
                "excerpt": extract_clip_excerpt(transcript, st, en),
                "metrics": metrics,
                "rating": "pending",
                "note": "",
            }
            valid_clips.append(clip_obj)
            running_existing.append(clip_obj)

        if idx < len(windows):
            time.sleep(2)

    valid_clips.sort(key=lambda x: x["start"])
    return valid_clips


def _compute_local_wps(words: list, idx: int, window_sec: float = 2.5) -> float:
    center = (words[idx]["s"] + words[idx]["e"]) / 2.0
    w_start = center - window_sec
    w_end = center + window_sec
    nearby = [w for w in words if w["e"] >= w_start and w["s"] <= w_end]
    if len(nearby) <= 1:
        return 1.8
    span = max(0.6, nearby[-1]["e"] - nearby[0]["s"])
    return len(nearby) / span


def generate_clip_srt(
    transcript: dict,
    clip_start: float,
    clip_end: float,
    out_srt: str,
) -> tuple:
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
                    txt = w["w"].strip()
                    if txt:
                        words.append({
                            "w": txt,
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
    wps_samples = []

    def flush():
        if not buf:
            return
        text = " ".join(x["w"] for x in buf).strip()
        st = buf[0]["s"]
        en = max(buf[-1]["e"], st + 0.22)
        en = min(en, clip_dur)
        if text:
            entries.append((st, en, text))
        buf.clear()

    for i, w in enumerate(words):
        tok = w["w"]
        wps = _compute_local_wps(words, i)
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


def cut_clip_video(stream_urls: list, start: float, end: float, out_mp4: str):
    duration = max(1.0, end - start)
    local_tmp = os.path.join("/tmp", os.path.basename(out_mp4))
    if os.path.exists(local_tmp):
        os.remove(local_tmp)

    input_args = []
    for u in stream_urls[:2]:
        input_args += ["-ss", f"{start:.2f}", "-i", u]

    encoders = [
        ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "21"],
        ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"],
    ]

    last_err = ""
    success = False
    for enc in encoders:
        cmd = (
            ["ffmpeg", "-y", "-nostdin", "-v", "error"]
            + input_args
            + ["-t", f"{duration:.2f}"]
            + enc
            + ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", local_tmp]
        )
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=240)
        if res.returncode == 0 and os.path.exists(local_tmp) and os.path.getsize(local_tmp) > 10_000:
            success = True
            break
        last_err = res.stderr.strip().splitlines()[-1] if res.stderr.strip() else f"exit code {res.returncode}"

    if not success:
        raise RuntimeError(f"فشل FFmpeg أثناء القص: {last_err}")

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
):
    meta_path = os.path.join(vod_dir, "meta.json")
    transcript_path = os.path.join(vod_dir, "transcript.json")
    manifest_path = os.path.join(vod_dir, "clips_manifest.json")
    clips_dir = os.path.join(vod_dir, "clips")
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
        log("   إذا تبي لقطات إضافية من نفس البث، فعّل خيار EXTRA_CLIPS وشغّل الخلية.")
        return existing_clips

    new_clips = select_clips(
        api_key=api_key,
        vod_dir=vod_dir,
        drive_root=drive_root,
        existing_clips=existing_clips,
    )

    if not new_clips:
        log("⚠️ ما لقى الذكاء الاصطناعي لقطات جديدة تستاهل القص.")
        return existing_clips

    stream_urls = resolve_stream_urls(vod_url)
    start_index = len(existing_clips) + 1
    log(f"\n🎬 تم اختيار {len(new_clips)} لقطة! جاري القصّ السريع وتوليد ملفات الـ .srt الديناميكية...\n")

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

        t0 = time.time()
        try:
            cut_clip_video(stream_urls, clip["start"], clip["end"], mp4_path)
            sub_count, avg_wps = generate_clip_srt(transcript, clip["start"], clip["end"], srt_path)
            size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
            clip["id"] = clip_num
            clip["video_path"] = mp4_path
            clip["srt_path"] = srt_path
            clip["avg_wps"] = avg_wps
            existing_clips.append(clip)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(existing_clips, f, ensure_ascii=False, indent=2)
            log(
                f"    ✅ جاهز خلال {time.time() - t0:.1f}ث! ({size_mb:.1f} MB | "
                f"سبتايتل {sub_count} سطر | سرعة الكلام {avg_wps} كلمة/ث)\n"
            )
        except Exception as e:
            log(f"    ❌ فشل قص اللقطة رقم {clip_num}: {e}\n")

    log(f"🎉 خلصنا! كل اللقطات + ملفات الـ .srt محفوظة داخل:\n   {clips_dir}")
    return existing_clips


# === END OF stage3_clipper.py ===
