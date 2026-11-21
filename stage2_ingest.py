import gc
import json
import os
import re
import subprocess
import time
import numpy as np
import requests
from faster_whisper import WhisperModel

TWITCH_GQL_URL = "https://gql.twitch.tv/gql"
TWITCH_CLIENT_ID = "kimne78kx3ncx6brgo4mv6wki5h1ko"
CHAT_QUERY_HASH = "b70a3591ff0f4e0313d126c6a1502d79a1c02baebb288227c582044aa76adf6a"

HYPE_PATTERNS = re.compile(
    r"(ه{3,}|😂|💀|😭|🔥|\?{2,}|؟{2,}|وش|اما|لاا+|كفو|موت|ضحك|lul|kekw|omegalul|lmao|clip)",
    re.IGNORECASE,
)


def extract_vod_id(vod_url: str) -> str:
    match = re.search(r"/videos/(\d+)", vod_url)
    if not match:
        raise ValueError("❌ الرابط غير صحيح! لازم يكون رابط بث تويتش مثل: https://www.twitch.tv/videos/123456789")
    return match.group(1)


def _needs_rebuild(file_path: str, meta_path: str, test_minutes) -> bool:
    if not os.path.exists(file_path):
        return True
    if not os.path.exists(meta_path):
        return True
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    # إذا الملف القديم كان تجريبي (جزئي) والحين نبي البث كامل، نعيد تحميله
    if meta.get("test_minutes") != test_minutes:
        return True
    return False


def download_chat(vod_id: str, out_path: str, test_minutes=None):
    print(f"💬 جاري سحب شات البث ({vod_id})...")
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Content-Type": "application/json",
    }
    max_sec = test_minutes * 60 if test_minutes else None
    messages = []
    cursor = None
    pages = 0

    while True:
        variables = {"videoID": str(vod_id)}
        if cursor:
            variables["cursor"] = cursor
        else:
            variables["contentOffsetSeconds"] = 0

        payload = [{
            "operationName": "VideoCommentsByOffsetOrCursor",
            "variables": variables,
            "extensions": {
                "persistedQuery": {
                    "version": 1,
                    "sha256Hash": CHAT_QUERY_HASH,
                }
            },
        }]

        try:
            resp = requests.post(TWITCH_GQL_URL, headers=headers, json=payload, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            video = data[0].get("data", {}).get("video")
            if not video or not video.get("comments"):
                break
            comments = video["comments"]
            edges = comments.get("edges", [])
            if not edges:
                break
        except Exception as e:
            print(f"⚠️ تعذر إكمال سحب الشات ({e}) — بنكمل باللي انسحب + تحليل الصوت والكلام.")
            break

        reached_limit = False
        for edge in edges:
            node = edge.get("node", {})
            offset = node.get("contentOffsetSeconds", 0)
            if max_sec is not None and offset > max_sec:
                reached_limit = True
                break

            fragments = node.get("message", {}).get("fragments", [])
            text = "".join(f.get("text", "") for f in fragments).strip()
            has_emote = any(f.get("emote") is not None for f in fragments)
            user = (node.get("commenter") or {}).get("displayName", "unknown")

            if text:
                messages.append({
                    "t": offset,
                    "u": user,
                    "m": text,
                    "e": has_emote,
                })

        pages += 1
        if pages % 25 == 0:
            print(f"   ... تم سحب {len(messages)} رسالة حتى الآن")

        page_info = comments.get("pageInfo", {})
        if reached_limit or not page_info.get("hasNextPage"):
            break

        cursor = edges[-1].get("cursor")
        if not cursor:
            break

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    print(f"✅ تم حفظ الشات ({len(messages)} رسالة)")
    return messages


def download_audio(vod_url: str, out_mp3: str, test_minutes=None):
    print("🎧 جاري تحميل صوت البث فقط (بدون فيديو لتوفير الوقت والمساحة)...")
    base_template = out_mp3.rsplit(".", 1)[0] + ".%(ext)s"
    if os.path.exists(out_mp3):
        os.remove(out_mp3)

    cmd = [
        "yt-dlp",
        "-f", "bestaudio/best",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "--no-playlist",
        "-o", base_template,
    ]
    if test_minutes:
        end_sec = int(test_minutes * 60)
        h, rem = divmod(end_sec, 3600)
        m, s = divmod(rem, 60)
        cmd += ["--download-sections", f"*00:00:00-{h:02d}:{m:02d}:{s:02d}"]
        print(f"   🧪 وضع التجربة السريعة: أول {test_minutes} دقائق فقط")

    cmd.append(vod_url)
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(out_mp3) / (1024 * 1024)
    print(f"✅ تم تحميل الصوت بنجاح ({size_mb:.1f} MB)")


def analyze_hype(audio_path: str, chat_path: str, out_path: str, window_sec: int = 10):
    print("📊 جاري تحليل رادار الحماس (ارتفاع الصوت + انفجار الشات)...")

    # 1. استخراج الصوت كـ PCM 8kHz mono لحساب شدة الصوت بسرعة خارقة وبأقل رام
    sr = 8000
    cmd = [
        "ffmpeg", "-v", "error", "-i", audio_path,
        "-ac", "1", "-ar", str(sr), "-f", "s16le", "-"
    ]
    raw = subprocess.check_output(cmd)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total_sec = len(samples) / sr
    num_windows = max(1, int(np.ceil(total_sec / window_sec)))

    audio_rms = np.zeros(num_windows, dtype=np.float32)
    step = sr * window_sec
    for i in range(num_windows):
        chunk = samples[i * step : (i + 1) * step]
        if len(chunk) > 0:
            audio_rms[i] = float(np.sqrt(np.mean(chunk ** 2)))

    # 2. تحليل الشات لكل نافذة (10 ثوانٍ)
    with open(chat_path, "r", encoding="utf-8") as f:
        messages = json.load(f)

    chat_count = np.zeros(num_windows, dtype=np.float32)
    chat_hype = np.zeros(num_windows, dtype=np.float32)
    for msg in messages:
        idx = int(msg["t"] // window_sec)
        if 0 <= idx < num_windows:
            chat_count[idx] += 1
            bonus = 1.0
            if HYPE_PATTERNS.search(msg["m"]):
                bonus += 1.5
            if msg.get("e"):
                bonus += 0.5
            chat_hype[idx] += bonus

    # 3. تطبيع الدرجات (Normalization) ودمجها
    def norm(arr):
        mx = np.percentile(arr, 98) if len(arr) > 5 else np.max(arr)
        return np.clip(arr / mx, 0, 1.5) if mx > 0 else np.zeros_like(arr)

    a_norm = norm(audio_rms)
    c_norm = norm(chat_hype)
    has_chat = len(messages) >= 20

    timeline = []
    for i in range(num_windows):
        score = (0.55 * c_norm[i] + 0.45 * a_norm[i]) if has_chat else float(a_norm[i])
        timeline.append({
            "start": i * window_sec,
            "end": (i + 1) * window_sec,
            "audio": round(float(a_norm[i]), 3),
            "chat_msgs": int(chat_count[i]),
            "chat_score": round(float(c_norm[i]), 3),
            "hype": round(float(score), 3),
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, ensure_ascii=False)
    print(f"✅ تم تحليل {num_windows} نافذة زمنية ({total_sec/60:.1f} دقيقة)")
    return timeline


def transcribe_audio(audio_path: str, out_path: str):
    print("🎙️ جاري تفريغ الكلام العربي عبر Faster-Whisper (large-v3-turbo) على T4 GPU...")
    t0 = time.time()
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    segments_gen, info = model.transcribe(
        audio_path,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    segments_data = []
    last_log_min = 0

    for seg in segments_gen:
        words = []
        for w in (seg.words or []):
            txt = w.word.strip()
            if txt:
                words.append({
                    "w": txt,
                    "s": round(w.start, 2),
                    "e": round(w.end, 2),
                })

        text = seg.text.strip()
        if text:
            segments_data.append({
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": text,
                "words": words,
            })

        cur_min = int(seg.end // 60)
        if cur_min >= last_log_min + 5:
            last_log_min = cur_min
            print(f"   ... تم تفريغ {cur_min} دقيقة من أصل {info.duration/60:.1f} دقيقة")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"duration": round(info.duration, 2), "segments": segments_data},
            f,
            ensure_ascii=False,
        )

    del model
    gc.collect()

    elapsed = time.time() - t0
    print(f"✅ اكتمل التفريغ! ({len(segments_data)} جملة في {elapsed:.1f} ثانية)")
    return segments_data


def run_stage2(vod_url: str, drive_root: str, test_minutes=None):
    vod_id = extract_vod_id(vod_url)
    vod_dir = os.path.join(drive_root, "vods", vod_id)
    os.makedirs(vod_dir, exist_ok=True)

    meta_path = os.path.join(vod_dir, "meta.json")
    chat_path = os.path.join(vod_dir, "chat.json")
    audio_path = os.path.join(vod_dir, "audio.mp3")
    hype_path = os.path.join(vod_dir, "hype_timeline.json")
    transcript_path = os.path.join(vod_dir, "transcript.json")

    rebuild = _needs_rebuild(transcript_path, meta_path, test_minutes)

    if rebuild or not os.path.exists(chat_path):
        download_chat(vod_id, chat_path, test_minutes=test_minutes)
    else:
        print("⏭️ الشات موجود مسبقاً على Drive — تم التخطي")

    if rebuild or not os.path.exists(audio_path):
        download_audio(vod_url, audio_path, test_minutes=test_minutes)
    else:
        print("⏭️ الصوت موجود مسبقاً على Drive — تم التخطي")

    if rebuild or not os.path.exists(hype_path):
        analyze_hype(audio_path, chat_path, hype_path)
    else:
        print("⏭️ رادار الحماس موجود مسبقاً على Drive — تم التخطي")

    if rebuild or not os.path.exists(transcript_path):
        transcribe_audio(audio_path, transcript_path)
    else:
        print("⏭️ التفريغ النصي موجود مسبقاً على Drive — تم التخطي")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"vod_id": vod_id, "vod_url": vod_url, "test_minutes": test_minutes}, f, ensure_ascii=False)

    # طباعة عينة من التفريغ للتأكد من جودة العربي
    with open(transcript_path, "r", encoding="utf-8") as f:
        tr = json.load(f)
    print("\n📝 عينة من أول 5 جمل مفرّغة:")
    for seg in tr["segments"][:5]:
        print(f"   [{seg['start']:>6.1f}s - {seg['end']:>6.1f}s] {seg['text']}")

    print(f"\n🎉 المرحلة 2 جاهزة ومحفوظة في: {vod_dir}")
    return vod_dir
