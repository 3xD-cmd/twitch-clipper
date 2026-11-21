import ctypes
import gc
import glob
import json
import os
import re
import shutil
import site
import subprocess
import sys
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


def log(msg: str):
    print(msg, flush=True)


def preload_cuda_libs():
    dirs = set()
    roots = list(site.getsitepackages()) + [p for p in sys.path if "packages" in p]
    for root in roots:
        if os.path.isdir(root):
            for d in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
                if os.path.isdir(d):
                    dirs.add(d)

    priority = [
        "libcublasLt.so.12",
        "libcublas.so.12",
        "libcudnn.so.9",
        "libcudnn_ops.so.9",
        "libcudnn_cnn.so.9",
    ]
    for lib_name in priority:
        for d in sorted(dirs):
            path = os.path.join(d, lib_name)
            if os.path.exists(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break


def extract_vod_id(vod_url: str) -> str:
    match = re.search(r"/videos/(\d+)", vod_url)
    if not match:
        raise ValueError("❌ الرابط غير صحيح! لازم يكون رابط بث تويتش مثل: https://www.twitch.tv/videos/123456789")
    return match.group(1)


def _normalize_test_minutes(test_minutes):
    if test_minutes is None or test_minutes == "":
        return None
    val = int(test_minutes)
    return val if val > 0 else None


def _prepare_checkpoint(vod_dir: str, vod_id: str, vod_url: str, test_minutes):
    meta_path = os.path.join(vod_dir, "meta.json")
    old_meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)

    if old_meta and old_meta.get("test_minutes") != test_minutes:
        for fname in ["chat.json", "audio.mp3", "hype_timeline.json", "transcript.json", "clips_manifest.json"]:
            fpath = os.path.join(vod_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        clips_dir = os.path.join(vod_dir, "clips")
        if os.path.isdir(clips_dir):
            shutil.rmtree(clips_dir, ignore_errors=True)
        log("🔄 تم التحويل من وضع التجربة إلى البث الكامل — تم تصفير ملفات التجربة للبدء على نظافة")

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"vod_id": vod_id, "vod_url": vod_url, "test_minutes": test_minutes},
            f,
            ensure_ascii=False,
        )


def load_audio_ffmpeg(audio_path: str, sr: int = 16000) -> np.ndarray:
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", audio_path,
        "-ac", "1", "-ar", str(sr),
        "-f", "s16le", "-",
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return np.frombuffer(res.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def download_chat(vod_id: str, out_path: str, test_minutes=None):
    log(f"💬 جاري سحب شات البث ({vod_id})...")
    headers = {
        "Client-ID": TWITCH_CLIENT_ID,
        "Content-Type": "application/json",
    }
    max_sec = test_minutes * 60 if test_minutes else None
    max_pages = 40 if test_minutes else 2500
    time_budget = 30 if test_minutes else 180

    messages = []
    seen_cursors = set()
    cursor = None
    pages = 0
    t_start = time.time()

    while pages < max_pages:
        if time.time() - t_start > time_budget:
            log(f"⏱️ انتهت مهلة سحب الشات ({time_budget} ثانية) — بنكمل باللي انسحب ({len(messages)} رسالة)")
            break

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
            resp = requests.post(TWITCH_GQL_URL, headers=headers, json=payload, timeout=(5, 12))
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
            log(f"⚠️ تعذر إكمال سحب الشات ({e}) — بنكمل بالصوت والكلام.")
            break

        reached_limit = False
        last_offset = 0
        for edge in edges:
            node = edge.get("node", {})
            offset = node.get("contentOffsetSeconds", 0)
            last_offset = offset
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
        if pages % 20 == 0:
            log(f"   ... تم سحب {len(messages)} رسالة (وصلنا للدقيقة {last_offset / 60:.1f})")

        page_info = comments.get("pageInfo", {})
        if reached_limit or not page_info.get("hasNextPage"):
            break

        next_cursor = edges[-1].get("cursor")
        if not next_cursor or next_cursor in seen_cursors:
            break
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    log(f"✅ تم حفظ الشات ({len(messages)} رسالة)")
    return messages


def download_audio(vod_url: str, out_mp3: str, test_minutes=None):
    log("🎧 جاري تحميل صوت البث فقط (بدون فيديو لتوفير الوقت والمساحة)...")
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
        log(f"   🧪 وضع التجربة السريعة: أول {test_minutes} دقائق فقط")
    else:
        log("   🚀 وضع البث الكامل — جاري سحب صوت البث من بدايته لنهايته...")

    cmd.append(vod_url)
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(out_mp3) / (1024 * 1024)
    log(f"✅ تم تحميل الصوت بنجاح ({size_mb:.1f} MB)")


def analyze_hype(audio_path: str, chat_path: str, out_path: str, window_sec: int = 10):
    log("📊 جاري تحليل رادار الحماس (ارتفاع الصوت + انفجار الشات)...")

    sr = 8000
    samples = load_audio_ffmpeg(audio_path, sr=sr)
    total_sec = len(samples) / sr
    num_windows = max(1, int(np.ceil(total_sec / window_sec)))

    audio_rms = np.zeros(num_windows, dtype=np.float32)
    step = sr * window_sec
    for i in range(num_windows):
        chunk = samples[i * step : (i + 1) * step]
        if len(chunk) > 0:
            audio_rms[i] = float(np.sqrt(np.mean(chunk ** 2)))

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
    log(f"✅ تم تحليل {num_windows} نافذة زمنية ({total_sec / 60:.1f} دقيقة)")
    return timeline


def transcribe_audio(audio_path: str, out_path: str):
    log("🎙️ جاري تجهيز الصوت ومكتبات CUDA وتحميل موديل Faster-Whisper (large-v3-turbo)...")
    t0 = time.time()

    preload_cuda_libs()
    audio_array = load_audio_ffmpeg(audio_path, sr=16000)
    total_dur = len(audio_array) / 16000.0
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
    log(f"🚀 بدأ التفريغ العربي على كرت T4 ({total_dur / 60:.1f} دقيقة)...")

    segments_gen, _ = model.transcribe(
        audio_array,
        language="ar",
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    segments_data = []
    log_step = 10 if total_dur > 3600 else 5
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
        if cur_min >= last_log_min + log_step:
            last_log_min = cur_min
            log(f"   ... تم تفريغ {cur_min} دقيقة من أصل {total_dur / 60:.1f} دقيقة")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"duration": round(total_dur, 2), "segments": segments_data},
            f,
            ensure_ascii=False,
        )

    del model, audio_array
    gc.collect()

    elapsed = time.time() - t0
    log(f"✅ اكتمل التفريغ! ({len(segments_data)} جملة في {elapsed / 60:.1f} دقيقة)")
    return segments_data


def run_stage2(vod_url: str, drive_root: str, test_minutes=None):
    test_minutes = _normalize_test_minutes(test_minutes)
    vod_id = extract_vod_id(vod_url)
    vod_dir = os.path.join(drive_root, "vods", vod_id)
    os.makedirs(vod_dir, exist_ok=True)

    _prepare_checkpoint(vod_dir, vod_id, vod_url, test_minutes)

    chat_path = os.path.join(vod_dir, "chat.json")
    audio_path = os.path.join(vod_dir, "audio.mp3")
    hype_path = os.path.join(vod_dir, "hype_timeline.json")
    transcript_path = os.path.join(vod_dir, "transcript.json")

    if not os.path.exists(chat_path):
        download_chat(vod_id, chat_path, test_minutes=test_minutes)
    else:
        log("⏭️ الشات موجود مسبقاً على Drive — تم التخطي")

    if not os.path.exists(audio_path):
        download_audio(vod_url, audio_path, test_minutes=test_minutes)
    else:
        log("⏭️ الصوت موجود مسبقاً على Drive — تم التخطي")

    if not os.path.exists(hype_path):
        analyze_hype(audio_path, chat_path, hype_path)
    else:
        log("⏭️ رادار الحماس موجود مسبقاً على Drive — تم التخطي")

    if not os.path.exists(transcript_path):
        transcribe_audio(audio_path, transcript_path)
    else:
        log("⏭️ التفريغ النصي موجود مسبقاً على Drive — تم التخطي")

    with open(transcript_path, "r", encoding="utf-8") as f:
        tr = json.load(f)
    log(f"\n🎉 المرحلة 2 جاهزة! مدة البث المفرّغ: {tr['duration'] / 60:.1f} دقيقة ({len(tr['segments'])} جملة)")
    log(f"📁 المسار: {vod_dir}")
    return vod_dir
