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


def _prepare_checkpoint(vod_dir: str, vod_id: str, vod_url: str, test_minutes):
    meta_path = os.path.join(vod_dir, "meta.json")
    old_meta = None
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            old_meta = json.load(f)

    if old_meta and old_meta.get("test_minutes") != test_minutes:
        for fname in ["chat.json", "audio.mp3", "hype_timeline.json", "transcript.json"]:
            fpath = os.path.join(vod_dir, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        print("🔄 تغيّرت مدة البث المطلوبة — تم تصفير الملفات القديمة للبدء على نظافة")

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
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


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
