"""
setup_tools.py
Colab environment check for TwitchClipper (v3). Run after pip install:
    python setup_tools.py                      (normal check, no network calls to Twitch)
    python setup_tools.py --vod <VOD URL>      (adds a live yt-dlp -J test on that VOD)
    python setup_tools.py --skip-cuda          (skips the CUDA 12 ldconfig step)

What it does
  * CUDA 12 runtime fix (kept from the battle tested version): registers the pip installed
    nvidia/*/lib folders with ldconfig so faster-whisper never fails with
    "libcublas.so.12 not found", then verifies that libcublas.so.12 loads (and reports cuDNN).
  * Versions of every tool and package the pipeline uses (FFmpeg, ffprobe, yt-dlp,
    faster-whisper, transformers / torch for the AST tagger, google-genai, gradio, ...).
  * GPU check that is informative, not alarming: on the CPU runtime (Stage 3 and Stage 4)
    no GPU is expected; on the T4 runtime it confirms the card and tests NVENC.
  * v3 runtime capability checks:
      - Gradio 6 custom gr.HTML components (html_template, css_template, js_on_load) for the
        Studio timeline editor (studio_timeline.timeline_supported(), with a signature check
        as a fallback). Without it the Studio uses the v2 controls.
      - google.genai types.VideoMetadata (video fps for the Critic and Golden clips) and
        types.MediaResolution (low media resolution). Without them the gateway still works
        and falls back to its defaults.
      - yt-dlp version age and JSON dump support (-J), offline by default.
  * Free disk space on /tmp and on Google Drive, plus the memory folder (rules, golden clips).
"""

import argparse
import glob
import inspect
import json
import os
import re
import shutil
import site
import subprocess
import sys
import time
from datetime import date

try:
    from importlib import metadata as _metadata
except ImportError:  # pragma: no cover
    _metadata = None


PACKAGES = (
    ("yt-dlp", "yt-dlp"),
    ("faster-whisper", "faster-whisper"),
    ("ctranslate2", "ctranslate2"),
    ("google-genai", "google-genai"),
    ("pydantic", "pydantic"),
    ("gradio", "gradio"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("transformers", "transformers (مصنف الأصوات AST)"),
    ("torch", "torch (مصنف الأصوات AST)"),
    ("nvidia-cublas-cu12", "nvidia-cublas-cu12"),
    ("nvidia-cudnn-cu12", "nvidia-cudnn-cu12"),
)
OPTIONAL_MARK = "AST"

DRIVE_ROOT = "/content/drive/MyDrive/TwitchClipper"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
GOLDEN_EXTS = (".mp4", ".mov", ".m4v", ".webm", ".mkv")
YTDLP_OLD_DAYS = 60


# ============================================================================
# Tools and packages
# ============================================================================

def check_tool(cmd, name):
    """Print the first line of a tool's version output."""
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30)
        first_line = out.strip().splitlines()[0] if out.strip() else "متوفر"
        print(f"✅ {name}: {first_line}")
        return True
    except Exception as e:
        print(f"❌ مشكلة في {name}: {e}")
        return False


def setup_cuda_libs():
    """Battle tested CUDA 12 fix: ldconfig on the pip nvidia libs, then a cuBLAS 12 load test."""
    dirs = set()
    roots = list(site.getsitepackages()) + [p for p in sys.path if "packages" in p]
    for root in roots:
        if os.path.isdir(root):
            for d in glob.glob(os.path.join(root, "nvidia", "*", "lib")):
                if os.path.isdir(d):
                    dirs.add(d)

    if dirs:
        try:
            with open("/etc/ld.so.conf.d/nvidia_pip.conf", "w") as f:
                f.write("\n".join(sorted(dirs)) + "\n")
            subprocess.run(["ldconfig"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ تنبيه ldconfig: {e}")

    ok = True
    try:
        subprocess.run(
            [sys.executable, "-c", "import ctypes; ctypes.CDLL('libcublas.so.12', mode=ctypes.RTLD_GLOBAL)"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("✅ CUDA cuBLAS 12: جاهز وشغال")
    except Exception as e:
        ok = False
        print(f"❌ مشكلة في CUDA cuBLAS 12: {e}")

    cudnn_found = None
    for name in ("libcudnn.so.9", "libcudnn.so.8"):
        res = subprocess.run(
            [sys.executable, "-c", f"import ctypes; ctypes.CDLL('{name}', mode=ctypes.RTLD_GLOBAL)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if res.returncode == 0:
            cudnn_found = name
            break
    if cudnn_found:
        print(f"✅ CUDA cuDNN: جاهز ({cudnn_found})")
    else:
        print("ℹ️ مكتبة cuDNN ما انحملت. المرحلة 2 تحملها بنفسها قبل Whisper، ولو فشلت تأكد إن nvidia-cudnn-cu12 مثبتة")
    return ok


def package_versions():
    print("📦 إصدارات المكتبات:")
    versions = {}
    for dist, label in PACKAGES:
        version = None
        if _metadata is not None:
            try:
                version = _metadata.version(dist)
            except Exception:
                version = None
        versions[dist] = version
        if version:
            print(f"   ✅ {label}: {version}")
        else:
            optional = OPTIONAL_MARK in label
            mark = "ℹ️" if optional else "❌"
            hint = "اختياري، المرحلة 2 تكمل بدونه" if optional else "غير مثبت، شغّل pip install مرة ثانية"
            print(f"   {mark} {label}: {hint}")
    return versions


def check_imports():
    """Import test for the packages each stage needs (a broken install often only shows here)."""
    print("🧩 فحص الاستيراد:")
    results = {}
    tests = (
        ("faster_whisper", "faster-whisper (المرحلة 2)", False),
        ("ctranslate2", "ctranslate2 (المرحلة 2)", False),
        ("google.genai", "google-genai (Gemini)", False),
        ("pydantic", "pydantic", False),
        ("gradio", "gradio (الاستوديو)", False),
        ("numpy", "numpy", False),
        ("requests", "requests", False),
        ("transformers", "transformers (مصنف AST)", True),
    )
    for module, label, optional in tests:
        try:
            __import__(module)
            results[module] = True
            print(f"   ✅ {label}")
        except Exception as e:
            results[module] = False
            mark = "ℹ️" if optional else "❌"
            hint = "اختياري" if optional else "يحتاج إعادة تثبيت"
            print(f"   {mark} {label}: {hint} ({str(e).splitlines()[0][:120] if str(e) else type(e).__name__})")
    if results.get("transformers"):
        try:
            from transformers import ASTForAudioClassification  # noqa: F401
            print("   ✅ مصنف الأصوات AST متوفر (يتحمل أول مرة في المرحلة 2)")
        except Exception:
            print("   ℹ️ نسخة transformers ما فيها AST، المرحلة 2 تستخدم التصنيف التقديري")
    return results


def check_gpu():
    """True when a usable NVIDIA GPU is present."""
    if shutil.which("nvidia-smi") is None:
        print("ℹ️ وضع المعالج (CPU): ما فيه كرت شاشة، وهذا طبيعي للمرحلة 3 والمرحلة 4. المرحلة 2 (التفريغ) تحتاج T4.")
        return False
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            stderr=subprocess.STDOUT, text=True, timeout=30,
        ).strip()
    except Exception as e:
        print(f"ℹ️ كرت الشاشة ما استجاب ({e}). إذا بتشغّل المرحلة 2 اختر T4 من Runtime")
        return False
    if not out:
        print("ℹ️ ما لقينا كرت شاشة نشط")
        return False
    print(f"✅ كرت الشاشة: {out.splitlines()[0]}")
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
        if n > 0:
            print(f"✅ ctranslate2 يشوف كرت الشاشة ({n})، Whisper بيشتغل على T4")
        else:
            print("⚠️ ctranslate2 ما شاف كرت الشاشة. أعد تشغيل خلية التجهيز بعد ما يخلص pip")
    except Exception:
        pass
    return True


def check_nvenc():
    try:
        res = subprocess.run(
            ["ffmpeg", "-hide_banner", "-v", "error", "-f", "lavfi", "-i", "color=black:s=320x240:d=0.3",
             "-c:v", "h264_nvenc", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=30,
        )
        if res.returncode == 0:
            print("✅ ترميز الفيديو بكرت الشاشة (h264_nvenc): شغال")
        else:
            print("ℹ️ ترميز h264_nvenc غير متاح، القص بيستخدم libx264 على المعالج تلقائياً")
    except Exception:
        print("ℹ️ ما قدرنا نختبر h264_nvenc، القص بيستخدم libx264 تلقائياً")


def check_disk():
    for path, label in (("/tmp", "مساحة Colab المؤقتة (/tmp)"), (DRIVE_ROOT, "Google Drive")):
        if not os.path.exists(path):
            continue
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)
            mark = "✅" if free_gb >= 5 else "⚠️"
            print(f"{mark} {label}: متبقي {free_gb:.1f} GB")
        except Exception:
            continue


# ============================================================================
# v3 capability checks
# ============================================================================

def check_gradio_timeline():
    """Gradio 6 custom gr.HTML (html_template, css_template, js_on_load) for the timeline editor."""
    try:
        import gradio as gr
    except Exception as e:
        print(f"❌ gradio ما انحمل: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}")
        return False
    version = str(getattr(gr, "__version__", "?"))
    supported = None
    if PROJECT_DIR not in sys.path:
        sys.path.insert(0, PROJECT_DIR)
    try:
        import studio_timeline
        supported = bool(studio_timeline.timeline_supported())
    except Exception:
        supported = None
    if supported is None:
        try:
            major = int(version.split(".")[0])
            params = inspect.signature(gr.HTML.__init__).parameters
            supported = major >= 6 and all(p in params for p in ("html_template", "css_template", "js_on_load"))
        except Exception:
            supported = False
    if supported:
        print(f"✅ Gradio {version}: المحرر الزمني في الاستوديو مدعوم (gr.HTML المخصص)")
    else:
        print(f"⚠️ Gradio {version}: المحرر الزمني غير مدعوم، الاستوديو بيستخدم التحكم العادي. "
              "للمحرر ثبّت gradio>=6.0,<7")
    return bool(supported)


def check_genai_types():
    """google.genai types used by the v3 gateway: VideoMetadata and MediaResolution."""
    try:
        from google.genai import types
    except Exception as e:
        print(f"❌ google-genai ما انحمل: {str(e).splitlines()[0][:120] if str(e) else type(e).__name__}")
        return {"video_metadata": False, "media_resolution": False}
    result = {"video_metadata": False, "media_resolution": False}
    try:
        part = types.Part(inline_data=types.Blob(data=b"\x00", mime_type="video/mp4"),
                          video_metadata=types.VideoMetadata(fps=2.0))
        result["video_metadata"] = getattr(getattr(part, "video_metadata", None), "fps", None) == 2.0
    except Exception:
        result["video_metadata"] = False
    try:
        low = types.MediaResolution.MEDIA_RESOLUTION_LOW
        cfg = types.GenerateContentConfig(media_resolution=low)
        result["media_resolution"] = getattr(cfg, "media_resolution", None) is not None
    except Exception:
        result["media_resolution"] = False
    if result["video_metadata"]:
        print("✅ Gemini VideoMetadata: نقدر نحدد عدد اللقطات بالثانية لفيديو الناقد واللقطات الذهبية")
    else:
        print("⚠️ Gemini VideoMetadata غير متوفر: الفيديو ينرسل بالإعداد الافتراضي (1 لقطة بالثانية). حدّث google-genai")
    if result["media_resolution"]:
        print("✅ Gemini MediaResolution: دقة الوسائط المنخفضة متاحة (توفير توكنات)")
    else:
        print("⚠️ Gemini MediaResolution غير متوفر: الطلبات تشتغل بالدقة الافتراضية وتستهلك توكنات أكثر. حدّث google-genai")
    return result


def _ytdlp_version_date(text):
    match = re.search(r"(\d{4})\.(\d{2})\.(\d{2})", text or "")
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def check_ytdlp(vod_url=None):
    """yt-dlp version, age and JSON dump support. A live -J test runs only when vod_url is given."""
    try:
        version = subprocess.check_output(["yt-dlp", "--version"], stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except Exception as e:
        print(f"❌ مشكلة في yt-dlp: {e}")
        return False
    built = _ytdlp_version_date(version)
    age = (date.today() - built).days if built else None
    if age is not None and age > YTDLP_OLD_DAYS:
        print(f"⚠️ yt-dlp {version}: عمر النسخة {age} يوم، تويتش يتغير كثير. شغّل: pip install -U yt-dlp")
    else:
        print(f"✅ yt-dlp {version}" + (f" (عمرها {age} يوم)" if age is not None else ""))
    try:
        help_text = subprocess.check_output(["yt-dlp", "--help"], stderr=subprocess.STDOUT, text=True, timeout=30)
        has_json = "--dump-single-json" in help_text or "-J," in help_text
    except Exception:
        has_json = False
    if has_json:
        print("✅ yt-dlp يدعم تفريغ JSON (-J) اللي نستخدمه لاستخراج الماستر والبروكسي بطلب واحد")
    else:
        print("⚠️ yt-dlp ما يدعم -J، استخراج الروابط بيرجع للطريقة القديمة -g")
    if not vod_url:
        return has_json
    print(f"🌐 اختبار مباشر على البث: {vod_url}")
    t0 = time.time()
    try:
        res = subprocess.run(["yt-dlp", "-J", "--no-playlist", "--no-warnings", vod_url],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("❌ yt-dlp تأخر أكثر من دقيقتين ورجع بدون نتيجة")
        return False
    if res.returncode != 0:
        err = (res.stderr or "").strip().splitlines()
        print(f"❌ yt-dlp فشل: {err[-1][:200] if err else 'سبب غير معروف'}")
        return False
    try:
        info = json.loads(res.stdout)
    except ValueError:
        print("❌ رد yt-dlp مو JSON صالح")
        return False
    formats = [f for f in info.get("formats") or [] if f.get("vcodec") not in (None, "none")]
    heights = sorted({int(f.get("height") or 0) for f in formats if f.get("height")}, reverse=True)
    duration = float(info.get("duration") or 0.0)
    print(f"✅ البث: «{str(info.get('title') or '')[:60]}» مدته {duration / 3600:.1f} ساعة، "
          f"الجودات: {', '.join(f'{h}p' for h in heights[:6]) or 'غير معروفة'} "
          f"(خلال {time.time() - t0:.0f} ثانية)")
    return True


def check_memory():
    """Memory folder status: rules file and golden clips (read only, never changes anything)."""
    memory = os.path.join(DRIVE_ROOT, "memory")
    if not os.path.isdir(memory):
        print("ℹ️ مجلد memory غير موجود بعد (Drive غير مربوط أو أول تشغيل)")
        return
    rules = os.path.join(memory, "my_editing_rules.md")
    print("✅ ملف قواعد الذوق موجود" if os.path.isfile(rules) else "ℹ️ ملف قواعد الذوق غير موجود بعد")
    golden = os.path.join(memory, "golden_clips")
    count = 0
    if os.path.isdir(golden):
        for _root, _dirs, files in os.walk(golden):
            count += sum(1 for f in files if os.path.splitext(f)[1].lower() in GOLDEN_EXTS)
    if count:
        print(f"🌟 اللقطات الذهبية: {count} فيديو في memory/golden_clips")
    else:
        print("ℹ️ ما فيه لقطات ذهبية بعد. حط أفضل 8 إلى 15 مقطع جاهز في memory/golden_clips")


# ============================================================================
# Entry point
# ============================================================================

def run_checks(vod_url=None, skip_cuda=False):
    """Run every check and return a small summary dict (also usable from the notebook)."""
    print(f"🐍 Python {sys.version.split()[0]}")
    summary = {}
    summary["cuda"] = None if skip_cuda else setup_cuda_libs()
    summary["ffmpeg"] = check_tool(["ffmpeg", "-version"], "FFmpeg")
    summary["ffprobe"] = check_tool(["ffprobe", "-version"], "ffprobe")
    summary["ytdlp"] = check_ytdlp(vod_url)
    summary["packages"] = package_versions()
    summary["imports"] = check_imports()
    summary["gpu"] = check_gpu()
    if summary["gpu"]:
        check_nvenc()
    print("🔬 فحص قدرات النسخة 3:")
    summary["timeline"] = check_gradio_timeline()
    summary["genai"] = check_genai_types()
    check_disk()
    check_memory()
    problems = [name for name in ("ffmpeg", "ffprobe") if not summary.get(name)]
    problems += [m for m, ok in (summary.get("imports") or {}).items() if not ok and m != "transformers"]
    if problems:
        print(f"⚠️ فحص البيئة خلص مع ملاحظات: {', '.join(problems)}")
    else:
        print("🎉 فحص البيئة خلص")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="فحص بيئة TwitchClipper")
    parser.add_argument("--vod", default=None, help="رابط بث لاختبار yt-dlp -J مباشرة (اختياري)")
    parser.add_argument("--skip-cuda", action="store_true", help="تخطي خطوة ldconfig لمكتبات CUDA")
    args = parser.parse_args(argv)
    run_checks(vod_url=args.vod, skip_cuda=args.skip_cuda)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# === END OF setup_tools.py ===
