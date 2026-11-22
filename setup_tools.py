"""
setup_tools.py
Colab environment check for TwitchClipper. Run after pip install:
    python setup_tools.py

What it does
  * CUDA 12 runtime fix (kept from the battle tested version): registers the pip installed
    nvidia/*/lib folders with ldconfig so faster-whisper never fails with
    "libcublas.so.12 not found", then verifies that libcublas.so.12 loads.
  * Prints the versions of every tool and package the pipeline uses.
  * GPU check that is informative, not alarming: on the CPU runtime (Stage 3 and Stage 4)
    no GPU is expected; on the T4 runtime it confirms the card and tests NVENC.
  * Free disk space on /tmp and on Google Drive.
"""

import glob
import os
import shutil
import site
import subprocess
import sys

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

DRIVE_ROOT = "/content/drive/MyDrive/TwitchClipper"


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

    try:
        subprocess.run(
            [sys.executable, "-c", "import ctypes; ctypes.CDLL('libcublas.so.12')"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("✅ CUDA cuBLAS 12: جاهز وشغال")
    except Exception as e:
        print(f"❌ مشكلة في CUDA cuBLAS 12: {e}")


def package_versions():
    print("📦 إصدارات المكتبات:")
    for dist, label in PACKAGES:
        version = None
        if _metadata is not None:
            try:
                version = _metadata.version(dist)
            except Exception:
                version = None
        if version:
            print(f"   ✅ {label}: {version}")
        else:
            optional = "AST" in label
            mark = "ℹ️" if optional else "❌"
            hint = "اختياري، المرحلة 2 تكمل بدونه" if optional else "غير مثبت، شغّل pip install مرة ثانية"
            print(f"   {mark} {label}: {hint}")


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


if __name__ == "__main__":
    print(f"🐍 Python {sys.version.split()[0]}")
    setup_cuda_libs()
    check_tool(["yt-dlp", "--version"], "yt-dlp")
    check_tool(["ffmpeg", "-version"], "FFmpeg")
    package_versions()
    if check_gpu():
        check_nvenc()
    check_disk()
    print("🎉 فحص البيئة خلص")


# === END OF setup_tools.py ===
