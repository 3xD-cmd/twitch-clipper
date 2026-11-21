import glob
import os
import site
import subprocess
import sys


def check_tool(cmd, name):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        first_line = out.strip().splitlines()[0]
        print(f"✅ {name}: {first_line}")
    except Exception as e:
        print(f"❌ مشكلة في {name}: {e}")


def setup_cuda_libs():
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
            subprocess.run(["ldconfig"], check=True)
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


if __name__ == "__main__":
    setup_cuda_libs()
    check_tool(["yt-dlp", "--version"], "yt-dlp")
    check_tool(["ffmpeg", "-version"], "FFmpeg")
    check_tool(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        "GPU",
    )
