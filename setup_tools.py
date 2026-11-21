import subprocess


def check_tool(cmd, name):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        first_line = out.strip().splitlines()[0]
        print(f"✅ {name}: {first_line}")
    except Exception as e:
        print(f"❌ مشكلة في {name}: {e}")


if __name__ == "__main__":
    check_tool(["yt-dlp", "--version"], "yt-dlp")
    check_tool(["ffmpeg", "-version"], "FFmpeg")
    check_tool(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        "GPU",
    )
