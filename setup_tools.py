import json
import os
import stat
import subprocess
import urllib.request
import zipfile

BIN_DIR = "/content/bin"
os.makedirs(BIN_DIR, exist_ok=True)


def download_twitch_cli():
    exe_path = os.path.join(BIN_DIR, "TwitchDownloaderCLI")
    if os.path.exists(exe_path):
        print("✅ TwitchDownloaderCLI جاهز مسبقاً")
        return exe_path

    print("⬇️ جاري تحميل أحدث نسخة من TwitchDownloaderCLI...")
    api_url = "https://api.github.com/repos/lay295/TwitchDownloader/releases/latest"
    req = urllib.request.Request(api_url, headers={"User-Agent": "TwitchClipper"})
    with urllib.request.urlopen(req) as resp:
        release = json.loads(resp.read().decode())

    zip_url = None
    for asset in release.get("assets", []):
        name = asset["name"].lower()
        if "cli" in name and "linux-x64" in name and name.endswith(".zip"):
            zip_url = asset["browser_download_url"]
            break

    if not zip_url:
        raise RuntimeError("❌ ما لقينا نسخة Linux-x64 في GitHub Releases!")

    zip_path = os.path.join(BIN_DIR, "tdcli.zip")
    urllib.request.urlretrieve(zip_url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(BIN_DIR)
    os.remove(zip_path)

    os.chmod(exe_path, os.stat(exe_path).st_mode | stat.S_IEXEC)
    print("✅ تم تثبيت TwitchDownloaderCLI بنجاح")
    return exe_path


def check_tool(cmd, name):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        first_line = out.strip().splitlines()[0]
        print(f"✅ {name}: {first_line}")
    except Exception as e:
        print(f"❌ مشكلة في {name}: {e}")


if __name__ == "__main__":
    cli = download_twitch_cli()
    check_tool([cli, "--version"], "TwitchDownloaderCLI")
    check_tool(["yt-dlp", "--version"], "yt-dlp")
    check_tool(["ffmpeg", "-version"], "FFmpeg")
    check_tool(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], "GPU")
