import os, serial, numpy as np
import time, requests, base64, subprocess, io
from pathlib import Path
from threading import Thread

from dotenv import load_dotenv
load_dotenv()

GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
UART_PORT     = "COM4"
BAUD          = 115200
RETRAIN_EVERY = 5    

GITHUB_REPO = "rakshithprabhu6-cell/stm32-f429-ota_new"

CUBIDE = (
    r"C:\Users\HP\AppData\Local\Temp"
    r"\STM32CubeIDE_aeda5489-2276-46b0-8bc6-63f09bd9c873"
    r"\C\ST\STM32CubeIDE_1.19.0\STM32CubeIDE"
    r"\stm32cubeide.exe"
)
WORKSPACE = r"C:\Users\HP\.stm32cubeaistudio\workspace"
PROJECT   = "stm32_f429"
BIN_PATH  = fr"{WORKSPACE}\{PROJECT}\Debug\{PROJECT}.bin"

MAGIC       = 0xAB
SAMPLE_SIZE = 2 + 28 * 28
HEADERS     = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}

# Folders
CORR_DIR    = Path(r"C:\STM32_OTA1\corrections")
INVALID_DIR = Path(r"C:\STM32_OTA1\invalid_samples")
CORR_DIR.mkdir(parents=True, exist_ok=True)
INVALID_DIR.mkdir(parents=True, exist_ok=True)


#  Serial helpers
def connect_serial(port, baud, retries=20, delay=3):
    for i in range(retries):
        try:
            s = serial.Serial(port, baud, timeout=5)
            print(f"[SERIAL] ✓ Connected on {port}")
            return s
        except Exception as e:
            print(f"[SERIAL] Waiting for port... ({i+1}/{retries})")
            time.sleep(delay)
    raise Exception("[SERIAL] Could not connect after retries")


#  GitHub helpers
def upload_sample(label, pixels):
    """Upload .npy file to GitHub corrections/ folder"""
    ts    = int(time.time() * 1000)

    
    buf = io.BytesIO()
    np.save(buf, pixels)
    content = base64.b64encode(buf.getvalue()).decode()

    
    if label == 10:
        fname = f"invalid_{ts}.npy"
    else:
        fname = f"label{label}_{ts}.npy"

    r = requests.put(
        f"https://api.github.com/repos/{GITHUB_REPO}/contents/corrections/{fname}",
        headers=HEADERS,
        json={
            "message": f"Add correction: {fname}",
            "content": content
        },
        timeout=30
    )
    ok = r.status_code in [200, 201]
    print(f"[UPLOAD] {'✓ OK' if ok else 'FAILED'} {fname}  (HTTP {r.status_code})")
    return ok


def wait_for_release(sha_before):
    print("[CLOUD] GitHub Actions started...")
    print("[CLOUD] Waiting for new firmware (~2 min)...")

    last_seen_tag = None
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            last_seen_tag = r.json().get("tag_name")
            print(f"[CLOUD] Current release: {last_seen_tag or 'none'}")
    except:
        pass

    for attempt in range(72):   
        time.sleep(10)
        elapsed = (attempt + 1) * 10
        try:
            r = requests.get(
                f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
                headers=HEADERS, timeout=15
            )
            if r.status_code == 200:
                release = r.json()
                tag     = release.get("tag_name")
                if tag and tag != last_seen_tag:
                    for asset in release.get("assets", []):
                        if asset["name"].endswith(".keras"):
                            print(f"\n[CLOUD] ✓ New release: {tag}")
                            return asset["browser_download_url"]
                    print(f"[CLOUD] Release {tag} found but no .keras asset yet...")
        except Exception as e:
            print(f"[CLOUD] Check error: {e}")

        mins = elapsed // 60
        secs = elapsed % 60
        print(f"[CLOUD] Still waiting... {mins}m {secs:02d}s")

    print("[CLOUD] Timeout — no new release after 12 min")
    return None


def download_firmware(url):
    save_path = r"C:\STM32_OTA1\cloud_firmware.keras"
    print("[DOWNLOAD] Downloading model from release...")

    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest",
        headers=HEADERS, timeout=15
    )
    if r.status_code != 200:
        print(f"[DOWNLOAD] Cannot get release: {r.status_code}")
        return None

    asset_id = None
    for asset in r.json().get("assets", []):
        if asset["name"].endswith(".keras"):
            asset_id = asset["id"]
            break

    if not asset_id:
        print("[DOWNLOAD] No .keras asset found")
        return None

    r = requests.get(
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/assets/{asset_id}",
        headers={**HEADERS, "Accept": "application/octet-stream"},
        allow_redirects=True,
        timeout=60
    )
    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        print(f"[DOWNLOAD] ✓ {len(r.content)//1024} KB saved → {save_path}")
        return save_path

    print(f"[DOWNLOAD] FAILED: {r.status_code}")
    return None


def flash_board(bin_path):
    print("[FLASH] Running full pipeline (convert+build+flash)...")
    import sys
    sys.path.insert(0, r"C:\STM32_OTA1")
    from auto_pipeline1 import step2_stedgeai, step3_copy, step4_build, step5_flash

    for name, fn in [
        ("Convert", step2_stedgeai),
        ("Copy",    step3_copy),
        ("Build",   step4_build),
        ("Flash",   step5_flash),
    ]:
        print(f"[FLASH] → {name}...")
        if not fn():
            print(f"[FLASH] ✗ {name} failed")
            return False
        print(f"[FLASH] ✓ {name} done")
    return True


def get_current_sha():
    try:
        r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/commits/main",
            headers=HEADERS, timeout=10
        )
        if r.status_code == 200:
            return r.json()["sha"]
    except:
        pass
    return None


def handle_sample(label, pixels):
    """Upload sample to GitHub → wait for cloud retrain → flash"""
    label_str = "Invalid" if label == 10 else str(label)
    print(f"\n[CLOUD] Handling label={label_str}")
    print("[CLOUD] Uploading to GitHub...")

    sha = get_current_sha()
    if not upload_sample(label, pixels):
        print("[CLOUD] Upload failed — skipping pipeline")
        return

    print("[CLOUD] ✓ Uploaded! Waiting for GitHub Actions...")
    firmware_url = wait_for_release(sha)

    if firmware_url:
        bin_path = download_firmware(firmware_url)
        if bin_path:
            flash_board(bin_path)
            print("\n✅ Board updated with cloud-trained model!")
        else:
            print("[CLOUD] Download failed")
    else:
        print("[ERROR] No firmware received from cloud")


# Main listener
def listen():
    ser = connect_serial(UART_PORT, BAUD)

    print("=" * 45)
    print("  STM32 CLOUD OTA READY")
    print(f"  Repo         : {GITHUB_REPO}")
    print(f"  Retrain every: {RETRAIN_EVERY} corrections")
    print("=" * 45)
    print("  Draw → PREDICT → WRONG → select label")
    print("  Cloud retrains in ~2 min automatically")
    print("=" * 45 + "\n")

    buf              = bytearray()
    correction_count = 0

    while True:
        try:
            buf += ser.read(ser.in_waiting or 1)

            while len(buf) >= SAMPLE_SIZE:
                idx = buf.find(MAGIC)
                if idx == -1:
                    buf.clear()
                    break
                if idx > 0:
                    print(f"[UART] Skipping {idx} garbage bytes")
                    buf = buf[idx:]
                if len(buf) < SAMPLE_SIZE:
                    break

                label  = buf[1]
                pixels = np.frombuffer(
                    buf[2:2+28*28], dtype=np.uint8
                ).reshape(28, 28).copy()
                buf = buf[SAMPLE_SIZE:]

                if label > 10:
                    print(f"[RECV] Bad label {label} — skip")
                    continue

                ts = int(time.time() * 1000)

                # Save locally
                if label == 10:
                    
                    np.save(str(INVALID_DIR / f"invalid_{ts}.npy"), pixels)
                    np.save(str(CORR_DIR   / f"label10_{ts}.npy"), pixels)
                    total = len(list(INVALID_DIR.glob("*.npy")))
                    print(f"[RECV] ✓ Invalid/letter saved  total={total}")
                else:
                    np.save(str(CORR_DIR / f"label{label}_{ts}.npy"), pixels)
                    print(f"[RECV] ✓ label={label} saved")

                
                correction_count += 1
                print(f"[RECV] Corrections: {correction_count}/{RETRAIN_EVERY}")

                if correction_count % RETRAIN_EVERY == 0:
                    print(f"[PIPELINE] {RETRAIN_EVERY} corrections — triggering cloud retrain!")
                    correction_count = 0
                    Thread(
                        target=handle_sample,
                        args=(label, pixels),
                        daemon=True
                    ).start()
                else:
                    remaining = RETRAIN_EVERY - (correction_count % RETRAIN_EVERY)
                    print(f"[RECV] Need {remaining} more before retraining")

        except serial.SerialException as e:
            print(f"[SERIAL] Lost connection: {e}")
            try:
                ser.close()
            except:
                pass
            time.sleep(5)
            ser = connect_serial(UART_PORT, BAUD)
            buf = bytearray()

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(1)


if __name__ == "__main__":
    listen()