#!/usr/bin/env python3
import os, csv, time, wave, argparse, datetime as dt
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# -------- Helpers --------
def pick_input(hints=("Jabra","WebCam")):
    devs = sd.query_devices()
    for hint in hints:
        for i, d in enumerate(devs):
            if d.get("max_input_channels",0)>0 and hint.lower() in d["name"].lower():
                return i, d["name"]
    for i, d in enumerate(devs):
        if d.get("max_input_channels",0)>0:
            return i, d["name"]
    return None, "default"

def record_wav(path, seconds=10.0, fs=16000, device=None):
    if device is not None:
        sd.default.device = (device, None)
    sd.default.samplerate = fs
    sd.default.channels = 1
    print(f"🎤 Recording {seconds:.0f}s…")
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    pcm16 = (audio.flatten()*32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm16.tobytes())

def ensure_csv(path):
    if not os.path.exists(path) or os.stat(path).st_size==0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_iso","model","compute_type","input_device","sr_hz",
                "audio_sec","segments","tokens","tokens_per_s",
                "words","words_per_s","chars","chars_per_s",
                "compute_sec","rtf","text"
            ])

def bench_one(model, wav_path, audio_sec, model_name, compute_type, in_name, sr, csv_path):
    t0 = time.perf_counter()
    seg_iter, info = model.transcribe(wav_path, beam_size=1)  # light decoding
    segs = list(seg_iter)
    t1 = time.perf_counter()

    text = " ".join(s.text.strip() for s in segs).strip()
    tokens = sum(len(getattr(s, "tokens", []) or []) for s in segs)  # Whisper subword tokens
    words  = len(text.split()) if text else 0
    chars  = len(text)

    compute_sec = t1 - t0
    rtf = compute_sec / max(audio_sec, 1e-9)        # <1.0 means faster-than-real-time
    tok_per_s = tokens / max(compute_sec, 1e-9)
    wps = words / max(compute_sec, 1e-9)
    cps = chars / max(compute_sec, 1e-9)

    # append CSV
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            dt.datetime.now().isoformat(timespec="seconds"),
            model_name, compute_type, in_name, sr,
            f"{audio_sec:.2f}", len(segs), tokens, f"{tok_per_s:.2f}",
            words, f"{wps:.2f}", chars, f"{cps:.2f}",
            f"{compute_sec:.2f}", f"{rtf:.2f}", text
        ])

    # short report
    print("\n--- Run report ---")
    print(f"Model            : {model_name} ({compute_type})")
    try:
        print(f"Detected language: {getattr(info,'language','n/a')}")
    except Exception:
        print("Detected language: n/a")
    print(f"Input device     : {in_name} @ {sr} Hz")
    print(f"Audio duration   : {audio_sec:.2f} s")
    print(f"Compute time     : {compute_sec:.2f} s")
    print(f"Segments         : {len(segs)}")
    print(f"Tokens           : {tokens}  |  Tokens/sec: {tok_per_s:.2f}")
    print(f"Words/Chars      : {words}/{chars}  |  WPS: {wps:.2f}  CPS: {cps:.2f}")
    print(f"RTF              : {rtf:.2f}  (<1.0 is faster than real-time)")
    print(f"CSV appended     : {csv_path}\n")

    return {
        "tokens_per_s": tok_per_s,
        "rtf": rtf,
        "wps": wps,
        "cps": cps,
        "compute_sec": compute_sec
    }

# -------- Main --------
def main():
    ap = argparse.ArgumentParser(description="Whisper tokens/sec benchmark")
    ap.add_argument("--model", default="small.en", help="e.g., small.en | small | base.en | tiny.en")
    ap.add_argument("--compute-type", default="int8", help="int8 | int8_float32 | float32")
    ap.add_argument("--duration", type=float, default=10.0, help="recording length if no --file is given")
    ap.add_argument("--file", help="existing audio (wav/mp3/flac). Skips recording if provided")
    ap.add_argument("--sr", type=int, default=16000, help="recording sample rate")
    ap.add_argument("--runs", type=int, default=1, help="repeat runs (best of N for stability)")
    ap.add_argument("--csv", default="asr_bench.csv")
    args = ap.parse_args()

    ensure_csv(args.csv)

    # pick device only if recording
    if not args.file:
        in_dev, in_name = pick_input()
    else:
        in_dev, in_name = None, os.path.basename(args.file)

    print(f"🧠 Loading model '{args.model}' ({args.compute_type}) …")
    model = WhisperModel(args.model, compute_type=args.compute_type)

    results = []
    for i in range(args.runs):
        if not args.file:
            wav = "bench_clip.wav"
            input(f"[Run {i+1}/{args.runs}] Press ENTER to record {args.duration:.0f}s… ")
            record_wav(wav, seconds=args.duration, fs=args.sr, device=in_dev)
            audio_sec = args.duration
            sr = args.sr
        else:
            wav = args.file
            # get duration from file header if wav; else fall back to --duration
            audio_sec = args.duration
            sr = args.sr
            try:
                if wav.lower().endswith(".wav"):
                    with wave.open(wav, "rb") as wf:
                        frames = wf.getnframes(); sr = wf.getframerate()
                        audio_sec = frames / float(sr)
            except Exception:
                pass

        res = bench_one(model, wav, audio_sec, args.model, args.compute_type, in_name, sr, args.csv)
        results.append(res)

    if len(results) > 1:
        avg_tps = sum(r["tokens_per_s"] for r in results) / len(results)
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print("=== Session summary ===")
        print(f"Avg Tokens/sec: {avg_tps:.2f}")
        print(f"Avg RTF       : {avg_rtf:.2f}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")

