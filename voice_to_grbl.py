#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, re, csv, time, wave, datetime as dt
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import serial

# --------------------------- CONFIG ---------------------------
# Audio / ASR
MODEL_NAME = "small.en"            # or "small" (multilingual)
SAMPLE_RATE = 16000
CAPTURE_SECONDS = 4.0              # speak a short command
INPUT_PREFERENCES = ["Jabra", "WebCam"]

# GRBL serial
GRBL_PORT = "/dev/ttyACM0"
GRBL_BAUD = 115200

# Motor / driver (edit for your hardware)
MOTOR_STEPS_PER_REV = 200          # 1.8° motor
MICROSTEP = 16                     # DIP switches on your driver
FEED_RPM = 60                      # "rotations per minute" since 1mm = 1 rotation
CSV_REPORT = "grbl_voice_report.csv"
# --------------------------------------------------------------


# ---- Utilities ------------------------------------------------
def pick_input(preferences=INPUT_PREFERENCES):
    devs = sd.query_devices()
    for pref in preferences:
        for i, d in enumerate(devs):
            if d.get("max_input_channels", 0) > 0 and pref.lower() in d["name"].lower():
                return i, d["name"]
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) > 0:
            return i, d["name"]
    return None, "default"

def record_wav(path, seconds=CAPTURE_SECONDS, fs=SAMPLE_RATE):
    print(f"🎤 Speak now ({seconds:.0f}s)…")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    pcm16 = (audio.flatten() * 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm16.tobytes())

def ensure_csv(path):
    if not os.path.exists(path) or os.stat(path).st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp_iso","transcript","direction","turns",
                "steps_per_turn","total_steps","gcode","grbl_reply"
            ])

# ---- Command parsing ------------------------------------------
_NUM_WORDS = {
    "zero":0, "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
    "eleven":11, "twelve":12,
    "half":0.5, "quarter":0.25
}
def words_to_number(text: str):
    # handle patterns like "one and a half", "two point five"
    text = text.lower().strip()
    text = text.replace("-", " ")
    m = re.search(r"(\d+(\.\d+)?)", text)
    if m:
        return float(m.group(1))
    # "X and a half"
    m = re.search(r"(\w+)\s+(and\s+a\s+)?half", text)
    if m and m.group(1) in _NUM_WORDS:
        return _NUM_WORDS[m.group(1)] + 0.5
    # "X and a quarter"
    m = re.search(r"(\w+)\s+(and\s+a\s+)?quarter", text)
    if m and m.group(1) in _NUM_WORDS:
        return _NUM_WORDS[m.group(1)] + 0.25
    # single word number
    for w,v in _NUM_WORDS.items():
        if re.search(rf"\b{w}\b", text):
            return float(v)
    return None

def parse_command(text: str):
    """
    Returns (direction, turns) or ("stop", 0) or (None, None)
    direction in {"forward","backward"}
    """
    t = text.lower()
    if any(w in t for w in ["stop", "hold", "pause"]):
        return ("stop", 0.0)

    dir_ = None
    if any(w in t for w in ["forward","clockwise","cw"]):
        dir_ = "forward"
    if any(w in t for w in ["backward","reverse","counter","anticlockwise","ccw"]):
        dir_ = "backward" if dir_ is None else dir_  # keep first match

    # find units: spins/turns/rotations
    if re.search(r"\b(spin|spins|turn|turns|rotation|rotations|rev|revs|revolution)\b", t):
        turns = words_to_number(t)
    else:
        # tolerate: "forward two", "reverse 1.5"
        turns = words_to_number(t)

    return (dir_, turns)

# ---- GRBL helpers ---------------------------------------------
def grbl_open(port=GRBL_PORT, baud=GRBL_BAUD):
    ser = serial.Serial(port, baud)
    time.sleep(2)
    ser.write(b"\r\n\r\n")
    time.sleep(2)
    ser.flushInput()
    return ser

def grbl_send(ser, line: str, echo=True, wait=0.1):
    if echo: print(">>", line.strip())
    ser.write((line+"\n").encode("ascii"))
    time.sleep(wait)
    out = ser.read_all().decode(errors="ignore")
    if out and echo:
        print(out.strip())
    return out

def configure_for_rotations(ser):
    steps_per_rotation = MOTOR_STEPS_PER_REV * MICROSTEP  # e.g., 200 * 16 = 3200
    # Map: 1 mm == 1 rotation → $100 = steps_per_rotation
    grbl_send(ser, f"$100={steps_per_rotation:.3f}")
    # feed (mm/min) == rotations/min because of mapping
    grbl_send(ser, f"$110={FEED_RPM:.1f}")
    grbl_send(ser, "G91")  # relative moves

def move_rotations(ser, direction: str, turns: float):
    sign = +1.0 if direction == "forward" else -1.0
    distance_mm = sign * turns                # because 1 mm = 1 rotation
    g = f"G1 X{distance_mm:.4f} F{FEED_RPM:.2f}"
    reply = grbl_send(ser, g, echo=True, wait=0.2)
    return g, reply

# ---- Main ------------------------------------------------------
def main():
    # Audio device
    in_dev, dev_name = pick_input()
    if in_dev is not None:
        sd.default.device = (in_dev, None)
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1
    print(f"🎧 Mic: {dev_name}")

    # Load Whisper
    print(f"🧠 Loading model '{MODEL_NAME}' (int8)…")
    asr = WhisperModel(MODEL_NAME, compute_type="int8")

    # Open GRBL
    print("🔌 Opening GRBL…")
    ser = grbl_open()
    configure_for_rotations(ser)

    ensure_csv(CSV_REPORT)
    steps_per_turn = MOTOR_STEPS_PER_REV * MICROSTEP

    print("\nSay commands like:\n"
          "  • 'forward two spins'\n"
          "  • 'backward 1.5 turns'\n"
          "  • 'stop'\n"
          "Press ENTER to listen; Ctrl+C to quit.\n")

    while True:
        input("▶️  Press ENTER and speak… ")
        wav_path = "utterance.wav"
        record_wav(wav_path)

        # Transcribe
        segs, info = asr.transcribe(wav_path, beam_size=1)
        text = " ".join(s.text.strip() for s in segs).strip()
        print(f"📝 Transcript: {text!r}")

        direction, turns = parse_command(text)
        if direction == "stop":
            print("⏸ Feed hold")
            ser.write(b"\x85")         # realtime feed hold
            time.sleep(0.5)
            grbl_send(ser, "?")
            continue

        if (direction is None) or (turns is None):
            print("🤷 Couldn’t understand direction/turns. Try again.")
            continue

        # Send move
        gcode, reply = move_rotations(ser, direction, turns)
        total_steps = int(round(steps_per_turn * turns))

        # Log
        with open(CSV_REPORT, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                dt.datetime.now().isoformat(timespec="seconds"),
                text, direction, f"{turns:.4f}",
                steps_per_turn, total_steps, gcode, reply.strip().replace("\n"," | ")
            ])

        # Console report
        print("\n--- Voice → Motion Report ---")
        print(f"Command:   {text!r}")
        print(f"Parsed:    {direction}  {turns:.4f} rotation(s)")
        print(f"G-code:    {gcode}")
        print(f"Steps/rot: {steps_per_turn}   Total steps: {total_steps}")
        print(f"GRBL:      {reply.strip() if reply else '(no reply)'}")
        print(f"CSV →      {CSV_REPORT}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye!")
