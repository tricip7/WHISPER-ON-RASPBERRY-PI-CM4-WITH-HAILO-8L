import wave, numpy as np, sounddevice as sd
from faster_whisper import WhisperModel

MODEL_NAME = "small.en"  # use "small" for multilingual; "small.en" is leaner/faster for English
SAMPLE_RATE = 16000
DUR = 3.0  # seconds

def pick_input():
    # Prefer Jabra if present; else fall back to WebCam; else default
    priority = ["Jabra", "WebCam"]
    devs = sd.query_devices()
    for name in priority:
        for i,d in enumerate(devs):
            if d.get("max_input_channels",0)>0 and name.lower() in d["name"].lower():
                return i
    # fallback to first input device
    for i,d in enumerate(devs):
        if d.get("max_input_channels",0)>0:
            return i
    return None

IN_DEV = pick_input()
if IN_DEV is not None:
    sd.default.device = (IN_DEV, None)
sd.default.samplerate = SAMPLE_RATE
sd.default.channels = 1

model = WhisperModel(MODEL_NAME, compute_type="int8")

def record(path, seconds=DUR, fs=SAMPLE_RATE):
    print("🎤 Speak...")
    audio = sd.rec(int(seconds*fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    pcm16 = (audio.flatten() * 32767).astype(np.int16)
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(fs)
        wf.writeframes(pcm16.tobytes())

while True:
    input("Press ENTER to record (Ctrl+C to quit)…")
    record("clip.wav")
    segs, info = model.transcribe("clip.wav", beam_size=1)
    text = " ".join(s.text for s in segs).strip()
    print(f"🗣 {text}")
