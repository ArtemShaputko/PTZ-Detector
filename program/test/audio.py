import sounddevice as sd
import numpy as np
from scipy.io import wavfile

DEVICE = 54       # поменяй если нужно
FS = 48000       # частота
DURATION = 6     # секунды записи
OUTPUT = "test.wav"

print(sd.query_devices(DEVICE))
print(f"\nЗапись {DURATION} сек с устройства {DEVICE}...")

audio = sd.rec(int(DURATION * FS), samplerate=FS, channels=1,
               dtype='int16', device=DEVICE)
sd.wait()

wavfile.write(OUTPUT, FS, audio)
print(f"Сохранено в {OUTPUT}")