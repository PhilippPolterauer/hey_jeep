# %%
import soundcard as sc
import numpy as np
import torch
import time
import matplotlib.axes
import matplotlib.pyplot as plt
from openwakeword.model import Model

# get the current default speaker on your system:
default_speaker = sc.default_speaker()
# get the current default microphone on your system:
default_mic = sc.default_microphone()
vadmodel, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
last = time.time()


# Instantiate the model(s)
owwmodel = Model(
    wakeword_models=["hey_jeep.tflite"],
    vad_threshold=0.2,
)



fig, ax = plt.subplots()
ax: matplotlib.axes.Axes
# the container for the time values
times = np.arange(-5, 0, 0.08)
vads = np.zeros_like(times)
owws = np.zeros_like(times)
dts = np.zeros_like(times)

vad_p, = ax.plot(
    times,
    vads,
    label="VAD",
)
oww_p, = ax.plot(
    times,
    owws,
    label="hey_jeep",
)
dt_p, = ax.plot(
    times,
    dts,
    label="compute",
)

ax.set_ylim(0, 1)  # Adjust the y-axis limits if needed
ax.set_xlabel("Time")
leg = ax.legend()


SAMPLE_RATE = 16000
FRAME_TIME_MS = 80
FRAME_COUNT = SAMPLE_RATE // 1000 * FRAME_TIME_MS  # at 16khz


def push(x, val):
    x[:-1] = x[1:]
    x[-1] = val


with default_mic.recorder(
    samplerate=16000, channels=1, blocksize=FRAME_COUNT
) as recorder:
    while True:
        now = time.time()

        chunk = recorder.record(FRAME_COUNT)
        dt = (now - last) * 1e3  # should be 80ms

        vad = vadmodel(torch.from_numpy(chunk.ravel()), 16000).item()
        active = prediction = owwmodel.predict(
            (chunk.ravel() * 32767).astype(np.int16)
        )["hey_jeep"]

        push(times, now)
        push(vads, vad)
        push(owws, active)
        push(dts, dt / 80)

        vad_p.set_ydata(vads)
        oww_p.set_ydata(owws)
        dt_p.set_ydata(dts)



        plt.pause(0.001)  # Pause to allow the plot to update
        last = now

# live plot
