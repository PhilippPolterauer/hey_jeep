# %%
import soundcard as sc
import numpy as np

# get a list of all speakers:
speakers = sc.all_speakers()
# get the current default speaker on your system:
default_speaker = sc.default_speaker()
# get a list of all microphones:
mics = sc.all_microphones()
# get the current default microphone on your system:
default_mic = sc.default_microphone()

print(default_speaker)
print(default_mic)

# %%
# record and play back one second of audio:
fs = 16000
rec_sec = 5
# %% record data
data = default_mic.record(samplerate=fs, numframes=fs * rec_sec)
data

# %%
import matplotlib.pyplot as plt

plt.plot((data * 32767).astype(np.int16))

# %% try to predict it useing openwakeword
#
# default_speaker.play(data, fs)
# data = default_mic.record(samplerate=fs, numframes=fs * rec_sec)

# the model should work for 16bit PCM mono audio at 16kHz
frame = (data * 32767).astype(np.int16)
frame = frame.ravel()
from openwakeword.model import Model

# Instantiate the model(s)
model = Model(
    wakeword_models=["hey jarvis"],
    vad_threshold=0.7,
)
chunk = (16 * 80 * 1 ) # 1 chunks of 80ms each at 16khz
predictions = []
for start in range(0, len(frame), chunk):
    end = start + chunk
    # Get predictions for the frame
    prediction = model.predict(frame[start:end])
    print(prediction)
    predictions.append(prediction["hey jarvis"])

predictions
import matplotlib.pyplot as plt

plt.plot(predictions)
# %%


# %%


# %%
