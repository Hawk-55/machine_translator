import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.audio.AudioIOTensor("/home/harshal/Music/TURKISH_FEMALE_BURCU.mp3")

print(audio)

audio_slice = audio[100:]

# To remove last dimension
audio_tensor = tf.squeeze(audio_slice)
print(audio_tensor)

# To play the audio
from pydub import AudioSegment
from pydub.playback import play

audio_clip = AudioSegment.from_mp3("/home/harshal/Music/TURKISH_FEMALE_BURCU.mp3")
#play(audio_clip)


# To draw graph of .mp3 file
import matplotlib.pyplot as plt

plt.figure()
plt.plot(audio_tensor.numpy())
#plt.show()


# To print the audio clip
import speech_recognition as sr

#file_audio = "/home/harshal/Music/TURKISH_FEMALE_BURCU.mp3"
audio_clip.export("file_audio.wav",format="wav")

file_audio = "file_audio.wav"

r = sr.Recognizer()

with sr.AudioFile(file_audio) as source:
     audio_data = r.record(source)
     text = r.recognize_google(audio_data)
     print (text)






