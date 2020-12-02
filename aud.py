import tensorflow as tf
import tensorflow_io as tfio

audio = tfio.audio.AudioIOTensor("/home/harshal/Music/TURKISH_FEMALE_BURCU.mp3")

print(audio)

audio_slice = audio[100:]

# To remove last dimension
audio_tensor = tf.squeeze(audio_slice, axis=[-1])
print(audio_tensor)
