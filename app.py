import streamlit as st
from pydub import AudioSegment
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import io

def mp3_to_mel_spectrogram(mp3_file):
    # Load the audio file
    audio = AudioSegment.from_file(mp3_file)
    audio_data = np.array(audio.get_array_of_samples())
    sample_rate = audio.frame_rate

    # Convert to floating-point audio data
    audio_data = audio_data.astype(np.float32) / 32767.0

    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Plot the mel spectrogram
    fig = plt.figure(figsize=(12, 8))
    librosa.display.specshow(mel_spec_db, sr=sample_rate, x_axis='off', y_axis='off')
    plt.axis('off')
    st.pyplot(fig)

    img = io.BytesIO()
    plt.savefig(img, format='png')
    
    btn = st.download_button(label='Save as Image', data=img, file_name='spectrogram.png', mime='image/png')

# Create a Streamlit web app
def main():
    st.title('MP3 to Mel Spectrogram Converter')

    # File uploader
    mp3_file = st.file_uploader('Upload an MP3 file', type=['mp3'])

    if mp3_file is not None:
        st.audio(mp3_file)

        # Convert MP3 to mel spectrogram
        if st.button('Convert'):
            mp3_to_mel_spectrogram(mp3_file)

# Run the app
if __name__ == '__main__':
    main()
