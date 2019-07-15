# import the necessary packages
import os
from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import speech_recognition as sr
from pydub import AudioSegment as AS

# declaring key directories
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'


# create a spectrogram for each of the training wav files for a specified user
def trainingSpectrogram(username):
    source = DATABASE_DIR + username + '/audio/'
    i = 0
    wav_files = os.listdir(source)
    for wav in wav_files:
        i = i + 1
        # reading audio files of speaker
        sr, audio = read(source + wav)

        freq, times, spectrogram = signal.spectrogram(audio, sr)

        plt.pcolormesh(times, freq, spectrogram)
        plt.imshow(spectrogram, aspect='auto', origin='lower',
                   extent=[times.min(), times.max(), freq.min(), freq.max()])
        plt.title("Spectrogram of " + username + ' ' + wav)
        plt.xlabel('Time [sec]')
        plt.ylabel('Frequency [Hz]')
        plt.show()


def recognizeSpectrogram(username):
    source = DATABASE_DIR + username + '/audioComparison/'
    sr, audio = read(source + "loginAttempt.wav")
    freq, times, spectrogram = signal.spectrogram(audio, sr)

    plt.pcolormesh(times, freq, spectrogram)
    plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[times.min(), times.max(), freq.min(), freq.max()])
    plt.title("Spectrogram of " + username + " loginAttempt.wav")
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.show()


# Normalize the sound of all audio files for training data
def normalizeSoundTraining(username):
    i = 0
    avg_amplitude = -20.0  # measured in dBFS (decibels relative to full scale)
    wav_files = os.listdir(DATABASE_DIR + username + '/audio/')
    for wav in wav_files:
        i = i + 1
        audio = AS.from_file(DATABASE_DIR + username + '/audio/' + wav, "wav")
        change_in_dBFS = avg_amplitude - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        normalized_audio.export(DATABASE_DIR + username + '/audio/' + str(i) + '.wav', format='wav')


# Normalize the sound of the audio file created in attempts to login
def normalizeSoundRecognizing(username):
    avg_amplitude = -20.0  # measured in dBFS (decibels relative to full scale)
    audio = AS.from_file(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav', "wav")
    change_in_dBFS = avg_amplitude - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav', format='wav')


# eliminating the ambient noise in the training audio files
def eliminateAmbienceTraining(username):
    i = 0
    recognizer = sr.Recognizer()
    wav_files = os.listdir(DATABASE_DIR + username + '/audio/')
    for wav in wav_files:
        i = i + 1
        audio_file = sr.AudioFile(DATABASE_DIR + username + '/audio/' + wav)
        with audio_file as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            adjusted_audio = recognizer.record(source)

            # write adjusted audio to a WAV file
            with open(DATABASE_DIR + username + '/audio/' + wav, "wb") as file:
                file.write(adjusted_audio.get_wav_data())


# eliminating the ambient noise in the recognition audio file
def eliminateAmbienceRecognizing(username):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav')
    with audio_file as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        adjusted_audio = recognizer.record(source)

        # write adjusted audio to a WAV file
        with open(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav', "wb") as file:
            file.write(adjusted_audio.get_wav_data())


# if __name__ == '__main__':
#     recognizeSpectrogram("JohnDoe")
