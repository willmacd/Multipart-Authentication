# import the necessary packages
import os
import math
import wave
from scipy import signal
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import speech_recognition as sr
from pydub import AudioSegment as AS

# declaring key directories
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'


# trim wav files to a specified length
def trimWavFile(originPath, outputPath):
    # desired length of audio file (in milliseconds
    timeSplit = 3000

    # check duration of specified wave file
    wav = wave.open(originPath, 'r')
    frameRate = wav.getframerate()
    numFrames = wav.getnframes()
    duration = numFrames/float(frameRate)

    # get the number of times audio file can be split based on time per split and duration
    multiple = math.floor(divmod(duration, timeSplit/1000)[0])
    for i in range(multiple):
        splitNum = len(os.listdir(ROOT_DIR + '/randomSpectrograms/'))
        splitWav = AS.from_wav(originPath)
        splitWav = splitWav[timeSplit*i:timeSplit*(i+1)]
        splitWav.export(str(outputPath) + 'split' + str(splitNum) + '.wav', format='wav')


# create a spectrogram for each of the training wav files for a specified user
def trainingSpectrogram(username):
    train_source = DATABASE_DIR + username + '/audioTraining/user/'
    validation_source = DATABASE_DIR + username + '/audioValidation/user/'

    source_list = [train_source, validation_source]

    for source in source_list:
        for i in range(len(os.listdir(source))):
            # if file already exists, remove it from directory
            if str(i) + ".png" in os.listdir(source):
                os.unlink(source + username + str(i) + ".png")

        if source is train_source:
            i = 0
        else:
            i = 3

        wav_files = os.listdir(source)
        for wav in wav_files:
            # reading audio files of speaker
            sr, audio = read(source + wav)
            freq, times, spectrogram = signal.spectrogram(audio, sr)
            plt.pcolormesh(times, freq, spectrogram)
            fig = plt.imshow(spectrogram, aspect='auto', origin='lower',
                             extent=[times.min(), times.max(), freq.min(), freq.max()])
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(username + str(i) + '.png', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.0)
            os.rename(username + str(i) + '.png', source + username + str(i) + '.png')
            os.unlink(source + wav)
            i = i + 1
            # fig.axes.get_xaxis().set_visible(True)
            # fig.axes.get_yaxis().set_visible(True)
            # plt.title("Spectrogram of " + username + wav)
            # plt.xlabel('Time [sec]')
            # plt.ylabel('Frequency [Hz]')
            # plt.show()


# create a spectrogram for each the login attempt wav file
def recognizeSpectrogram(username):
    source = DATABASE_DIR + username + '/audioComparison/'

    # if file already exists, remove it from directory
    if "loginAttempt.png" in os.listdir(source):
        os.unlink(source + "loginAttempt.png")

    sr, audio = read(source + "loginAttempt.wav")
    freq, times, spectrogram = signal.spectrogram(audio, sr)
    plt.pcolormesh(times, freq, spectrogram)
    fig = plt.imshow(spectrogram, aspect='auto', origin='lower', extent=[times.min(), times.max(), freq.min(), freq.max()])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('loginAttempt.png', bbox_inches='tight', dpi=300, transparent=True, pad_inches=0.0)
    os.rename('loginAttempt.png', source + 'loginAttempt.png')
    os.unlink(source + 'loginAttempt.wav')

    # fig.axes.get_xaxis().set_visible(True)
    # fig.axes.get_yaxis().set_visible(True)
    # plt.title("Spectrogram of " + username + " loginAttempt.wav")
    # plt.xlabel('Time [sec]')
    # plt.ylabel('Frequency [Hz]')
    # plt.show()


# Normalize the sound of all audio files for training data
def normalizeSoundTraining(username):
    train_source = DATABASE_DIR + username + '/audioTraining/user/'
    validation_source = DATABASE_DIR + username + '/audioValidation/user/'
    source_list = [train_source, validation_source]

    for source in source_list:
        avg_amplitude = -20.0  # measured in dBFS (decibels relative to full scale)
        wav_files = os.listdir(source)
        for wav in wav_files:
            audio = AS.from_file(source + wav, "wav")
            change_in_dBFS = avg_amplitude - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dBFS)
            normalized_audio.export(source + wav, format='wav')


# Normalize the sound of the audio file created in attempts to login
def normalizeSoundRecognizing(username):
    avg_amplitude = -20.0  # measured in dBFS (decibels relative to full scale)
    audio = AS.from_file(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav', "wav")
    change_in_dBFS = avg_amplitude - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)
    normalized_audio.export(DATABASE_DIR + username + '/audioComparison/loginAttempt.wav', format='wav')


# eliminating the ambient noise in the training audio files
def eliminateAmbienceTraining(username):
    train_source = DATABASE_DIR + username + '/audioTraining/user/'
    validation_source = DATABASE_DIR + username + '/audioValidation/user/'
    source_list = [train_source, validation_source]

    i = 0
    recognizer = sr.Recognizer()
    for source in source_list:
        wav_files = os.listdir(source)
        for wav in wav_files:
            i = i + 1
            audio_file = sr.AudioFile(source + wav)
            with audio_file as sound:
                recognizer.adjust_for_ambient_noise(sound, duration=0.5)
                adjusted_audio = recognizer.record(sound)

                # write adjusted audio to a WAV file
                with open(source + wav, "wb") as file:
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
    # eliminateAmbienceTraining("Will")
    # trainingSpectrogram('Will')