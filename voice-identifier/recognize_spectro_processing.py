# import necessary packages
import os
import sys
import json
import filetype
import subprocess

from data_processing import normalizeSoundRecognizing, eliminateAmbienceRecognizing, recognizeSpectrogram

# specify important paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
DATABASE_DIR = ROOT_DIR + '/users/'

# fetch data passed through python shell
lines = sys.stdin.readline()
data = json.load(lines)
name = str(data['name'])


def process_spectro(username):
    # specify path to loginAttempt.wav file
    path = DATABASE_DIR + username + '/audioComparison/loginAttempt.wav'

    # ensure that loginAttempt audio file is in '.wav' format (will work for multiple audio files as well)
    kind = filetype.guess(path)
    if kind.extension != "wav":
        command = "ffmpeg -i " + path + " -ab 160k -ac 2 -ar 44100 -vn " + 'loginAttempt.wav'
        subprocess.call(command, shell=True)
        os.remove(path)
        os.rename('./loginAttempt.wav', path)

    # perform data processing functions to normalize the sound, eliminate background noise and create a spectrogram from
    # the login attempt found in audioComparison directory
    normalizeSoundRecognizing(username)
    eliminateAmbienceRecognizing(username)
    recognizeSpectrogram(username)


if __name__ == "__main__":
    process_spectro(name)
