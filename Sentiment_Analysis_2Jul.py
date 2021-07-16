
# coding: utf-8

# ### Packages required
# - !pip install librosa soundfile numpy sklearn pyaudio(for microphone) pydub(convert mp3 to WAV) 
# - SpeechRecognition portaudio ffmpeg 

# ### A critical task is source separation/ segmentation - customer, agent, background + pauses
# 
# - Path 1. Sentiment Analysis of audio after converting to text
# 
# - Path 2(a). Voice Activity Detection of Calls - separation of voice into Agent and Customer chunks (Google’s webrtc vad??). These chunks then need to be clustered. 
#  ==> However, this might not be needed if we can get dual-channel recordings. To be noted that a customer might
#      have more than 1 emotion in a call
# 
# - Path 2(b). Sentiment Analysis of audio using librosa -- Classification model trained on 
#     RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset (supervised learning)
#     This dataset contains only American english sounds/ emotions. Therefore, another datasets are required. However, the 
#     methodology would remain same
# 
# - Exporting/ pickling of models
# 
# - Path 2(c). Passing of 'customer' voice chunks into the model to understand sentiment
# 
# - Use weighted-average sentiment score of the 2 Paths

# #### Edits required
# 
# - /anaconda/lib/python3.6/site-packages/librosa/util/decorators.py --> 
# - changed import numba.decorators to numba.core.decorators
# 
# - SciPy >=0.19 uses "from scipy.special import comb" instead of "from scipy.misc import comb"

# In[ ]:

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import librosa
import soundfile
import os, glob
import numpy as np
import datetime
#from datetime import timedelta



# #### By default, when loading stereo audio files, the librosa.load() function downmixes to mono by averaging left- and right-channels, and then resamples the monophonic signal to the default rate sr=22050 Hz
# 
# 

# #### Usually call-center recordings are in dual-channel stereo format, with one channel for each speaker
#  - PyDub can be used to split such stereo recordings into mono files
# 
#  - WIP ==> to identify customer voice vs. agent voice (hopefully the calmer one)
# 
#  - Dependency on ffmpeg which contains ffprobe. ffmpeg needs to be installed using homebrew, while pip only installs the wrapper

# In[ ]:

from pydub import AudioSegment

folder = "/Users/guest1/Downloads/"

#stereo_phone_call = AudioSegment.from_file("/Users/guest1/Downloads/Karissa_Hobbs_-_09_-_Lets_Go_Fishin.mp3")

# Already a single-channel/mono sound file
stereo_phone_call = AudioSegment.from_file("/Users/guest1/Downloads/speech-emotion-recognition-ravdess-data/Actor_01/03-01-02-01-02-01-01.wav")

print(f"Stereo number channels: {stereo_phone_call.channels}")

# Split stereo phone call and check channels
channels = stereo_phone_call.split_to_mono()

# Save new channels separately
phone_call_channel_1 = channels[0]
phone_call_channel_2 = channels[1]

#phone_call_channel_1.export(folder + "ch1.mp3", format="mp3")
#phone_call_channel_2.export(folder + "ch2.mp3", format="mp3")

print(phone_call_channel_1.DEFAULT_CODECS)
print(phone_call_channel_2.DEFAULT_CODECS)



# In[ ]:

# breaking down sounds into smaller plots (images) comprising time domain and frequency domain
# Image processing is very advanced, and hence the current approach is to convert small blocks of sounds into
# images, and extract features

# WAV is uncompressed, mp3 is lossy compression

import matplotlib.pyplot as plt
from librosa import display

# Already a single-channel/mono sound file
file = "/Users/guest1/Downloads/speech-emotion-recognition-ravdess-data/Actor_01/03-01-02-01-02-01-01.wav"

y, sr = librosa.load(file) # time series, sample rate
print(y)
print(sr)
#y, sr = librosa.load(librosa.util.example_audio_file())

librosa.feature.melspectrogram(y=y, sr=sr)

D = np.abs(librosa.stft(y))**2  # D= Complex-valued matrix of short-term Fourier transform coefficients
print(D)

#S = librosa.feature.melspectrogram(S=D, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000) #spectogram

plt.figure(figsize=(10, 4))
S_dB = librosa.power_to_db(S, ref=np.max) #convert to decibels
librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000)

plt.colorbar(format='%+2.0f dB')
plt.title('Mel-frequency spectrogram')
plt.tight_layout()
plt.show()


y_harm, y_perc = librosa.effects.hpss(y)
plt.subplot(3, 1, 3)
librosa.display.waveplot(y_harm, sr=sr, alpha=0.25)
librosa.display.waveplot(y_perc, sr=sr, color='r', alpha=0.5)
plt.title('Harmonic + Percussive')
#plt.tight_layout()
plt.show()

plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.title('Monophonic')


# In[ ]:

#######
# Attempting to understand the Fourier Tempogram
#######

hop_length = 512
oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
tempogram = librosa.feature.fourier_tempogram(onset_envelope=oenv, sr=sr,
                                               hop_length=hop_length)
# Compute the auto-correlation tempogram, unnormalized to make comparison easier
ac_tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                          hop_length=hop_length, norm=None)
plt.figure(figsize=(8, 8))
plt.subplot(3, 1, 1)
plt.plot(oenv, label='Onset strength')
plt.xticks([])
plt.legend(frameon=True)
plt.axis('tight')
plt.subplot(3, 1, 2)
librosa.display.specshow(np.abs(tempogram), sr=sr, hop_length=hop_length,
                          x_axis='time', y_axis='fourier_tempo', cmap='magma')
plt.title('Fourier tempogram')
plt.subplot(3, 1, 3)
librosa.display.specshow(ac_tempogram, sr=sr, hop_length=hop_length,
                          x_axis='time', y_axis='tempo', cmap='magma')
plt.title('Autocorrelation tempogram')
plt.tight_layout()
plt.show()



# In[ ]:

#Extract features (mfcc, chroma, mel) from a sound file

# sound_file doesn't resample the audio to 22050 Hz unlike librosa

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        #print(X)
        sample_rate=sound_file.samplerate
        #print(sample_rate)
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:   #timbre?
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma: #pitch?
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
            #print(result)
    return result


# In[ ]:

#Emotions to observe
#observed_emotions=['angry', 'calm', 'disgust', 'fearful', 'happy']
selected_emotions=['angry', 'calm', 'happy', 'neutral' ]

#Emotions Dictionary
emotions={
  '01':'neutral', '02':'calm', '03':'happy', '04':'sad'
 ,'05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'
}


# In[ ]:

#function to load the data and extract features for each sound file

## Dataset downloaded from https://drive.google.com/file/d/1wWsrN2Ep7x6lWqOXfr4rpKGYrJhWc8z7/view
## this dataset contains 1441 .WAV files
## find . -type f | wc -l 

def load_data(fileName, test_size):
    x,y=[],[]
    workDir = '/Users/guest1/Downloads/'
    #fileName = "speech-emotion-recognition-ravdess-data/Actor_*/*.wav"
    
    print(" Inside loading data")

    for file in glob.glob(workDir + fileName):
        
        file_name=os.path.basename(file)
        #print(file_name)
        
        ## this block needs to change depending on dataset to extract y(target)
        # 03-01-05-01-01-01-21.wav -- picks up 05
        emotion=emotions[file_name.split("-")[2]]

        if emotion not in selected_emotions:
            continue
        ## this block needs to change depending on dataset to extract y(target)
        
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        #print(feature)
        #print(emotion)
        x.append(feature)
        y.append(emotion)
    
    #print(x)
    #print("==>")
    #print(np.array(x))
    #print(y)
        # train_test_split(DATA, TARGET, test_size=test_size, random_state=9)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

#load_data(test_size=0.25)


# In[ ]:

print ("Start Splitting data")

start_time = datetime.datetime.now()

#Split the dataset into training and test
x_train, x_test, y_train, y_test=load_data("speech-emotion-recognition-ravdess-data/Actor_*/*.wav", test_size=0.2)
#print(x_train)
#print(y_train)

print ("End Splitting data")

end_time = datetime.datetime.now()
print('\n...Ran in ' + str(end_time - start_time) + ' secs flat...')


# In[ ]:


#Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

print(f'Features extracted: {x_train.shape}')


# In[ ]:

# Random Forest Classifier = slowest, but best accuracy

# Last Accuracy: 76.30%

from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
rf_model = RandomForestClassifier(n_estimators=200)

#Train the model
rf_model.fit(x_train,y_train)
print(rf_model)

#Predict for the test set
y_pred = rf_model.predict(x_test)

#Calculate the accuracy of the model
rf_accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(rf_accuracy*100))


# In[ ]:

# Multi Layer Perceptron Classifier = widely varying accuracy

# Multi-layer Perceptron Classifier is sensitive to feature scaling. Initially scaling seemed to stablise the accuracy for 
# train/test split of 0.2. However, for a split of 0.3, the scaler seems to lower the accuracy

# Last Accuracy: 75.56%

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#y_train = scaler.transform(y_train)
#y_test = scaler.transform(y_test)

#model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), 
#learning_rate='adaptive', max_iter=500)

mlp_model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), 
                    learning_rate='adaptive', max_iter=5000)
#print(model)

#Train the model
mlp_model.fit(x_train,y_train)
print(mlp_model)

#Predict for the test set
y_pred = mlp_model.predict(x_test)

#Calculate the accuracy of the model
mlp_accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(mlp_accuracy*100))


# In[ ]:

# Gradient Boosted Decision Trees (GBDT)
from sklearn.ensemble import GradientBoostingClassifier

# Last Accuracy: 74.81%

gbdt_model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
gbdt_model.fit(x_train, y_train)

print(gbdt_model)

y_pred=gbdt_model.predict(x_test)
gbdt_accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(gbdt_accuracy*100))


# In[ ]:

# Logistic Regression
# Last Accuracy: newton-cg (70.37%), liblinear(65%)

from sklearn.linear_model import LogisticRegression

#X, y = load_iris(return_X_y=True)

#lr_model = LogisticRegression(penalty='l1', solver='liblinear')
lr_model = LogisticRegression(penalty='l2', solver='newton-cg')

lr_model.fit(x_train, y_train)

print(lr_model)

y_pred=lr_model.predict(x_test)
lr_accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(lr_accuracy*100))


# In[ ]:

# Gaussian Naive-Bayes = Fast, but low accuracy
from sklearn.naive_bayes import GaussianNB

# Last Accuracy: 53.33%

gnb_model=GaussianNB()

gnb_model.fit(x_train,y_train)
print(gnb_model)

#Predict for the test set
y_pred = gnb_model.predict(x_test)

#Calculate the accuracy of the model
gnb_accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(gnb_accuracy*100))


# In[ ]:

import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

classifiers = ['MLP', 'RF', 'GBDT', 'GNB']
y_pos = np.arange(len(classifiers))

accuracy = [mlp_accuracy*100.0, rf_accuracy*100.0, gbdt_accuracy*100.0, gnb_accuracy*100.0]

#fig, ax1 = plt.subplots()
#rects1 = ax1.bar(100, 50, 10, label='Rural + Urban')

plt.bar(y_pos, accuracy, alpha=0.5)
plt.xticks(y_pos, classifiers)
plt.ylabel('Accuracy')
plt.title('Classifier')

plt.show()


# In[ ]:

# save and read models

import pickle
from datetime import datetime
import time

timestamp = time.time()
dt_object = datetime.fromtimestamp(timestamp)

workDir = '/Users/guest1/Downloads/'
model_file = 'RF_Model_' + str(dt_object) + '.h5'
#print(model_file)

#print(rf_model)

model_pkl = open(model_file, 'wb')
pickle.dump(rf_model, model_pkl)
model_pkl.close()

#rf_model.save_weights(workDir + "rf_model.h5")


# In[ ]:

# take a sample file and check for accuracy
# emotions={
#  '01':'neutral', '02':'calm', 03':'happy', '04':'sad',
#  '05':'angry', '06':'fearful', '07':'disgust', '08':'surprised'
#  }

x = []
file = "/Users/guest1/Downloads/speech-emotion-recognition-ravdess-data/Actor_01/03-01-02-01-02-01-01.wav"
    
print("Emotion found for this file: ")

file_name=os.path.basename(file)
#print(file_name)
        
feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
#print(feature)
    
x.append(feature)
print(mlp_model.predict(x))

print("Emotion originally coded for this file")
emotion=emotions[file_name.split("-")[2]]
print(emotion)


# In[ ]:

#to do.... real-time web-socket/block-wise reading
# https://librosa.github.io/librosa/ioformats.html

#Librosa’s load function is meant for the common case where you want to load an entire (fragment of a) recording 
#into memory, but some applications require more flexibility. In these cases, we recommend using soundfile 
#directly. Reading audio files using soundfile is similar to the method in librosa. One important difference is 
#that the read data is of shape (nb_samples, nb_channels) compared to (nb_channels, nb_samples) in 
#librosa.core.load. Also the signal is not resampled to 22050 Hz by default, hence it would need be transposed 
#and resampled for further processing in librosa. The following example is equivalent to 
#librosa.load(librosa.util.example_audio_file()):

import librosa
import soundfile as sf

# Get example audio file
filename = librosa.util.example_audio_file()

data, samplerate = sf.read(filename, dtype='float32')
data = data.T
data_22k = librosa.resample(data, samplerate, 22050)

---

#For large audio signals it could be beneficial to not load the whole audio file into memory. 
#Librosa 0.7 introduces a streaming interface, which can be used to work on short fragments of audio sequentially.
#librosa.core.stream cuts an input file into blocks of audio, which correspond to a given number of frames, 
#which can be iterated over as in the following example:

import librosa

sr = librosa.get_samplerate('/path/to/file.wav')

# Set the frame parameters to be equivalent to the librosa defaults
# in the file's native sampling rate
frame_length = (2048 * sr) // 22050
hop_length = (512 * sr) // 22050

# Stream the data, working on 128 frames at a time
stream = librosa.stream('path/to/file.wav',
                        block_length=128,
                        frame_length=frame_length,
                        hop_length=hop_length)

chromas = []
for y in stream:
   chroma_block = librosa.feature.chroma_stft(y=y, sr=sr,
                                              n_fft=frame_length,
                                              hop_length=hop_length,
                                              center=False)
   chromas.append(chromas)
    

    
#Download and read from URL:

import soundfile as sf
import io

from six.moves.urllib.request import urlopen

url = "https://raw.githubusercontent.com/librosa/librosa/master/tests/data/test1_44100.wav"

data, samplerate = sf.read(io.BytesIO(urlopen(url).read()))



# In[ ]:

# util to convert mp3 to WAV

# !pip install pydub
#!pip install ffmpeg
#!pip install ffprobe

# Convert mp3 to wav on the command line ==> "./ffmpeg -i InboundSampleRecording.mp3 InboundSampleRecording.wav"

import sys
sys.path.append('/Users/guest1/Downloads/ffmpeg')

import os
import glob
from pydub import AudioSegment

print(os.environ['PATH'])

#Path to all mp3 files
file = '/Users/guest1/Downloads/Karissa_Hobbs_-_09_-_Lets_Go_Fishin.mp3'

#for file in sorted(glob.iglob(files_path)):
name = file.split('.mp3')[0]
print(name)
sound=AudioSegment.from_mp3(file)
sound.export(name + '.wav',format='wav')


# In[ ]:

# => Unstable, errors and data loss

#!pip install SpeechRecognition
#!pip install sentiment
#!pip install portaudio
#!pip install pyaudio

import speech_recognition as sr
#import sentiment as s

#RECORD SOUND
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)
    #RECOGNISE AUDIO
    text = r.recognize_google(audio)
    print(text)
try:
    #PRINT THE SENTIMENT AS WELL AS CONFIDENCE
    #print(s.sentiment(text))
    print("You said: " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


# In[27]:




# In[ ]:



