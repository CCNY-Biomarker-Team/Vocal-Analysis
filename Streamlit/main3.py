"""Code to Web Browser"""
# To run just put 'streamlit run /Users/yuxinzhu/.spyder-py3/main3.py' in the terminal window

import sounddevice as sd
from scipy.io.wavfile import write
from scipy.signal import filtfilt
import scipy
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk

st.title("SENIOR DESIGN APP DEMO TRIAL RUN :)")
#st.write just writes it
st.markdown('Streamlit is **really cool** :+1: :+1: :+1: :sunglasses:')

# LOADING SAMPLE DATA, (for cognative fatigue tasks if we finsih the vocal part of the app)
DATE_TIME = "date/time"
DATA_URL = (
    "http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz")

@st.cache(persist=True) #code like this is usually slow to run b/c have to go web and download and transform etc. 
#but it's not now, b/c of this st.cache function that stores the data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[DATE_TIME] = pd.to_datetime(data[DATE_TIME]) #convert one column of string data to date/time
    return data

data = load_data(100000) #this allows us to load 100000 lines of data from online
#hour = 12
#hour= st.slider('hour', 0, 23, 10) #this allows users to change the time instead of setting it at just one time
#hour= st.sidebar.slider('hour', 0, 23, 10) #this lets us move the slider to a side bar that flies out 
hour= st.sidebar.number_input('hour', 0, 23, 10) #this lets us input a number in the fly out 
data = data[data[DATE_TIME].dt.hour == hour] #this allows us to look at pick ups at noon/12 only so it's like a filter

# DISPLAYING DATA 
'Data', data #this allows data to be displayed 

if st.checkbox('Show Raw Data'): # this allows us to hide the raw data table, unless we/users manually click to view it 
    '## Raw Data at %sh' % hour, data


# RECORD DATA or UPLOAD DATA 
import pyaudio
import wave
import sys

def record(duration):
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = duration
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
    channels=channels,
    rate=sample_rate,
    input=True,
    output=True,
    frames_per_buffer=chunk)
    frames = []
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk)
        frames.append(data)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(filename, "wb") # "wb" means write only mode, but there's rb= read only mode
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close() 
        audio="recorded.wav"

st.sidebar.title("Duration")
duration = st.sidebar.slider("Recording duration", 0.0, 3600.0, 3.0)

if st.button("Start Recording"):
	with st.spinner("Recording..."):
		record(duration)

file_to_be_uploaded = st.file_uploader("Choose an audio file to upload", type="wav")

'''
# To convert mp3 to wav use the following code

from pydub import AudioSegment
sound = AudioSegment.from_mp3("/path/to/file.mp3")
sound.export("/output/path/file.wav", format="wav")
'''

'''
import os
cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

SOMETIMES YOUR WORKING IN THE WRONG DIRECTORY AND YOUR FILES ARE IN ANOTHER DIRECTORY 
'''

################################################################### Fundamental Frequency Code #############################################################################
fs=44100 #sampling rate
seconds=3 #duration of recording

fs,x = wav.read('output.wav') 
t = fs/10000
dPoints = x.size # Number of data points on x-axis
y = x/(2*fs) #adjusting the y-axis to the input
R = np.linspace(0,t, dPoints) # From 0 to t, to get it to the scale I want

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(R, y)
plt.xlabel('t in seconds')
plt.ylabel('f(t)')


# Plotting on streamlit is very different, this code below makes the graphs show on the 
# website front-end and the code for it to be on the front-end too!   
with st.echo(code_location='below'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    #R= 2, 4, 6, 8
    #y= 1, 2, 3, 4
    #ax.plot(R, y)
    
    fs,x = wav.read('output.wav') 
    t = fs/10000
    dPoints = x.size # Number of data points on x-axis
    y = x/(2*fs) #adjusting the y-axis to the input
    R = np.linspace(0,t, dPoints) # From 0 to t, to get it to the scale I want
    ax.plot(R, y)
    #ax.set_xlim(0, 1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("f(t)")

    st.write(fig)
###############################################################################

#Ploting code makes the graphs show on the website front-end
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(R, y)
#ax.set_xlim(0, 2000)
ax.set_xlabel("t in seconds")
ax.set_ylabel("f(t)")
st.pyplot(fig) 
###############################################################################

X = fft.fft(x)
N=X.size
freqbin=np.linspace(0,fs,N)
plt.subplot(3,1,2)
amp=np.abs(X)
plt.plot(freqbin, amp/30000)
plt.xlim([0, 2000])
plt.xlabel('v in Hz')
plt.ylabel('abs(F(v))')

fpitch = freqbin[np.argmax(amp)]
print('pitch value =', fpitch, 'Hz')

#Ploting code makes the graphs show on the website front-end
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(freqbin, amp/30000)
ax.set_xlim(0, 2000)
ax.set_xlabel("v in Hz")
ax.set_ylabel("abs(F(v))")
st.pyplot(fig) 

############################## Signl With Lowpass filter ########################
def Filter(signal):
    #fs = 44100
    lowcut = 70.0
    highcut = 300.0
    
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    
    order = 4
    
    b, a = scipy.signal.butter(order, [low,high], 'bandpass', analog=False)
    y = scipy.signal.filtfilt(b,a, signal, axis = 0)
    
    return(y)

plt.subplot(3,1,3)
fx = Filter(x)
X2 = fft.fft(fx) #converting to freq domain, a complex value
N2=X2.size #taking the length of the sample
freqbin2=np.linspace(0,fs,N2)
amp2=np.abs(X2) #taking only the magnitude portion of the complex value
plt.xlim([0, 600])
plt.xlabel('v in Hz')
plt.ylabel('abs(F(v))')
plt.plot(freqbin2, amp2/30000)

plt.tight_layout(pad=2.0)
plt.show()

#Ploting code makes the graphs show on the website front-end
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(freqbin2, amp2/30000)
ax.set_xlim(0, 600)
ax.set_xlabel("v in Hz")
ax.set_ylabel("abs(F(v))")
st.pyplot(fig) 


fpitch2 = freqbin2[np.argmax(amp2)]

if fpitch2 < 300:
    print('pitch value =',fpitch2, 'Hz')
   
else:
    fpitch2 = freqbin2[np.argmax(amp2)]
    print('pitch value =', fpitch2, 'Hz')



############################# FOR COMPARING VARIABILITY ############################### 
t=1,2,3,4
S=2,4,6,8
I=1,1,1,1
R=1,2,3,4

def plotsir(t, S, I, R):
  f, ax = plt.subplots(1,1,figsize=(10,4))
  ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='High')
  ax.plot(t, I, 'y', alpha=0.7, linewidth=2, label='Medium')
  ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Low')

  ax.set_xlabel('Time (days)')

  ax.yaxis.set_tick_params(length=0)
  ax.xaxis.set_tick_params(length=0)
  ax.grid(b=True, which='major', c='w', lw=2, ls='-')
  legend = ax.legend()
  legend.get_frame().set_alpha(0.5)
  for spine in ('top', 'right', 'bottom', 'left'):
      ax.spines[spine].set_visible(False)
      st.pyplot()

plotsir(t, S, I, R)

