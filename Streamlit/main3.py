"""Code to Web Application"""
# To run just put 'streamlit run /Users/yuxinzhu/.spyder-py3/main3.py' in the terminal window

#SOMETIMES YOUR WORKING IN THE WRONG DIRECTORY AND YOUR FILES ARE IN ANOTHER DIRECTORY 
#To check your working directory use the code below:
# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in working dir %r: %s" % (cwd, files))

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
from pydub import AudioSegment
from tempfile import mktemp


st.title("SENIOR DESIGN APP DEMO TRIAL RUN :)")
#st.write just writes it
st.markdown('Streamlit is **really cool** :+1: :+1: :+1: :sunglasses:')

################ FOR LOADING SAMPLE DATA
DATE_TIME = "date/time"
DATA_URL = ("http://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz")

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
hour= st.number_input('hour', 0, 23, 10) #this lets us input a number in the fly out 
data = data[data[DATE_TIME].dt.hour == hour] #this allows us to look at pick ups at noon/12 only so it's like a filter

################ FOR DISPLAYING DATA
'Data', data #this allows data to be displayed 

if st.checkbox('Show Raw Data'): # this allows us to hide the raw data table, unless we/users manually click to view it 
    '## Raw Data at %sh' % hour, data


################ TO RECORD DATA 
import IPython.display as ipd
import soundfile as sf
from pydub import AudioSegment 
from glob import glob
from scipy.io import wavfile as wav
import numpy as np
import IPython.display as ipd
from IPython.display import Audio
import pyaudio
import wave

################# TO CHECK MICROPHONE USED
import sounddevice as sd
print (sd.query_devices())
# My computer has two microphones, one built-in and one via USB, and I want to record using the USB mic. 
# The Stream class has an input_device_index for selecting the device, but it's unclear how this index correlates to the devices. 
# For example, how do I know which device index 0 refers to? 
# You can use the code above to check! 
# As you can see from below terminal output, when I put my headset to mic jack , 
# Index 1 is available as input. 1 Jack Mic (2 in, 0 out)
# While default laptop audio microphone comes up as index 2
################################

st.title("Duration")
#duration = st.slider("Recording duration", 0.0, 3600.0, 3.0)
duration = st.number_input("Recording duration", 0.0, 3600.0, 3.0)

def record_and_predict(duration):
    filename = "recorded.wav"
    chunk = 1024
    FORMAT = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    record_seconds = duration
    chosen_device_index= 0
    p = pyaudio.PyAudio()
    # open stream object as input & output
    stream = p.open(format=FORMAT,
                    channels=channels,
                    rate=sample_rate,
                    input_device_index=chosen_device_index,
                    input=True,
                    output=True,
                    frames_per_buffer=chunk)
    frames = []
    print("Recording...")
    for i in range(int(44100 / chunk * record_seconds)):
        data = stream.read(chunk, exception_on_overflow=False)
        frames.append(data)
    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(sample_rate)
    wf.writeframes(b"".join(frames))
    wf.close()                                                            
    audio="recorded.wav"

    if audio is not None:
        audio = AudioSegment.from_wav(audio)
        n = len(audio) 
        counter = 1
        interval = 1 * 1000
        overlap = 0
        start = 0
        end = 0
        flag = 0
        for i in range(0,  n, interval): 
            if i == 0: 
                start = 0
                end = interval 
            else: 
                start = end  -overlap
                end = start + interval  
            if end >= n: 
                end = n 
                flag = 1            
            chunk = audio[start:end] 
            filename = '/Users/yuxinzhu/.spyder-py3/Streamlit/chunk'+str(counter)+'.wav'
            chunk.export(filename, format ="wav") 
            counter = counter + 1
        for name in glob('notes/test2/*.wav'):
            filename = name
            #print_prediction(filename)

if st.button("Start Recording"):
    with st.spinner("Recording..."):
        record_and_predict(duration)

################ TO UPLOAD DATA 
file_to_be_uploaded = st.file_uploader("Choose an audio file to upload", type="wav")
file_to_be_uploaded = st.file_uploader("Choose an audio file to upload", type="mp3")

############ Temporary Conversion of .mp3 to .wav 
mp3_audio = AudioSegment.from_file("/Users/yuxinzhu/.spyder-py3/Streamlit/Charlie.mp3", format="mp3") 
#wname = mktemp('.wav')  # use temporary file
#mp3_audio.export(wname, format="wav")  # convert to wav
filename1='/Users/yuxinzhu/.spyder-py3/Streamlit/Charlie.wav'     
OUTFILE = mp3_audio.export(filename1, format="wav")

################################################################### Fundamental Frequency Code #############################################################################
fs=44100 #sampling rate
seconds=3 #duration of recording

fs,x = wav.read('/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav') # try 'C:/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav'
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
    
    fs,x = wav.read('/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav') # try 'C:/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav'
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

def Filter2(signal):
    #fs = 44100
    lowcut = 70.0
    highcut = 300.0
    
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    
    order = 2
    
    b, a = scipy.signal.butter(order, [low,high], 'bandpass', analog=False)
    yy = scipy.signal.filtfilt(b,a, signal, axis = 0)
    
    return(yy)


plt.subplot(3,1,3)
fx = Filter(x)
#fx = Filter(channel) #THIS CALLS ON THE FUNCTION TO FILTER THE RECORDING
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

'''
pitch = freqbin2[np.argmax(amp2)]

if pitch > 300:
    fx = Filter2(channel)
    X2 = fft.fft(fx)
    amp2 = np.abs(X2)/fs
    pitch = freqbin2[np.argmax(amp2)]
    
print('filtered pitch value =', round(pitch,2), 'Hz')

'''

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

