import streamlit as st
from multiapp import MultiApp
from apps import home, instruction # import your app modules here

app = MultiApp()
app.add_app("Home Page", home.app)
app.add_app("Instructions Page", instruction.app) # Add all your pages here below
#app.add_app("Main Record Page", main3.app)

################################### General Helpful Notes and Codes #####################################################
# To run just put '/Users/yuxinzhu/.spyder-py3/Streamlit/streamlit-multiapps/app.py' in the terminal window

#SOMETIMES YOUR WORKING IN THE WRONG DIRECTORY AND YOUR FILES ARE IN ANOTHER DIRECTORY 
#To check your working directory use the code below:
# import os
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in working dir %r: %s" % (cwd, files))
#########################################################################################################################

################################### TO RECORD VOCAL DATA ################################################################
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
from PIL import Image

st.title("Multiple Sclerosis Record (MS Rec) Patient Website Application")
st.markdown(':sunglasses: :+1: :+1: :+1: **MS Rec made just for *you* ** :+1: :+1: :+1: :sunglasses:')
image = Image.open(r"/Users/yuxinzhu/.spyder-py3/StreamlitSD/logo1.png")
st.image(image, use_column_width=True)

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
print (sd.get_portaudio_version())
################################

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
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import sounddevice as sd
import wavio

st.title("Vocal Recording")
st.header("Record your vocal tasks and voice below!")

def create_spectrogram(voice_sample):
    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title(f"Graphic Visualization of the Recorded Vocal file: {voice_sample}")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(2,1,2)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout(pad=2.0)
    plt.show()
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

# def record(duration=5, fs=44100):
#     sd.default.samplerate = fs
#     sd.default.channels = 1
#     myrecording = sd.rec(int(duration * fs))
#     sd.wait(duration)
#     return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None

st.markdown('*Please select the duration and name of your vocal recording below*:')
filename = st.text_input("Choose a filename: ")
durationset = st.number_input("Recording duration", 0.0, 3600.0, 3.0)
if st.button(f"Click to Record"):
    if filename == "":
        st.warning("Choose a filename.")
    else:
        record_state = st.text("Recording...")
        duration=durationset
        fs = 44100
        sd.default.channels = 1
        myrecording = sd.rec(int(duration * fs))
        sd.wait()
        #myrecording = record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")

        path_myrecording = f"/Users/yuxinzhu/.spyder-py3/StreamlitSD/{filename}.wav"
       
        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Done! Saved sample as {filename}.wav")

        st.audio(read_audio(path_myrecording))

        fig = create_spectrogram(path_myrecording)
        st.pyplot(fig)

        ################ TO OBTAIN THE WEB-RECORDING .WAV SAMPLING RATE 
        from scipy.io.wavfile import read as read_wav
        import os
        os.chdir('/Users/yuxinzhu/.spyder-py3/StreamlitSD/') # change to the file directory
        sampling_rate, data1=read_wav(f"{filename}.wav") # enter your filename
        print (sampling_rate)
        st.markdown('*Vocal Recording Sampling Rate For:*  **output.wav**')
        st.markdown(sampling_rate)
        ################################
st.write("**NOTE:** Please continue to record even if there is an *starting error shown below. * Thank You !")

################ TO UPLOAD DATA 
st.title("Vocal Sample File Upload")
file_to_be_uploaded = st.file_uploader("Choose an audio file to upload", type="wav")
file_to_be_uploaded = st.file_uploader("Choose an audio file to upload", type="mp3")
################################

################ CONVERSION OF .mp3 to .wav 
mp3_audio = AudioSegment.from_file("/Users/yuxinzhu/.spyder-py3/StreamlitSD/Charlie.mp3", format="mp3") 
#wname = mktemp('.wav')  # use temporary file
#mp3_audio.export(wname, format="wav")  # convert to wav
filename1='/Users/yuxinzhu/.spyder-py3/StreamlitSD/Charlie.wav'     
OUTFILE = mp3_audio.export(filename1, format="wav")
################################

#########################################################################################################################


################################### Fundamental Frequency Code ###########################################################
st.title("Vocal Sample Graphs & Data")
fs=44100 #sampling rate
seconds=3 #duration of recording

fs,x = wav.read('/Users/yuxinzhu/.spyder-py3/StreamlitSD/output.wav') # try 'C:/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav'
t = fs/10000
dPoints = x.size # Number of data points on x-axis
y = x/(2*fs) #adjusting the y-axis to the input
R = np.linspace(0,t, dPoints) # From 0 to t, to get it to the scale I want

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(R, y)
plt.xlabel('t in seconds')
plt.ylabel('f(t)')

# ###############################################################################

# # Plotting on streamlit is very different, this code below makes the graphs show on the 
# # website front-end and the code for it to be on the front-end too!   
# with st.echo(code_location='below'):
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
    
#     #R= 2, 4, 6, 8
#     #y= 1, 2, 3, 4
#     #ax.plot(R, y)
    
#     fs,x = wav.read('/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav') # try 'C:/Users/yuxinzhu/.spyder-py3/Streamlit/output.wav'
#     t = fs/10000
#     dPoints = x.size # Number of data points on x-axis
#     y = x/(2*fs) #adjusting the y-axis to the input
#     R = np.linspace(0,t, dPoints) # From 0 to t, to get it to the scale I want
#     ax.plot(R, y)
#     #ax.set_xlim(0, 1)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("f(t)")

#     st.write(fig)
# ###############################################################################

################ Graph #1 
#Ploting code makes graph 1 show on the website front-end
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(R, y)
#ax.set_xlim(0, 2000)
ax.set_xlabel("t in seconds")
ax.set_ylabel("f(t)")
st.pyplot(fig) 
################################

################ Graph #2
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

#makes the graph 2 show on the website front-end
fig, ax = plt.subplots(figsize=(5, 2))
ax.plot(freqbin, amp/30000)
ax.set_xlim(0, 2000)
ax.set_xlabel("v in Hz")
ax.set_ylabel("abs(F(v))")
st.pyplot(fig) 
################################

#########################################################################################################################

################################### Fundamental Frequency Lowpass filter Code ############################################
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
#########################################################################################################################

################ Graph #3
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
################################

fpitch2 = freqbin2[np.argmax(amp2)]

if fpitch2 < 300:
    print('pitch value =',fpitch2, 'Hz')
   
else:
    fpitch2 = freqbin2[np.argmax(amp2)]
    print('pitch value =', fpitch2, 'Hz')


# The main app
app.run()

