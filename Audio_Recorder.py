import pyaudio
import wave
import sys
def record(RECORD_SECONDS,WAVE_OUTPUT_PATH):
    try:
        CHUNK = 1024 
        FORMAT = pyaudio.paFloat32
        CHANNELS = 2 
        RATE = 22050 #sample rate
        
        p = pyaudio.PyAudio()
        
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK) #buffer
        
        print("-------------------Recording--------------------------")
        
        frames = []
        
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data) # 2 bytes(16 bits) per channel
        
        print("------------------------Done Recording----------------------------")
        
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        wf = wave.open(WAVE_OUTPUT_PATH, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
    except:
        sys.stderr.write("-----------Invalid save file path provided--------------------\n\n")
        input("Press any key to exit\n")
        sys.exit(-1)