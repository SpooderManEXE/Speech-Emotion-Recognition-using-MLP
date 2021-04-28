import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import feature
import speech_recognition as sr

def predict(filename):
    fmodel_path = 'Emotion_Voice_Detection_Model.h5'
    result=[]
    xtr = joblib.load('X.train')
    scaler = StandardScaler()
    scaler.fit_transform(xtr)
    clf2 = joblib.load(fmodel_path)
    j=np.array(feature.extract(filename))
    k = np.expand_dims(j, axis=0)
    k = scaler.transform(k)
    result.append(clf2.predict(k)[0])

    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
        try:
            text = r.recognize_google(audio_data)
        except sr.UnknownValueError:  
            text="*I could not understand audio*" 
        result.append(text)
    print(result)
