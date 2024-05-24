from flask import Flask,render_template,Response,jsonify, request
# import cv2
from googletrans import Translator
import cv2 
import pygame
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

import google.generativeai as genai     # google ai
import numpy as np  
import math
from io import BytesIO 
import time
from gtts import gTTS 

# from flask_socketio import SocketIO, emit

#Translation
from google.cloud import translate_v2 as translate 
translate_client = translate.Client()

# Gemini Ai google api key (very confidential)
GOOGLE_API_KEY = 'AIzaSyA9CwtsrKKi_JfJufpTVPj3WvfT8hWDbkQ'
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

#opencv
# cap = cv2.VideoCapture('http://192.0.0.4:8080/video')  # mobile camera
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2) 

# loaded_model = tf.keras.models.load_model('model.h5')
classifier = Classifier("Model/Model_6/keras_model.h5", "Model/Model_6/labels.txt")
offset = 20
imgSize = 256
counter = 0

# labels = ["Hello", "I am fine", "No", "How", "Please", "Thank you", "Yes"]
# labels_1 = ["Hello","How", "Iamfine","You","","Hold"]   # model 3
# labels = ["Thank you","Argue","Hi"]  # model 4
# labels_1 = ["Home","No","Pray","Hungry","Family","Help","Time","Car","Yes","Hello","Money","Love","Water","Sorry","I love you"]
labels_1 = ["Hi","How","You"]
labels_2 = ["A","F","R","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","Hold"]  

# Stores the update gestures
Input = []

s = ""
flag = True
prev = "Welcome"
language = "en" 
 
app=Flask(__name__)
# camera=cv2.VideoCapture(0)

def fun1():
    return Classifier("Model/Model_6/keras_model.h5", "Model/Model_6/labels.txt")
def fun2():
    return Classifier("Model/Alphabets_2/keras_model.h5", "Model/Alphabets_2/labels.txt")

def generate_frames(classifier,labels):
    Input = []
    s = ""
    flag = True
    global prev
    # prev = ""
    while True:
        
        ## read the camera frame
        success,img=cap.read()

        # imgOutput = img.copy()
        # img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)# mobile view
        hands,img = detector.findHands(img,False)  
        if hands:
            flag = True 
 
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            # imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
            # imgCropShape = imgCrop.shape
  
            # aspectRatio = h/w  

            prediction,index = classifier.getPrediction(img,draw=False)
 
            if labels[-1] != labels[index]: 
                Input.append(labels[index]) 
            
            cv2.rectangle(img, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                      cv2.FILLED)            
            cv2.putText(img, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            # cv2.rectangle(img, (x - offset, y - offset),   (x + w + offset, y + h + offset), (0, 255, 0), 4)
        else:
            if Input!=[]:
                if flag:
                    s = ""
                    for i in set(Input):
                        if i!="Hold":
                            s+=i+" "
                    print(s) 
                    if s!=" ":
                        response = model.generate_content("Correct the sentence without explanation or if it is correct just display the same : "+s)
                        print(response.text)
                    s = response.text
                    flag = False 

                    Input = []
                    prev = s 

        # cv2.putText(img,prev,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
        # change_prev(prev)
        # if not success:
        #     break
        # else:
        ret,buffer=cv2.imencode('.jpg',img)
        img=buffer.tobytes() 

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
   
# def change_prev(prev):
 
# def translate_text(text, target_language='en'):
#     translator = Translator()
#     translation = translator.translate(text, dest=target_language)
#     return translation.text

@app.route('/', methods=['GET','POST'])  
def index():
    # global prev 

    global language
    # selected_values = []
    if request.method == 'POST': 
        language = request.form.get('radio')
        # language = selected_values[-1]
    global prev
    # global language

    if data['language'] != language:
        translation = translate_client.translate(prev,source_language=data['language'], target_language=language)
    
        data['description'] = translation['translatedText']
        data['language'] = language
        prev = translation['translatedText']
        language = data['language']
    else:
        data['description'] = prev 
        prev = data['description']


    tts = gTTS(text=prev,lang=language)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    pygame.mixer.init()
    pygame.mixer.music.load(audio_buffer)  
    pygame.mixer.music.play()
    # return jsonify(data)
    return render_template('index.html',data=data,language = language)

@app.route('/video')
def video():
    classifier = fun1() 
    labels = labels_1
    return Response(generate_frames(classifier,labels),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video2')
def video2():
    classifier = fun2()
    labels = labels_2
    return Response(generate_frames(classifier,labels),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/video2')
def video11():
    classifier = fun1()
    labels = labels_1
    return Response(generate_frames(classifier,labels),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video/video') 
def video22():
    classifier = fun2()
    labels = labels_2
    return Response(generate_frames(classifier,labels),mimetype='multipart/x-mixed-replace; boundary=frame')

data = {
    'description': "Meaningful description displays here after detection of gestures...",
    'language'  : "en"
}
 
@app.route('/update', methods=['POST'])
def update_data():
    # Increment count
    global prev
    global language

    if data['language'] != language:
        translation = translate_client.translate(prev,source_language=data['language'], target_language=language)
    
        data['description'] = translation['translatedText']
        data['language'] = language
        prev = translation['translatedText']
        language = data['language']
    else:
        data['description'] = prev
        prev = data['description']

    # tts = gTTS(text=prev,lang=language)
    # audio_buffer = BytesIO()
    # tts.write_to_fp(audio_buffer)
    # audio_buffer.seek(0)
    # pygame.mixer.init()
    # pygame.mixer.music.load(audio_buffer)
    # pygame.mixer.music.play()
    return jsonify(data)

@app.route('/listen', methods=['POST'])
def listen():  
    tts = gTTS(text=prev,lang=language)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)

    # audio_path = 'audio.mp3'
    # tts.save(audio_path)
    # socketio.emit('audio', {'audio': audio_path},broadcast=True)

    audio_buffer.seek(0) 
    pygame.mixer.init()
    pygame.mixer.music.load(audio_buffer)
    pygame.mixer.music.play()
    # return jsonify(data)
    return render_template('index.html',data=data,language = language)

if __name__=="__main__":
    # app.run(debug=True,host='192.168.145.140', port=8080) # for mobile view
    app.run(debug=True) # for desktop view  