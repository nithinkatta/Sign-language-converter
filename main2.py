import cv2
import pygame
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import google.generativeai as genai
import numpy as np
import math
from io import BytesIO
import time
from gtts import gTTS

# Gemini Ai google api key (very confidential)
GOOGLE_API_KEY = 'AIzaSyBaj3yH1P6dNttlssEt_aAptFpLed4PoKw'
genai.configure(api_key=GOOGLE_API_KEY)

model = genai.GenerativeModel('gemini-pro')


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
# loaded_model = tf.keras.models.load_model('model.h5')
classifier = Classifier("Model/Model_1/keras_model.h5", "Model/Model_1/labels.txt")
offset = 20
imgSize = 256
counter = 0

labels = ["Hello", "I am fine", "No", "How", "Please", "Thank you", "Yes"]

# Stores the update gestures
Input = []
s = ""
flag = True
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        flag = True

        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        # Inside the loop where you resize the cropped image
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            # print("wCal:", wCal)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            # print("imgResizeShape:", imgResizeShape)
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
            imgWhite = cv2.resize(imgWhite, (256, 256))  # Resize to match model input shape
            # print("imgWhiteShape:", imgWhite.shape)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            # print("hCal:", hCal)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            # print("imgResizeShape:", imgResizeShape)
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize
            imgWhite = cv2.resize(imgWhite, (256, 256))  # Resize to match model input shape
            # print("imgWhiteShape:", imgWhite.shape)
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction, index)

        if labels[-1] != labels[index]:
            Input.append(labels[index])

        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                      cv2.FILLED)

        cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # cv2.imshow('ImageCrop', imgCrop)
        # cv2.imshow('ImageWhite', imgWhite)
    else:
        if Input!=[]:

            if flag:
                s = ""
                for i in set(Input):
                    s += i+" "
                response = model.generate_content("correct the sentence without explanation : " + s)
                print(response.text)
                s = response.text
                flag = False
                # print(type(response))
            # Input = []

                tts = gTTS(text=s, lang='en')

                # Create an in-memory file-like object
                audio_buffer = BytesIO()

                # Write the audio content to the file-like object
                tts.write_to_fp(audio_buffer)

                # Reset the buffer position to the beginning
                audio_buffer.seek(0)

                # Initialize pygame mixer
                pygame.mixer.init()

                # Load the audio buffer into pygame mixer
                pygame.mixer.music.load(audio_buffer)

                # Play the audio
                pygame.mixer.music.play()

                # Wait for the audio to finish playing
                while pygame.mixer.music.get_busy():
                    time.sleep(1)

            cv2.putText(imgOutput,s,(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    cv2.imshow('Image', imgOutput)
    cv2.waitKey(1)
