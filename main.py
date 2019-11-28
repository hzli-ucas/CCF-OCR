import cv2
import location_card
import location_text
import recognition

import os


test_path = './images'
test_files = [file for file in os.listdir(test_path) if file.endswith('.jpg')]

for file_name in test_files:
    img = cv2.imread(os.path.join(test_path, file_name), 0)

    vertexes = location_card.getCardVertexes(img)
    text_imgs = location_text.getTextFromCard(img, vertexes)
    contents = recognition.readTextImages(text_imgs)

    print(contents)
    cv2.imshow('img', img)
    cv2.waitKey(0)
