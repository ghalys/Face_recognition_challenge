import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from scipy.io import loadmat
import shutil



original_path = "C:/Users/asent/Desktop/S4_Barcelona/FaceDetection/FACE_DETECTION_CHALLENGE/"
imgPath     = original_path + "TRAINING/"

# # We start loading the data
# challenge = loadmat('AGC_Challenge3_Training.mat')
# data      = challenge['AGC_Challenge3_TRAINING'][0]
# data_extended = []

# for item in data:
#     label = item[0][0][0]
#     image = item[1][0]
#     # Handling faces, assuming faces data is in item[2]
#     faces = item[2] if len(item[2]) > 0 else []
#     data_extended.append([label, image, faces])

# # Convert the extended list to a DataFrame
# df_extended = pd.DataFrame(data_extended, columns=['Label', 'Image', 'Faces'])


# def count_version(df_extended):
#     images = [0 for i in range(81)] #index is the label
#     count_ = [0 for i in range(81)]

#     for k in range(len(df_extended['Label'])):
#         label = df_extended['Label'][k]
#         if label ==-1:
#             count_[0] +=1
#         else:
#             count_[label] +=1
#             images[label]= df_extended['Image'][k]
#     return(images,count_)

# images,count_ = count_version(df_extended)
# images_needed = []
# for id in range(len(count_)):
#     if count_[id]<10:
#         images_needed.append(id)
# print(images_needed)
            
      



        
# label_needed = 29
# for k in range(len(df_extended['Label'])):
#     if df_extended['Label'][k]== label_needed:
#         new_folder = os.path.join(original_path+"People_Needed/", str(label_needed))
#         if not os.path.exists(new_folder):
#             os.makedirs(new_folder)
#         path_to_image = imgPath+df_extended['Image'][k]
#         destination_photo = os.path.join(new_folder, os.path.basename(path_to_image))
#         shutil.copy(path_to_image, destination_photo)


def download_images(query,id, num_images, folder_path):
    query = query.replace(' ', '+')  # Remplacer les espaces par '+'
    url = f"https://www.google.com/search?hl=en&q={query}&tbm=isch"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    images = [img['src'] for img in soup.find_all('img')]

    #define the name of the folder
    folder_name = os.path.join(folder_path, str(id)+"_"+query.replace('+', '_'))
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)

    # Télécharger les premières 'num_images' images
    for i, img_url in enumerate(images[:num_images]):
        try:
            img_data = requests.get(img_url).content
            with open(os.path.join(folder_name, f'image_{i+1}.jpg'), 'wb') as file:
                file.write(img_data)
            print(f"Image {i+1} téléchargée dans {folder_name}")
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {i+1}: {e}")

# id = 60
# name = "Harry Connick Jr"
# download_images(name,id, 20, "Dataset")

# l =[8,9,14,26,39,41,60]


# image_name_ ="image_A"
# i = 1201
# folder_name = original_path + "Dataset"
# for name_folder in os.listdir(folder_name):
#     parts = name_folder.split("_")
#     label = parts[0]
#     path = original_path + "Dataset/" +name_folder
#     for filename in os.listdir(path):
#         if filename.endswith(".jpg"):  
#             nouveau_nom_fichier = f"{image_name_}{str(i)}.jpg"  # i est le compteur, peut être ajusté selon vos besoins
            
#             ancien_chemin = os.path.join(path, filename)
#             nouveau_chemin = os.path.join(path, nouveau_nom_fichier)
            
#             # rename the file
#             shutil.move(ancien_chemin, nouveau_chemin)
            
#             i += 1
    
    
# folder_name = original_path + "Dataset"
# for name_folder in os.listdir(folder_name):
#     parts = name_folder.split("_")
#     label = parts[0]
#     path = original_path + "Dataset/" +name_folder
#     for filename in os.listdir(path):
#         if filename.endswith(".jpg"): 
#             ancien_chemin = os.path.join(path, filename)
#             nouveau_chemin = os.path.join(original_path+"OurImages", filename)
#             shutil.copy(ancien_chemin, nouveau_chemin)

import cv2 as cv
#-- Face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

#parameter for the detection
scaleFactor  = 1.05
minNeighbors = 6
minsize      = 20
# maxsize      = 700

def MyFaceDetectionFunction(img): #from lab 1
    frame_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    #-- Detect faces
    faces = face_cascade.detectMultiScale(
        frame_gray, 
        scaleFactor  = scaleFactor, 
        minNeighbors = minNeighbors, 
        minSize      = (minsize, minsize),
        # maxSize      = (maxsize, maxsize)
    )
    return faces

def select_two_biggest_faces(faces):
    areas = []
    M =[]
    for (x,y,w,h) in faces:
        x1 = x
        y1 = y
        x2 = x1 + w 
        y2 = y1 + h
        M.append([x1,y1,x2,y2])
        areas.append(w*h)
        
    if len(M)>2:
        print("ooook")
        first_face = 0
        second_face = 1
        if areas[first_face]<areas[second_face]: #check if the biggest face is on index 2, then swap index
            first_face, second_face = 1, 0
        for j in range(2,len(M)):
            if areas[j]>areas[first_face]:
                second_face = first_face
                first_face  = j
            elif areas[j]> areas[second_face]:
                second_face = j
        M = M[first_face],M[second_face]
    return M

def See_detection(img,M):
    """draw the rectangle on the original image and save the image in the saving folder"""
    # M2 = MyFaceDetectionFunction(img)
    for rect in M:
        cv.rectangle(img,(rect[0],rect[1]),(rect[2],rect[3]),(0,0,255),4)
    cv.imshow("ok",img)
    cv.waitKey(3000)
    cv.destroyAllWindows()
    
# folder_name = original_path + "OurImages/"
# # folder_name = original_path + "TRAINING/"
# image_name_ = "image_A1202.jpg"
# img = cv.imread(folder_name+image_name_)
# M1 =MyFaceDetectionFunction(img)
# M = select_two_biggest_faces(M1)
# See_detection(img,M)


import scipy.io as sio
import numpy as np
new_data=[]
i=1
folder_name = original_path + "Dataset"
for name_folder in os.listdir(folder_name):
    parts = name_folder.split("_")
    label = parts[0]
    path = original_path + "Dataset/" +name_folder
    
    for filename in os.listdir(path):
        if filename.endswith(".jpg"): 
            image = filename
            print(filename)
            img = cv.imread(path+"/"+filename)
            M =MyFaceDetectionFunction(img)
            faces = select_two_biggest_faces(M)
            if len(faces)==1:
                new_data.append([image,label,faces[0]])
                
# # print(new_data[0])
# data_dict = {'donnees': new_data}

# # Enregistrez les données dans un fichier MATLAB
# sio.savemat('new_images.mat', data_dict)