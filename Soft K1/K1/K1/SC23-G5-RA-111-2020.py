import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt  
import pandas as pd
import sys

#Putanja do CSV fajla sa rjesenjima
csv_file = "bulbasaur_count.csv"

# Ucitavanje CSV fajla
df = pd.read_csv(csv_file)

#funkcija za racunanje MAE
def mean_absolute_error(measured_values, true_values):
    n = len(measured_values)
    if n != len(true_values):
        raise ValueError("The number of measured values must match the number of true values.")
    
    absolute_errors = [abs(measured - true) for measured, true in zip(measured_values, true_values)]
    mae = sum(absolute_errors) / n
    
    return mae

measured_v = []
true_v = []

for picture in df["Naziv slike"]:
    img = cv2.imread(sys.argv[1] + picture) # ucitavanje slike
    img_crop = img[200:,::]
    #plt.imshow(img_crop)  # prikazivanje isjecene slike
    #plt.show()
    image = cv2.cvtColor(img_crop,cv2.COLOR_BGR2RGB)
    image = cv2.convertScaleAbs(image, alpha= 2)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #plt.imshow(image)  # prikazivanje RGB slike sa osvjetljenjem alpha=2
    #plt.show()
    #plt.imshow(image_gray, 'gray')  # prikazivanje gray slike
    #plt.show()

    image_bin = cv2.adaptiveThreshold(image_gray,255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    imagewitherode = cv2.erode(image_bin,kernel,iterations=3)
    #plt.imshow(imagewitherode, 'gray')  # prikazivanje binarne slike sa erozijom
    #plt.show()

    contours, hierarchy = cv2.findContours(imagewitherode, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours_bulbasaur= []

    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour) 
        height, width = size

        if width > 40 and width < 160 and height > 40 and height < 155:
            contours_bulbasaur.append(contour) 
    
    row = df[df["Naziv slike"] == picture]
    bulbasaur_count = row["Broj bulbasaur-a"].values[0]

    measured_v.append(len(contours_bulbasaur))
    true_v.append(bulbasaur_count)
    print(picture + "-" + str(bulbasaur_count) + "-" + str(len(contours_bulbasaur)))
    img_copy = image.copy()
    cv2.drawContours(img_copy, contours_bulbasaur, -1, (255, 0, 0), 1)
    plt.imshow(img_copy)
    plt.show()

#print(measured_v)
#print(true_v)
mae = mean_absolute_error(measured_v,true_v)
print(str(mae))




