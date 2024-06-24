import numpy as np
import cv2 # OpenCV
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import sys
import collections
import math
from scipy import ndimage
from sklearn.cluster import KMeans

# keras
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

from tensorflow.keras.optimizers import SGD


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin

def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def merge_rectangles(rectangles):
    spojeni_pravougaonici = []
    while rectangles:
        x1, y1, w1, h1 = rectangles.pop(0)  # Uzmi prvi pravougaonik iz liste
        #print("1-")
        #print(x1,y1,w1,h1)
        
        for i in range( len(rectangles) -1, -1, -1):
            x2, y2, w2, h2 = rectangles[i]
           #print(x2,y2,w2,h2)

            if (
                abs(x1 - x2) < 19
                and abs(y1 + h1 - y2) < 15
            ):
                # Pravougaonici se mogu spojiti
                #print("pronadjen je ")
                x1 = min(x1, x2)
                y1 = min(y1, y2)
                w1 = max(x1 + w1, x2 + w2) - x1
                h1 = max(y1 + h1, y2 + h2) - y1

                # Ukloni spojene pravougaonike iz liste
                rectangles.pop(i)
        
        # Dodaj spojeni pravougaonik u rezultat
        spojeni_pravougaonici.append((x1, y1, w1, h1))

    return spojeni_pravougaonici

def select_roi_with_distances(image_orig, image_bin):
    contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = [] # lista sortiranih regiona po X osi
    regions_array = []

    rectangles_regions = []
    contours_rect = []
    for i in range(len(contours) - 1):  # Preskačem poslednju konturu jer je to cijeli prozor
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)
        contours_rect.append((x,y,w,h))   #formiram listu pravougaonih kontura
    
    
    for i in range(len(contours) - 1):  # Preskačem poslednju konturu jer je to cijeli prozor
        contour = contours[i]
        x, y, w, h = cv2.boundingRect(contour)

        # Provera da li se trenutna kontura nalazi unutar drugog pravougaonika
        is_contour_inside = any(
            (cx, cy, cw, ch) for (cx, cy, cw, ch) in contours_rect if x > cx and y > cy and x + w < cx + cw and y + h < cy + ch
        )
        
        if not is_contour_inside:
            rectangles_regions.append((x,y,w,h))


    
    rectangles_regions = sorted(rectangles_regions,  key = lambda x : x[0])
    rectangles_regions.reverse()
    #print(rectangles_regions)
    rectangles_regions = merge_rectangles(rectangles_regions)
    #print(rectangles_regions)

    for rectangle in rectangles_regions:
        x, y, w, h = rectangle
        if w>10 and h>20:
            region = image_bin[y:y+h+1, x:x+w+1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    #print(regions_array)
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    # izdvojiti sortirane parametre opisujucih pravougaonika
    # izracunati rastojanja izmedju svih susednih regiona po X osi i dodati ih u niz rastojanja
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2]) # x_next - (x_current + w_current)
        region_distances.append(distance)
    
    return image_orig, sorted_regions, region_distances

def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) # dati ulaz
    y_train = np.array(y_train, np.float32) # zeljeni izlazi na date ulaze
    
    #print("\nTraining started...")
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    #print("\nTraining completed...")
    return ann

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()


def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result_with_spaces(outputs, alphabet, k_means):
    # odredjivanje indeksa grupe koja odgovara rastojanju izmedju reci
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    
    result = alphabet[winner(outputs[0])]
    # iterativno dodavanje prepoznatih elemenata
    # dodavanje space karaktera ako je rastojanje izmedju dva slova odgovara rastojanju izmedju reci
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result

def display_result(outputs, alphabet):
    result = ''
    for output in outputs:
        result += alphabet[winner(output)]
    return result


#Putanja do CSV fajla sa rjesenjima
csv_file = sys.argv[1] + "res.csv"

# Ucitavanje CSV fajla
df = pd.read_csv(csv_file)

def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return abs(len(str1) - len(str2)) + sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def total_hamming_distance(predictions, true_values):
    total_distance = 0
    
    for pred, true_val in zip(predictions, true_values):
        total_distance += hamming_distance(pred, true_val)
    
    return total_distance

true_value = []
my_value = []

sort_letters = []

for picture in df["file"]:
    img = load_image(sys.argv[1] + 'pictures/' + picture) # ucitavanje slike
    img_crop = img[170:300,250:830]
    gray_image = image_gray(img_crop)
    binary_image = image_bin(gray_image)
    #display_image(binary_image)
    selected_regions, letters, distances = select_roi_with_distances(img_crop, binary_image)
    #print("Broj prepoznatih regiona: ", len(letters))

    #display_image(selected_regions)

    if(picture == 'captcha_1.jpg'):
        sort_letters.append(letters[0]) 
        sort_letters.append(letters[1]) 
        sort_letters.append(letters[2]) 
        sort_letters.append(letters[3])  
    elif(picture == 'captcha_2.jpg'):
        sort_letters.append(letters[1])
        sort_letters.append(letters[2])
        sort_letters.append(letters[5])
    elif(picture == 'captcha_3.jpg'):
        sort_letters.append(letters[0])
        sort_letters.append(letters[1])
        sort_letters.append(letters[3])
        sort_letters.append(letters[6])
    elif(picture == 'captcha_4.jpg'):
        sort_letters.append(letters[0])
        sort_letters.append(letters[2])
        sort_letters.append(letters[3])
        sort_letters.append(letters[4])
        sort_letters.append(letters[8])
    elif(picture == 'captcha_5.jpg'):
        sort_letters.append(letters[0])
        sort_letters.append(letters[2])
        sort_letters.append(letters[5])
        sort_letters.append(letters[6])
        sort_letters.append(letters[9])
    elif(picture == 'captcha_6.jpg'):
        sort_letters.append(letters[1])
    elif(picture == 'captcha_7.jpg'):
        sort_letters.append(letters[3])
        sort_letters.append(letters[7])
    elif(picture == 'captcha_8.jpg'):
        sort_letters.append(letters[3])
        sort_letters.append(letters[4])
        sort_letters.append(letters[6])
        sort_letters.append(letters[10])
    elif(picture == 'captcha_9.jpg'):
        sort_letters.append(letters[6])
    elif(picture =='captcha_10.jpg'):
        sort_letters.append(letters[2])
    

alphabet = ['k','l','e','p','t','o','j','š','v','s','a','z','m','ě','d','c','h','á','n','í','č','ť','ý','i','r','é','ž','y','b','ú']


inputs = prepare_for_ann(sort_letters)
outputs = convert_output(alphabet)
ann = create_ann(output_size=len(sort_letters))
ann = train_ann(ann, inputs, outputs, epochs=1000)

#result = ann.predict(np.array(inputs, np.float32))
#print(display_result(result, alphabet))

for picture in df["file"]:
    img = load_image(sys.argv[1] + 'pictures/' + picture) # ucitavanje slike
    img_crop = img[170:300,250:830]
    gray_image = image_gray(img_crop)
    binary_image = image_bin(gray_image)
    #display_image(binary_image)
    selected_regions, letters, distances = select_roi_with_distances(img_crop, binary_image)

    inputs = prepare_for_ann(letters)

    row = df[df["file"] == picture]
    true_text = row["text"].values[0]
    true_value.append(true_text)
    #print(distances)
    if any(d > 10 for d in distances):
        distances = np.array(distances).reshape(len(distances), 1)
        k_means = KMeans(n_clusters=2,  n_init=10)
        k_means.fit(distances)

        results = ann.predict(np.array(inputs, np.float32))
        print(picture + "-" + true_text + "-" + display_result_with_spaces(results, alphabet, k_means))
        my_value.append(display_result_with_spaces(results, alphabet, k_means))
    else:
        results = ann.predict(np.array(inputs, np.float32))
        print(picture + "-" + true_text + "-" + display_result(results, alphabet))
        my_value.append(display_result(results, alphabet))

#print("moje vrijednosti:",  my_value)
#print("tacne vrijednosti:", true_value)
print(total_hamming_distance(my_value,true_value))