import os
import numpy as np
import cv2 # OpenCV
from sklearn.svm import SVC # SVM klasifikator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier # KNN
import matplotlib.pyplot as plt
import pandas as pd
import sys

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

train_dir = 'data2/pictures/'

pos_imgs = []
neg_imgs = []

for img_name in os.listdir(train_dir):
    img_path = os.path.join(train_dir, img_name)
    img = load_image(img_path)
    #print(img.shape)
    if 'p_' in img_name:
        pos_imgs.append(img)
    elif 'n_' in img_name:
        neg_imgs.append(img)
        
#print("Positive images #: ", len(pos_imgs))
#print("Negative images #: ", len(neg_imgs))


pos_features = []
neg_features = []
labels = []

nbins = 9 # broj binova
cell_size = (8, 8) # broj piksela po celiji
block_size = (2, 2) # broj celija po bloku

hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1], 
                                  img.shape[0] // cell_size[0] * cell_size[0]),
                        _blockSize=(block_size[1] * cell_size[1],
                                    block_size[0] * cell_size[0]),
                        _blockStride=(cell_size[1], cell_size[0]),
                        _cellSize=(cell_size[1], cell_size[0]),
                        _nbins=nbins)

for img in pos_imgs:
    pos_features.append(hog.compute(img))
    labels.append(1)

for img in neg_imgs:
    neg_features.append(hog.compute(img))
    labels.append(0)

pos_features = np.array(pos_features)
neg_features = np.array(neg_features)
x = np.vstack((pos_features, neg_features))
y = np.array(labels)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#print('Train shape: ', x_train.shape, y_train.shape)
#print('Test shape: ', x_test.shape, y_test.shape)

clf_svm = SVC(kernel='linear', probability=True) 
clf_svm.fit(x_train, y_train)
y_train_pred = clf_svm.predict(x_train)
y_test_pred = clf_svm.predict(x_test)
#print("Train accuracy: ", accuracy_score(y_train, y_train_pred))
#print("Validation accuracy: ", accuracy_score(y_test, y_test_pred))


def detect_line(img):
    # detekcija koordinata linije koristeci Hough transformaciju
    img = img[80:,::]
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(gray_img,'gray')
    #plt.show()
    edges_img = cv2.Canny(gray_img, 30, 130, apertureSize=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges_img = cv2.dilate(edges_img,kernel,iterations=1)
       
    # minimalna duzina linije
    min_line_length = 50
    
    # Hough transformacija
    lines = cv2.HoughLinesP(image=edges_img, rho=1, theta=np.pi/180, threshold=10, lines=np.array([]),
                            minLineLength=min_line_length, maxLineGap=20)
    
     
    filtered_lines = [line[0] for line in lines if line[0][0] == line[0][2]]   
    
    if filtered_lines:
            print(filtered_lines[0])
            print("Filtrirane linije [[x1 y1 x2 y2]]: \n", filtered_lines[0])
                        
            x1 = filtered_lines[0][0]
            y1 = 2080 - filtered_lines[0][1]
            x2 = filtered_lines[0][2]
            y2 = 2080   
    else:
        x1=0
        y1=0
        x2=0
        y2=0
       
    #plt.imshow(edges_img, "gray")
    #plt.show()
    return (x1, y1, x2, y2)


def detect_cross(right_x, xx):   
    print(xx , "-", right_x , "=" , xx-right_x)
    return 0.0 <= (xx - right_x) <= 119

def process_video(video_path, hog_descriptor, classifier):
    # procesiranje jednog videa
    sum_of_cars = 0
      
    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_num) # indeksiranje frejmova  
    
    # analiza videa frejm po frejm
    while True:
        frame_num += 1
        grabbed, frame = cap.read()

        # ako frejm nije zahvacen
        if not grabbed:
            break
        
        print(str(frame_num))
        line_coords = detect_line(frame)
                      
        #posto su jednacine prave vetrikalne linije nemamo parametre k i n vec je to samo prava recimo x=2
            
        # izdvajanje krajnjih y koordinata linije i koliko je x
        xx = line_coords[0]
        line_up_y = line_coords[3]
        line_down_y = line_coords[1]
        print("y1 = ", line_up_y)
        print("y2 = ", line_down_y)
        
        
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image = image[80:,::]
        #plt.imshow(image)
        #plt.show()
        frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        #plt.imshow(frame_gray, "gray")
        #plt.show()

        #frame_bin = cv2.adaptiveThreshold(frame_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
        #plt.imshow(frame_bin, 'gray')
        #plt.show()

        lower_threshold = 73
        upper_threshold = 130
        frame_bin = cv2.inRange(frame_gray, lower_threshold, upper_threshold)
        frame_bin = cv2.bitwise_not(frame_bin) 
        #plt.imshow(frame_bin, 'gray')
        #plt.show()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
        frame_bin = cv2.dilate(frame_bin, kernel, iterations=2)
        frame_bin = cv2.erode(frame_bin,kernel,iterations=1)
        #plt.imshow(frame_bin, 'gray')
        #plt.show()
        
        contours, _ = cv2.findContours(frame_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_new= []
        for contour in contours:
            center, size, angle = cv2.minAreaRect(contour) 
            height, width = size
            #print(height, width)

            if width > 100 and width < 1200 and height > 100 and height < 1000:
                contours_new.append(contour) 

        rectangles = [cv2.boundingRect(contour) for contour in contours_new]      
                
        filtered_rectangles = [(x, y, w, h) for x, y, w, h in rectangles if w >= 200 and h >= 100]
        
        imgcopy = image.copy()
        for rect in filtered_rectangles:
            x, y, w, h = rect
            cv2.rectangle(imgcopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        print(len(filtered_rectangles))
        cv2.drawContours(imgcopy, contours_new, -1, (0, 0, 255), 1)
            
        # svaki pravougaonik je opisan sa x,y,w,h
        for rectangle in filtered_rectangles:
            
            x, y, w, h = rectangle
            if h > 100: 
                print(rectangle)
                # isecanje auta i skaliranje na 120x60
                car = image[y:y+h, x:x+w]
                car = cv2.resize(car, (120, 60), interpolation=cv2.INTER_NEAREST)
                #plt.imshow(car)
                #plt.show()
                
                # hog
                car_features = hog_descriptor.compute(car).reshape(1, -1)
                predicted_car = classifier.predict(car_features)[0]
                print(predicted_car)
                
                # gornja desna ivica auta
                right_x = x + w
                center_y = 2080 - y 
                print(center_y)
                print(line_down_y)
                print(line_up_y)
                if (line_down_y <= center_y <= line_up_y)  and (detect_cross(right_x, xx)):
                    print("PRECI CE PREKO!!!!!!")
                    sum_of_cars += predicted_car
            
        print("trenutna suma je:", sum_of_cars)
        print("*********************************")
        plt.imshow(imgcopy)
        plt.show()
        

    cap.release()
    return sum_of_cars


suma = process_video("data2/videos/segment_4.mp4", hog, clf_svm)
print(suma)
print("krajjjjjjj")


csv_file = sys.argv[1] + "counts.csv"

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

for video in df["Naziv_videa"]:
    suma = process_video(sys.argv[1] + 'videos/' +  video + '.mp4', hog, clf_svm)
    row = df[df["Naziv_videa"] == video]
    car_count = row["Broj_kolizija"].values[0]

    measured_v.append(suma)
    true_v.append(car_count)
    print(video + '.mp4' + "-" + str(car_count) + "-" + str(suma))

mae = mean_absolute_error(measured_v,true_v)
print(str(mae))