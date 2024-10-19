import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
import pyodbc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, ConfusionMatrixDisplay
import pandas as pd
import seaborn as sns

# Tkinter penceresini oluştur
window = tk.Tk()
window.title("Face recognition system")

# Kullanıcı bilgileri giriş alanları ve etiketleri
l1 = tk.Label(window, text="Name", font=("Algerian", 20))
l1.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

actual_ids = []
predicted_ids = []

# Eğitici buton fonksiyonu
def train_classifier():
    # Veri dizinini al
    data_dir = "C:\\Users\\Metehan\\OneDrive\\Masaüstü\\FaceRecegnotion\\data"
    # Dizindeki her bir dosya için
    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]
    faces = []
    ids = []
    # Her görüntüyü işle
    for image in path:
        img = Image.open(image).convert('L')  # Resmi siyah beyaza dönüştür
        imageNp = np.array(img, 'uint8')  # Numpy dizisine dönüştür
        id = int(os.path.split(image)[1].split(".")[1])  # Kimliği al
        faces.append(imageNp)  # Yüzü ekle
        ids.append(id)  # Kimliği ekle
    ids = np.array(ids)
    # Sınıflandırıcıyı eğit ve kaydet
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")
    messagebox.showinfo('Result', 'Training dataset completed!!!')

# Eğitici buton
b1 = tk.Button(window, text="Training", font=("Algerian", 20), bg='orange', fg='red', command=train_classifier)
b1.grid(column=0, row=4)

# Yüz tespiti fonksiyonu
def detect_face():
    # Çerçeve çizme fonksiyonu
    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gri tona dönüştür
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)  # Yüzleri tespit et

        coords = []

        for(x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)  # Yüz etrafına dikdörtgen çiz
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])  # Yüzü tanı
            confidence = int(100*(1-pred/300))  # Güvenilirlik hesapla

            # MSSQL bağlantısı
            conn = pyodbc.connect('DRIVER={SQL Server};SERVER=Metehan-AYDIN;DATABASE=Authorized_user;UID=sa;PWD=Gs/*1905')
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM my_table WHERE id=?", (id,))
            row = cursor.fetchone()
            s = row[0] if row else "UNKNOWN"
            conn.close()

            # Güvenilirlik oranına göre kişiyi yazdır
            if confidence > 84:
                cv2.putText(img, s+" (%"+str(confidence)+")", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
                actual_ids.append(id)  # Gerçek ID
                predicted_ids.append(id)  # Tahmin edilen ID
            else:
                cv2.putText(img, "UNKNOWN", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                actual_ids.append(id)
                predicted_ids.append(-1)  # Tahmin edilemeyenler için -1 kullan

            coords = [x, y, w, h]
        return coords

    # Tanıma fonksiyonu
    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    # Yüz sınıflandırıcısı ve tanıyıcısını başlat
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, img = video_capture.read()
        img = recognize(img, clf, faceCascade)
        cv2.imshow("face detection", img)

        if cv2.waitKey(1) == 13:
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Yüz tespiti butonu
b2 = tk.Button(window, text="Detect the face", font=("Algerian", 20), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=4)

# Veri seti oluşturma fonksiyonu
def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        # MSSQL bağlantısı
        conn = pyodbc.connect('DRIVER={SQL Server};SERVER=Metehan-AYDIN;DATABASE=Authorized_user;UID=sa;PWD=Gs/*1905')
        cursor = conn.cursor()
        # İsim kontrolü ve varsa güncelleme
        cursor.execute("SELECT * FROM my_table WHERE Name=?", (t1.get(),))
        existing_user = cursor.fetchone()
        if existing_user:
            id = existing_user[0]
            # Varolan kullanıcıyı güncelle
            sql = "UPDATE my_table SET Age=?, Address=? WHERE id=?"
            val = (t2.get(), t3.get(), id)
            cursor.execute(sql, val)
        else:
            # Yeni bir kullanıcı ekle
            cursor.execute("SELECT COUNT(*) FROM my_table")
            id = cursor.fetchone()[0] + 1
            sql = "INSERT INTO my_table(id, Name, Age, Address) VALUES (?, ?, ?, ?)"
            val = (id, t1.get(), t2.get(), t3.get())
            cursor.execute(sql, val)
        conn.commit()
        # Yüz sınıflandırıcısı
        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        # Yüz kesme fonksiyonu
        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                return None
            for(x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
            return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 0

        while True:
            ret, frame = cap.read()
            if face_cropped(frame) is not None:
                img_id += 1
                face = cv2.resize(face_cropped(frame), (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                
                # Fotoğraf dosya adını kullanıcı ismi ve sıra numarası olarak oluştur
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                
                # Eğer dosya zaten varsa üzerine yazmak yerine numarayı artır
                while os.path.exists(file_name_path):
                    img_id += 1
                    file_name_path = f"data/user.{id}.{img_id}.jpg"
                    
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)
                if cv2.waitKey(1) == 13 or int(img_id) == 400:
                    break
        cap.release()
        cv2.destroyAllWindows()
        
        messagebox.showinfo('Result', 'Generating dataset completed!!!')

# Veri seti oluşturma butonu
b3 = tk.Button(window, text="Generate dataset", font=("Algerian", 20), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=2, row=4)

# Grafik çizim fonksiyonu
def draw_graphs():
    # Sınıflandırma raporu
    report = classification_report(actual_ids, predicted_ids, output_dict=True)
    df = pd.DataFrame(report).transpose()

    # Grafik çizimi
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.iloc[:-1, :].T, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title("Classification Report")
    plt.show()

    # Karmaşıklık matrisi
    cm = confusion_matrix(actual_ids, predicted_ids)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# Grafik çizim butonu
b4 = tk.Button(window, text="Draw Graphs", font=("Algerian", 20), bg='blue', fg='white', command=draw_graphs)
b4.grid(column=3, row=4)

# Pencere boyutunu ayarla ve döngüyü başlat
window.geometry("1000x400")
window.mainloop()
