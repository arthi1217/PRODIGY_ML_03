import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

IMG_SIZE = 96
MAX_IMAGES = 500

dataset_path = r"cats_and_dogs_filtered\train"
cat_path = os.path.join(dataset_path, "Cat")
dog_path = os.path.join(dataset_path, "Dog")

print("Cat path:", cat_path)
print("Dog path:", dog_path)

data = []
labels = []
for img in os.listdir(cat_path)[:MAX_IMAGES]:
    img_path = os.path.join(cat_path, img)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image.flatten())
    labels.append(0)

for img in os.listdir(dog_path)[:MAX_IMAGES]:
    img_path = os.path.join(dog_path, img)
    image = cv2.imread(img_path)
    if image is None:
        continue
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    data.append(image.flatten())
    labels.append(1)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", gamma="scale")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("SVM Accuracy:", accuracy)
