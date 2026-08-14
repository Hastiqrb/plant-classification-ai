import os
import cv2
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import json  # اضافه شده

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# تنظیمات اولیه
original_dataset = r'D:\plant-classifier-project01\datasets\PlantVillage'
balanced_dataset = r'D:\plant-classifier-project01\datasets_balanced'

TOTAL_IMAGES = 16000
IMG_SIZE = 128
VALID_EXT = ['.jpg', '.jpeg', '.png']

# ساخت Balanced Dataset
if os.path.exists(balanced_dataset):
    print("⚠ Removing previous balanced dataset...")
    shutil.rmtree(balanced_dataset)

os.makedirs(balanced_dataset, exist_ok=True)

classes = sorted([
    d for d in os.listdir(original_dataset)
    if os.path.isdir(os.path.join(original_dataset, d))
])

num_classes = len(classes)
per_class = TOTAL_IMAGES // num_classes

print(f"🔁 Creating balanced dataset ({per_class} images per class)")

for cls in classes:
    src = os.path.join(original_dataset, cls)
    dst = os.path.join(balanced_dataset, cls)
    os.makedirs(dst, exist_ok=True)

    images = [f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in VALID_EXT]
    random.shuffle(images)

    for img in images[:per_class]:
        shutil.copy(os.path.join(src, img), os.path.join(dst, img))

print("✅ Balanced dataset created\n")

# بارگذاری داده‌ها
data, labels = [], []

for label, cls in enumerate(classes):
    folder = os.path.join(balanced_dataset, cls)
    for img_name in os.listdir(folder):
        if os.path.splitext(img_name)[1].lower() in VALID_EXT:
            img = cv2.imread(os.path.join(folder, img_name))
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img / 255.0
                data.append(img)
                labels.append(label)

X = np.array(data)
y = np.array(labels)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}\n")

# مدل CNN
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

cnn_model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

cnn_history = cnn_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

# ========== ذخیره مدل CNN ==========
cnn_model.save('plant_cnn_model.h5')
print("✅ مدل CNN ذخیره شد: plant_cnn_model.h5")
# ===================================

# مدل MLP
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat = X_test.reshape(len(X_test), -1)

mean = np.mean(X_train_flat, axis=0)
std = np.std(X_train_flat, axis=0) + 1e-7

X_train_flat = (X_train_flat - mean) / std
X_test_flat = (X_test_flat - mean) / std

mlp_model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_flat.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

mlp_history = mlp_model.fit(
    X_train_flat, y_train,
    validation_data=(X_test_flat, y_test),
    epochs=60,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# رسم نمودار
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(cnn_history.history['val_accuracy'], label='CNN')
plt.plot(mlp_history.history['val_accuracy'], label='MLP')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(cnn_history.history['val_loss'], label='CNN')
plt.plot(mlp_history.history['val_loss'], label='MLP')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ارزیابی
def evaluate_model(name, y_true, y_pred):
    print(f"\n🔹 {name} Evaluation")
    print(f"Accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    print(f"Recall   : {recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")
    print(f"F1 Score : {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.2f}%")

y_pred_cnn = np.argmax(cnn_model.predict(X_test), axis=1)
y_pred_mlp = np.argmax(mlp_model.predict(X_test_flat), axis=1)

evaluate_model("CNN", y_test, y_pred_cnn)
evaluate_model("MLP", y_test, y_pred_mlp)

# ========== ذخیره کلاس‌ها ==========
with open('classes.json', 'w') as f:
    json.dump(classes, f)
print("✅ کلاس‌ها ذخیره شد: classes.json")
# ===================================

# نمایش نتایج
num_images = 12
rows = 3
cols = 4

plt.figure(figsize=(14, 10))

for i in range(num_images):
    idx = random.randint(0, len(X_test) - 1)
    img = X_test[idx]
    pred_class = classes[np.argmax(cnn_model.predict(img[np.newaxis, ...]))]

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Predicted: {pred_class}", fontsize=10)

plt.suptitle("Plant Disease Detection Results", fontsize=16)
plt.tight_layout()
plt.show()