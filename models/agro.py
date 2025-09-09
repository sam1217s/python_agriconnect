import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.metrics import accuracy_score
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os

def load_leaf_data(num_samples_per_class=500):  # 500 para balance (reduce si lento)
    healthy_dir = 'data/leaves/healthy/'
    sick_dir = 'data/leaves/sick/'
    healthy_paths = glob(os.path.join(healthy_dir, '*.jpg')) + glob(os.path.join(healthy_dir, '*.png'))
    sick_paths = glob(os.path.join(sick_dir, '*.jpg')) + glob(os.path.join(sick_dir, '*.png'))
    print(f"Paths encontrados: Healthy {len(healthy_paths)}, Sick {len(sick_paths)}")

    # Carga y preprocesa (resize 224x224, normaliza)
    healthy_imgs = []
    for p in healthy_paths[:num_samples_per_class]:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            healthy_imgs.append(img)
    
    sick_imgs = []
    for p in sick_paths[:num_samples_per_class]:
        img = cv2.imread(p)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            sick_imgs.append(img)
    
    images = np.array(healthy_imgs + sick_imgs)
    labels = np.array([0] * len(healthy_imgs) + [1] * len(sick_imgs))
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    print(f"Datos: Train {X_train.shape}, Test {X_test.shape}")
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = load_leaf_data()
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False  # Transfer learning ligero (Día 2)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
preds = (model.predict(X_test) > 0.5).astype(int).flatten()
acc = accuracy_score(y_test, preds)
print(f"Accuracy AgroExpert: {acc:.2f} (hojas sanas/enfermas con PlantVillage datos reales)")
model.save('models/agro_model.keras')  # Formato moderno (evita warning HDF5)