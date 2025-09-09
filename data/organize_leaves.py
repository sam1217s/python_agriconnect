import os
import shutil
from glob import glob

# Ruta fuente: tus clases directamente en data/leaves/ (Apple___Apple_scab, Tomato___healthy, etc.)
source_dir = 'data/leaves/'

# Carpetas binarias de salida (ya creadas)
healthy_dir = 'data/leaves/healthy/'
sick_dir = 'data/leaves/sick/'
os.makedirs(healthy_dir, exist_ok=True)
os.makedirs(sick_dir, exist_ok=True)

# Copia healthy: clases que terminan en '___healthy' (14 clases: Tomato___healthy, Potato___healthy, etc.)
healthy_count = 0
for class_folder in os.listdir(source_dir):
    if class_folder in ['healthy', 'sick']: continue  # Salta para evitar copia de sí misma
    if class_folder.endswith('___healthy'):
        class_path = os.path.join(source_dir, class_folder)
        img_files = glob(os.path.join(class_path, '*.jpg')) + glob(os.path.join(class_path, '*.png'))
        for img_file in img_files[:500]:  # 500 por clase (~7k total)
            shutil.copy(img_file, healthy_dir)
            healthy_count += 1

# Copia sick: clases de enfermedades (24 clases: Tomato___Late_blight, Potato___Early_blight, Grape___Black_rot, etc.)
sick_count = 0
for class_folder in os.listdir(source_dir):
    if class_folder in ['healthy', 'sick']: continue  # Salta
    if not class_folder.endswith('___healthy'):
        class_path = os.path.join(source_dir, class_folder)
        img_files = glob(os.path.join(class_path, '*.jpg')) + glob(os.path.join(class_path, '*.png'))
        for img_file in img_files[:500]:  # Balancea ~12k
            shutil.copy(img_file, sick_dir)
            sick_count += 1

print(f"Healthy: {healthy_count} imágenes (sanas de 14 especies: Tomato___healthy, Potato___healthy, etc.)")
print(f"Sick: {sick_count} imágenes (enfermas de 24 clases: Tomato___Late_blight, Grape___Black_rot, etc.)")
print("¡Listo! Carpetas healthy/sick llenas con datos reales PlantVillage para agro.py.")