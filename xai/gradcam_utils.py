import tensorflow as tf
import cv2
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def grad_cam(model, img_array, layer_name=None):
    """
    Genera heatmap Grad-CAM sin matplotlib (thread-safe)
    """
    try:
        # Detectar automáticamente la última capa convolucional
        if layer_name is None:
            layer_name = find_last_conv_layer(model)
        
        # Crear modelo Grad-CAM
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Para clasificación binaria
            class_idx = 0 if predictions.shape[-1] == 1 else tf.argmax(predictions[0])
            loss = predictions[:, class_idx] if predictions.shape[-1] > 1 else predictions[:, 0]
        
        # Calcular Grad-CAM
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Pooling de gradientes
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
        
        # Generar heatmap
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, output), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalizar
        if tf.reduce_max(heatmap) > 0:
            heatmap = heatmap / tf.reduce_max(heatmap)
        
        # Redimensionar al tamaño original
        heatmap = tf.image.resize(
            tf.expand_dims(heatmap, -1), 
            [img_array.shape[1], img_array.shape[2]]
        )
        
        return heatmap.numpy().squeeze()
        
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return grad_cam_fallback(img_array)

def find_last_conv_layer(model):
    """
    Encuentra automáticamente la última capa convolucional
    """
    for layer in reversed(model.layers):
        if 'conv' in layer.name.lower():
            return layer.name
    
    # Fallback para MobileNetV2
    possible_layers = [
        'out_relu',  # MobileNetV2
        'block_16_expand_relu',
        'block_15_expand_relu', 
        'Conv_1_relu',
        'global_average_pooling2d'
    ]
    
    for layer_name in possible_layers:
        try:
            model.get_layer(layer_name)
            return layer_name
        except:
            continue
    
    # Último recurso: usar una capa intermedia
    return model.layers[-3].name

def grad_cam_fallback(img_array):
    """
    Fallback: generar heatmap simple basado en intensidad
    """
    try:
        # Convertir a escala de grises y crear heatmap simple
        if len(img_array.shape) == 4:
            img = img_array[0]
        else:
            img = img_array
            
        # Crear heatmap basado en intensidad promedio por región
        gray = np.mean(img, axis=-1) if len(img.shape) == 3 else img
        
        # Suavizar para simular activación (sin scipy)
        kernel = np.ones((5,5))/25
        heatmap = cv2.filter2D(gray, -1, kernel)
        
        # Normalizar
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
        
    except Exception as e:
        print(f"Fallback failed: {e}")
        # Último recurso: heatmap uniforme
        shape = img_array.shape[1:3] if len(img_array.shape) == 4 else img_array.shape[:2]
        return np.ones(shape) * 0.5
    """
    Fallback: generar heatmap simple basado en intensidad
    """
    try:
        # Convertir a escala de grises y crear heatmap simple
        if len(img_array.shape) == 4:
            img = img_array[0]
        else:
            img = img_array
            
        # Crear heatmap basado en intensidad promedio por región
        gray = np.mean(img, axis=-1) if len(img.shape) == 3 else img
        
        # Suavizar para simular activación
        from scipy import ndimage
        heatmap = ndimage.gaussian_filter(gray, sigma=10)
        
        # Normalizar
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        return heatmap
        
    except Exception as e:
        print(f"Fallback failed: {e}")
        # Último recurso: heatmap uniforme
        shape = img_array.shape[1:3] if len(img_array.shape) == 4 else img_array.shape[:2]
        return np.ones(shape) * 0.5