#!/usr/bin/env python
# coding: utf-8

# Importación de librerías

# In[76]:


import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Lectura del modelo

# In[73]:


model = load_model('C:/Users/user/Desktop/Master big data/Trimestre 3/Tecnicas de Desarrollo Avanzado de Aplicaciones Big Data/Actividades/Actividad 2/model_keras_catdog.h5')


# Dimensiones de a las que se va a ajustar la imagen

# In[74]:


IMG_WIDTH = 200
IMG_HEIGHT = 200


# Función que tras el tratamiento de la imagen para poder ser utilizada como entrada del modelo devuelve la etiqueta predicha y la probabilidad

# In[ ]:


def prediccion(imagen):
    imagen = imagen.resize((IMG_WIDTH, IMG_HEIGHT))  
    imagen = np.array(imagen) / 255.0 
    imagen = np.expand_dims(imagen, axis=0) 

    pred = model.predict(imagen, verbose=False)
    prob = pred[0][0]
    if(prob > 0.5):
        return ["perro", prob]
    else:
        return["gato", 1-prob]


# Pone un título en la aplicación de streamlit

# In[ ]:


st.title("Modelo para distinguir entre gatos y perros")


# Lee la imagen subida, la muestra y da la predicción del modelo para esa imagen

# In[68]:


imagen_subida = st.file_uploader('Sube una imagen', type='jpg')

if imagen_subida is not None:
    imagen = Image.open(imagen_subida)
    st.image(imagen, caption='Imagen subida', use_column_width=True)
    
    pred = prediccion(imagen)
    st.write("El modelo predice que es un", pred[0], " con una probabilidad de ", str(np.round(pred[1], 2)))

