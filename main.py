import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configuração da página do Streamlit
st.set_page_config(page_title="Detector de Objetos - YOLOv8", layout="centered")

st.title("📷 Identificação de Imagens com YOLOv8")
st.write("Faça o upload de uma imagem para identificar os objetos presentes nela.")

# Inicialização do modelo (faz o download automático do modelo nano na primeira execução)
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Botão de Upload da Aplicação
uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Converter o arquivo enviado para uma imagem PIL
    image = Image.open(uploaded_file)
    
    # Criar colunas para o ANTES e DEPOIS
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        st.image(image, use_container_width=True)
    
    # Executar a inferência do modelo YOLO
    # convertemos para array numpy pois o YOLO lida bem com formato OpenCV/Numpy
    img_array = np.array(image)
    results = model(img_array)
    
    # O YOLO já possui um método nativo para desenhar as predições e retornar a imagem
    res_plotted = results[0].plot()
    
    with col2:
        st.subheader("Objetos Identificados")
        st.image(res_plotted, channels="BGR", use_container_width=True)
        
    # Exibir resumo dos objetos em texto abaixo
    st.markdown("---")
    st.subheader("📝 Relatório de Detecção")
    
    boxes = results[0].boxes
    if len(boxes) == 0:
        st.info("Nenhum objeto conhecido foi detectado na imagem.")
    else:
        for box in boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            conf = float(box.conf[0]) * 100
            st.write(f"• **{label.upper()}** com {conf:.2f}% de confiança.")
