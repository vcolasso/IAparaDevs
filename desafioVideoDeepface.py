import cv2
import numpy as np
from PIL import Image
import os
from facenet_pytorch import MTCNN

# Inicializar o detector MTCNN
mtcnn = MTCNN(keep_all=True, thresholds=[0.7, 0.8, 0.9])

# Função para detectar e analisar rostos com DeepFace
def detect_and_analyze_faces(video_path, output_path):

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #print("Total de frames:")
    print(f'\ntotal de frames: {total_frames}')

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    #for _ in tqdm(range(total_frames), desc="Processando vídeo"):
    while cap.isOpened():
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Detectar rostos na imagem
        boxes, probs = mtcnn.detect(frame)

        # Configurações de filtro
        min_confidence = 0.75  # Limite mínimo de confiança
        #min_face_size = 60     # Tamanho mínimo de largura/altura para um rosto

        # Converter a imagem para formato OpenCV (BGR)
        img_cv2 = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # Verificar se algum rosto foi detectado
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                # Extrair as coordenadas do retângulo
                x1, y1, x2, y2 = map(int, box)  # Converte as coordenadas para inteiros
                
                # Filtrar rostos com base na confiança e no tamanho da caixa delimitadora
                if prob >= min_confidence: # and (x2 - x1) >= min_face_size and (y2 - y1) >= min_face_size:
                    # Desenhar o retângulo ao redor do rosto
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Adicionar uma etiqueta de confiança
                    cv2.putText(frame, f"{prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    print(f'rosto não detectado probabilidade {prob}, tamanho x {(x2 - x1)}, tamanho y {(y2 - y1)}')
        

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

        # Exibir o frame processado
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    
# Caminho para o vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'desafio.mp4')  # Nome do vídeo de entrada
output_video_path = os.path.join(script_dir, 'output_desafio.mp4')  # Nome do vídeo de saída
# Processar o vídeo
detect_and_analyze_faces(input_video_path, output_video_path)