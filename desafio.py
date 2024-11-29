import cv2
import numpy as np
import mediapipe as mp
import os
import torch
from deepface import DeepFace
import math
import random
 

# Inicializar o MediaPipe Pose para detectar os movimentos
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Inicializa o MediaPipe para fazer reconhecimentos faciais
mp_face_detection = mp.solutions.face_detection # Usar essa lib para detectar múltiplos rostos, é mais eficaz
mp_face_mesh = mp.solutions.face_mesh # Usar essa lib para detectar as anomalias faciais
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=6, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Carregar modelo YOLOv5 pré-treinado para detectar múltiplas pessoas
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Função para calcular a distância euclidiana
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Função para detectar caretas
def is_grimace(landmarks, image_width, image_height):
    # Convertendo landmarks para coordenadas na imagem
    points = [(int(lm.x * image_width), int(lm.y * image_height)) for lm in landmarks]

    # Definindo landmarks importantes
    left_eye = points[159]  # Ponto da pálpebra superior do olho esquerdo
    right_eye = points[386]  # Pálpebra superior do olho direito
    mouth_left = points[61]  # Canto esquerdo da boca
    mouth_right = points[291]  # Canto direito da boca
    mouth_top = points[13]  # Centro superior da boca
    mouth_bottom = points[14]  # Centro inferior da boca

    # Medir proporções da boca
    mouth_width = euclidean_distance(mouth_left, mouth_right)
    mouth_height = euclidean_distance(mouth_top, mouth_bottom)

    # Detecção de caretas (boca exageradamente aberta)
    if mouth_height / mouth_width > 0.6:  # Limite ajustável
        return True

    # Detecção de sobrancelhas levantadas (opcional)
    left_eyebrow = points[70]  # Acima do olho esquerdo
    right_eyebrow = points[300]  # Acima do olho direito
    left_eyebrow_distance = euclidean_distance(left_eye, left_eyebrow)
    right_eyebrow_distance = euclidean_distance(right_eye, right_eyebrow)

    if left_eyebrow_distance > 0.1 * image_height or right_eyebrow_distance > 0.1 * image_height:  # Limite ajustável
        return True

    # Caretas assimétricas (canto da boca muito deslocado)
    mouth_asymmetry = abs(mouth_left[1] - mouth_right[1])
    if mouth_asymmetry > 0.05 * image_height:  # Limite ajustável
        return True

    return False

def gravar_arquivo_resumo (path, resumo_texto):
    with open(path,'w') as f:
        for linha in resumo_texto:
            f.write('{}\n'.format(linha))
        f.close()


# Função para verificar se o braço está levantado
def is_arm_up(landmarks):
    
    #Verifica se o braço esquerdo ou direito está levantado com base na posição dos punhos acima dos ombros.
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    # Verifica se o punho está acima do ombro correspondente
    left_arm_up = left_wrist.y < left_shoulder.y
    right_arm_up = right_wrist.y < right_shoulder.y

    # Retorna True se qualquer braço estiver levantado
    return left_arm_up or right_arm_up

# Dicionário para traduzir emoções para PT-BR
emotion_translation = {
    "happy": "Feliz",
    "sad": "Triste",
    "neutral": "Neutro",
    "angry": "Raiva",
    "surprise": "Surpresa",
    "fear": "Medo",
    "disgust": "Nojo"
}

# Função para detectar e analisar rostos com DeepFace
def detect_and_analyze_faces(video_path, output_path):
    
    # inicia contadores
    count_anomalo_facial = 0
    # Variável para contar movimentos 
    total_movimentos = 0

    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)

    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter as dimensões e FPS do vídeo original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Dicionário para armazenar o estado de cada pessoa
    person_movements = {}

    # Dicionário para armazenar as emoções detectadas
    emotions = {}
        
    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    emocao_atual = ''
    
    # Inicia a detecção de rostos com o modelo de curta distância (model_selection=1) e baixa confiança (para detectar mais rostos)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection:

        while cap.isOpened():

            success, frame = cap.read() # Lê um frame do vídeo
            if not success: # Sai do loop se a leitura falhar (fim do vídeo)
                break

            # Converte a imagem para RGB (MediaPipe)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Usando o face_detection para detectar os rostos, é mais eficaz que o face_mesh
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:

                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Recortar o rosto para análise de emoção
                    face_image = frame[y:y+h, x:x+w]

                    try:
                        # Detecção de emoções com DeepFace
                        analysis = DeepFace.analyze(face_image, actions=['emotion'], enforce_detection=False)
                        
                        # Ajuste com base na estrutura do retorno
                        emotion = ""
                        if isinstance(analysis, list):  # Algumas versões retornam uma lista
                            if len(analysis) > 0:
                                emotion = analysis[0]['dominant_emotion']
                        elif isinstance(analysis, dict):  # Outras retornam um dicionário diretamente
                            emotion = analysis['dominant_emotion']
                        else:
                            emotion = "Desconhecido"

                        if len(emotion) > 0: 

                            # Traduz a emoção para PT-BR
                            emotion_pt = emotion_translation.get(emotion.lower(), "Desconhecido").lower()

                            if emotion_pt not in emotions:
                                r_aleatorio = random.randint(0, 255)
                                g_aleatorio = random.randint(0, 255)
                                b_aleatorio = random.randint(0, 255)
                                cor = (r_aleatorio, g_aleatorio, b_aleatorio)
                                emotions[emotion_pt] = {"quantidade": 1, "cor": cor}
                            else:
                                if emocao_atual != emotions[emotion_pt] :                                    
                                    emotions[emotion_pt]["quantidade"] += 1
                                    emocao_atual = emotions[emotion_pt]   

                            cv2.rectangle(frame, (x, y), (x + w, y + h), emotions[emotion_pt]["cor"], 2)
                            cv2.putText(frame, emotion_pt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, emotions[emotion_pt]["cor"], 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Erro ao analisar emoção: {e}")


            # Detectar rostos para verificar anomalias
            results_anomalia = face_mesh.process(rgb_frame)

            if results_anomalia.multi_face_landmarks:

                for face_landmarks in results_anomalia.multi_face_landmarks:

                    # Vê se tem anomalia facial. Nesta lógica, um rosto deitado é considerado uma anomalia
                    if is_grimace(face_landmarks.landmark, frame_width, frame_height):
                        cv2.putText(frame, "Anomalia facial!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        count_anomalo_facial += 1


            # Detectar múltiplas pessoas no quadro
            results_person_detection = model(rgb_frame)  # YOLOv5 detecta objetos no quadro
            detections = results_person_detection.pandas().xyxy[0]

            # Iterando sobre as pessoas detectadas
            for index, det in detections.iterrows():

                if det['name'] == 'person':  # Foco em pessoas
                    xmin, ymin, xmax, ymax = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    person_roi = frame[ymin:ymax, xmin:xmax]  # Recorte da região da pessoa

                    # Criar ID único com base nas coordenadas
                    person_id = f"{xmin}_{ymin}_{xmax}_{ymax}"
                    if person_id not in person_movements:
                        person_movements[person_id] = {"arm_up": False, "movements": 0}

                    # Processar pose da pessoa
                    person_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    results_pose = pose.process(person_rgb)

                    if results_pose.pose_landmarks:
                        # Redimensionar coordenadas para o quadro original
                        h, w, _ = person_roi.shape
                        for landmark in results_pose.pose_landmarks.landmark:
                            cx = int(xmin + landmark.x * w)
                            cy = int(ymin + landmark.y * h)
                            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

                        # Verificar se o braço está levantado
                        if is_arm_up(results_pose.pose_landmarks.landmark):
                            if not person_movements[person_id]["arm_up"]:
                                person_movements[person_id]["arm_up"] = True
                                person_movements[person_id]["movements"] += 1
                                total_movimentos +=1
                                # Exibir contagem de movimentos no quadro
                                movement_text = f"Movimento detectado "
                                cv2.putText(frame, movement_text, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                                
                        else:
                            person_movements[person_id]["arm_up"] = False

                        
            # Escrever o frame processado no vídeo de saída
            out.write(frame)

            # Exibir o frame processado
            cv2.imshow('Video', frame)
            # Apertando esc + q interrompe o processamento
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    #gravando resumo
    resumo = []
    resumo.append('Total de frames: ' + str(total_frames))
    resumo.append('Total de movimentos: ' + str(total_movimentos))
    resumo.append('Total de anomalias faciais: ' + str(count_anomalo_facial))

    for index, (key, value) in enumerate(emotions.items()):
        resumo.append(f"Total de {key}:  {value['quantidade']}")

    
    #Exibindo totais 
    for texto in resumo:
        print (texto)
    
    #Gravando arqquivo de resumo
    gravar_arquivo_resumo (os.path.join(script_dir, 'resumo.txt'), resumo)
    
    
# Caminho para o vídeo de entrada e saída
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'desafio.mp4')  # Nome do vídeo de entrada
output_video_path = os.path.join(script_dir, 'output_desafio.mp4')  # Nome do vídeo de saída

# Processar o vídeo
detect_and_analyze_faces(input_video_path, output_video_path)
