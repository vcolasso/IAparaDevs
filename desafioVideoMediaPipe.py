import cv2
import numpy as np
import mediapipe as mp
import os
import torch
from deepface import DeepFace

# Inicializa os módulos de detecção de rosto e desenho do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Carregar modelo YOLOv5 pré-treinado
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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

# Função para detectar e analisar rostos com DeepFace
def detect_and_analyze_faces(video_path, output_path):

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
    
    #print("Total de frames:")
    print(f'\ntotal de frames: {total_frames}')

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    #for _ in tqdm(range(total_frames), desc="Processando vídeo"):
    # Inicia a detecção de rostos com o modelo de curta distância (model_selection=1) e baixa confiança (para detectar mais rostos)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection:

        while cap.isOpened():

            success, frame = cap.read() # Lê um frame do vídeo
            if not success: # Sai do loop se a leitura falhar (fim do vídeo)
                break

            # Impede que o MediaPipe modifique a framem original diretamente (melhora o desempenho)
            frame.flags.writeable = False

            # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Processa a imagem para detectar rostos
            results = face_detection.process(frame)

            # Permite a modificação da imagem novamente
            frame.flags.writeable = True

            # Converte a imagem de volta para BGR para exibir com OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Variáveis para contar movimentos dos braços
            arm_up = False
            arm_movements_count = 0

            # Desenha os retângulos de detecção de rosto na imagem
            if results.detections: # verifica se encontrou algum rosto

                for detection in results.detections: # Itera sobre as detecções
                    
                    #mp_drawing.draw_detection(image, detection) # Desenha a detecção na imagem

                    # Obtém as coordenadas da detecção
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Recorta o rosto detectado
                    face_image = frame[y:y+h, x:x+w]

                    try:
                        # Usa o DeepFace para detectar emoções
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

                            # Desenhar um retângulo ao redor de cada rosto
                            cor = (0, 255, 0)
                            if(emotion.lower() == 'happy'):
                                cor = (150, 25, 255) 
                            if(emotion.lower() == 'fear'):
                                cor = (255, 150, 25) 
                            if(emotion.lower() == 'surprise'):
                                cor = (147,0,255)
                            if(emotion.lower() == 'sad'):
                                cor = (215,255,0) 

                            # Desenha o bounding box
                            cv2.rectangle(frame, (x, y), (x + w, y + h), cor, 2)

                            # Exibe a emoção detectada no rosto
                            cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Erro ao analisar emoção: {e}")


            # Detectar múltiplas pessoas no quadro
            results_person_detection = model(frame)  # YOLOv5 detecta objetos no quadro
            detections = results_person_detection.pandas().xyxy[0]

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

                        # Desenhar a pose
                        #mp_drawing.draw_landmarks(
                        #    image,
                        #    results_pose.pose_landmarks,
                        #    mp_pose.POSE_CONNECTIONS,
                        #    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        #)

                        # Verificar se o braço está levantado
                        if is_arm_up(results_pose.pose_landmarks.landmark):
                            if not person_movements[person_id]["arm_up"]:
                                person_movements[person_id]["arm_up"] = True
                                person_movements[person_id]["movements"] += 1
                        else:
                            person_movements[person_id]["arm_up"] = False

                        # Exibir contagem de movimentos no quadro
                        movement_text = f"Movimentos: {person_movements[person_id]['movements']}"
                        cv2.putText(frame, movement_text, (xmin, ymin - 10), cv2.FONT_ITALIC, 0.6, (255, 150, 120), 2)
            
            
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