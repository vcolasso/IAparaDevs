import cv2
import numpy as np
import mediapipe as mp
import os

from deepface import DeepFace

# Inicializa os módulos de detecção de rosto e desenho do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Inicializar o MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

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
    
    #print("Total de frames:")
    print(f'\ntotal de frames: {total_frames}')

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    #for _ in tqdm(range(total_frames), desc="Processando vídeo"):
    # Inicia a detecção de rostos com o modelo de curta distância (model_selection=1) e baixa confiança (para detectar mais rostos)
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.1) as face_detection:

        while cap.isOpened():

            success, image = cap.read() # Lê um frame do vídeo
            if not success: # Sai do loop se a leitura falhar (fim do vídeo)
                break

            # Impede que o MediaPipe modifique a imagem original diretamente (melhora o desempenho)
            image.flags.writeable = False

            # Converte a imagem de BGR (OpenCV) para RGB (MediaPipe)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Processa a imagem para detectar rostos
            results = face_detection.process(image)

            # Permite a modificação da imagem novamente
            image.flags.writeable = True

            # Converte a imagem de volta para BGR para exibir com OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Variáveis para contar movimentos dos braços
            arm_up = False
            arm_movements_count = 0

            # Função para verificar se o braço está levantado
            def is_arm_up(landmarks):

                left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
                right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
                left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
                right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
                left_arm_up = left_elbow.y < left_eye.y
                right_arm_up = right_elbow.y < right_eye.y
                return left_arm_up or right_arm_up

            # Desenha os retângulos de detecção na imagem
            if results.detections: # verifica se encontrou algum rosto

                for detection in results.detections: # Itera sobre as detecções
                    
                    #mp_drawing.draw_detection(image, detection) # Desenha a detecção na imagem

                    # Obtém as coordenadas da detecção
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = image.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                    # Recorta o rosto detectado
                    face_image = image[y:y+h, x:x+w]

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
                            cv2.rectangle(image, (x, y), (x + w, y + h), cor, 2)

                            # Exibe a emoção detectada no rosto
                            cv2.putText(image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, cor, 2, cv2.LINE_AA)

                    except Exception as e:
                        print(f"Erro ao analisar emoção: {e}")


            # Processar o frame para detectar a pose
            results_arms_detection = pose.process(image)

            # Desenhar as anotações da pose no frame
            if results_arms_detection.pose_landmarks:

                mp_drawing.draw_landmarks(image, results_arms_detection.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Verificar se o braço está levantado
                if is_arm_up(results_arms_detection.pose_landmarks.landmark):
                    if not arm_up:
                        arm_up = True
                        arm_movements_count += 1
                else:
                    arm_up = False

                # Exibir a contagem de movimentos dos braços no frame
                cv2.putText(image, f'Movimentos dos bracos: {arm_movements_count}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Escrever o frame processado no vídeo de saída
            out.write(image)

            # Exibir o frame processado
            cv2.imshow('Video', image)
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