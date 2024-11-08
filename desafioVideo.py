import cv2
import mediapipe as mp
import os
from deepface import DeepFace
from tqdm import tqdm
from mtcnn import MTCNN


def detect_pose_and_count_emotions_and_arms(video_path, output_path):
    # Inicializar o MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    
    # Carregar o classificador de rosto do OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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
    print(total_frames)

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    ########################## outra lib para detectar rostos ############################
    # Inicializar o detector MTCNN
    detector = MTCNN()
        
    # Variáveis para contar movimentos dos braços
    #arm_up = False
    #arm_movements_count = 0

    # Função para verificar se o braço está levantado
    #def is_arm_up(landmarks):
    #    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    #    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    #    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    #    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    #    left_arm_up = left_elbow.y < left_eye.y
    #    right_arm_up = right_elbow.y < right_eye.y

    #    return left_arm_up or right_arm_up

    # Loop para processar cada frame do vídeo com barra de progresso
    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        # Ler um frame do vídeo
        ret, frame = cap.read()

        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        # Converter o frame para RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Detectar rostos usando outra lib MTCNN
        faces = detector.detect_faces(rgb_frame)
        
        #print(f"{len(faces)} rostos detectados")

        # Para cada rosto detectado, realizar o reconhecimento
        for face in faces:
            x, y, w, h = face['box']
       
            # Extrair o rosto da imagem
            face_img = frame[y:y+h, x:x+w]
        
            try:
                # Analisar o frame para detectar faces e expressões
                result = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)

                # Iterar sobre cada face detectada
                min_width, min_height = 90, 90  # por exemplo
                for face in result:
                    # Obter a caixa delimitadora da face
                    x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                    
                    # Obter a emoção dominante
                    dominant_emotion = face['dominant_emotion']
                    
                    if w > min_width and h > min_height:
                    # Só considera como rosto se passar pelo filtro de tamanho
                    
                        # Desenhar um retângulo ao redor de cada rosto
                        cor = (0, 255, 0)
                        if(dominant_emotion.lower() == 'happy'):
                            cor = (0, 0, 255) 
                        if(dominant_emotion.lower() == 'fear'):
                            cor = (255, 0, 0) 
                        if(dominant_emotion.lower() == 'surprise'):
                            cor = (255,20,147)
                        if(dominant_emotion.lower() == 'sad'):
                            cor = (255,215,0) 
                        
                        cv2.rectangle(face_img, (x, y), (x + w, y + h), cor, 2)
                        
                        # Desenhar um retângulo ao redor da face
                        #cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        #cv2.rectangle(face_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Escrever a emoção dominante acima da face
                        cv2.putText(face_img, dominant_emotion, (x, y+20), cv2.FONT_ITALIC, 0.9, cor, 2)
                    
                        #print("Resultado para o rosto detectado:")
                        #print(result)
            except Exception as e:
                print(f"Erro no reconhecimento do rosto: {e}")

        # Desenhar um retângulo ao redor de cada rosto
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        ############################### para pegar emotions ###################################
        
        
        
        
        ###########################################################################################

        # Processar o frame para detectar a pose
        #results = pose.process(rgb_frame)

        # Desenhar as anotações da pose no frame
        #if results.pose_landmarks:
        #    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Verificar se o braço está levantado
        #    if is_arm_up(results.pose_landmarks.landmark):
        #        if not arm_up:
        #            arm_up = True
        #            arm_movements_count += 1
        #    else:
        #        arm_up = False

            # Exibir a contagem de movimentos dos braços no frame
        #    cv2.putText(frame, f'Movimentos dos bracos: {arm_movements_count}', (10, 30),
        #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
detect_pose_and_count_emotions_and_arms(input_video_path, output_video_path)