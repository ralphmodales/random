import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

MESH_ANNOTATIONS = {
    'rightEyeUpper0': [246, 161, 160, 159, 158, 157, 173],
    'rightEyeLower0': [33, 7, 163, 144, 145, 153, 154, 155, 133],
    'rightEyeUpper1': [247, 30, 29, 27, 28, 56, 190],
    'rightEyeLower1': [130, 25, 110, 24, 23, 22, 26, 112, 243],
    'rightEyeUpper2': [113, 225, 224, 223, 222, 221, 189],
    'rightEyeLower2': [226, 31, 228, 229, 230, 231, 232, 233, 244],
    'rightEyeLower3': [143, 111, 117, 118, 119, 120, 121, 128, 245],
    
    'leftEyeUpper0': [466, 388, 387, 386, 385, 384, 398],
    'leftEyeLower0': [263, 249, 390, 373, 374, 380, 381, 382, 362],
    'leftEyeUpper1': [467, 260, 259, 257, 258, 286, 414],
    'leftEyeLower1': [359, 255, 339, 254, 253, 252, 256, 341, 463],
    'leftEyeUpper2': [342, 445, 444, 443, 442, 441, 413],
    'leftEyeLower2': [446, 261, 448, 449, 450, 451, 452, 453, 464],
    'leftEyeLower3': [372, 340, 346, 347, 348, 349, 350, 357, 465],

    'rightEyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    'leftEyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
}

def draw_eye_outline(canvas, face_landmarks):
    right_eye_points = (MESH_ANNOTATIONS['rightEyeUpper0'] + 
                        list(reversed(MESH_ANNOTATIONS['rightEyeLower0'])))
    
    left_eye_points = (MESH_ANNOTATIONS['leftEyeUpper0'] + 
                       list(reversed(MESH_ANNOTATIONS['leftEyeLower0'])))
    
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_points = convert_landmarks_to_points(right_eye_points)
    left_points = convert_landmarks_to_points(left_eye_points)
    
    right_points = np.array(right_points, np.int32).reshape((-1, 1, 2))
    left_points = np.array(left_points, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(canvas, [right_points], True, (255, 0, 0), 2)  
    cv2.polylines(canvas, [left_points], True, (255, 0, 0), 2)   

def draw_iris_outline(canvas, face_landmarks): 
    right_iris = [474, 475, 476, 477]
    left_iris = [469, 470, 471, 472]
    
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_iris_points = convert_landmarks_to_points(right_iris)
    left_iris_points = convert_landmarks_to_points(left_iris)
    
    right_iris_points = np.array(right_iris_points, np.int32).reshape((-1, 1, 2))
    left_iris_points = np.array(left_iris_points, np.int32).reshape((-1, 1, 2))
    
    gojo_blue = (230, 180, 50)  
     
    cv2.polylines(canvas, [right_iris_points], True, gojo_blue, 2)  
    cv2.polylines(canvas, [left_iris_points], True, gojo_blue, 2)

def draw_eyebrow_outline(canvas, face_landmarks):
    def convert_landmarks_to_points(landmark_indices):
        points = []
        for idx in landmark_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * canvas.shape[1])
            y = int(landmark.y * canvas.shape[0])
            points.append((x, y))
        return points
    
    right_eyebrow_points = convert_landmarks_to_points(MESH_ANNOTATIONS['rightEyebrow'])
    left_eyebrow_points = convert_landmarks_to_points(MESH_ANNOTATIONS['leftEyebrow'])
    
    right_eyebrow_points = np.array(right_eyebrow_points, np.int32).reshape((-1, 1, 2))
    left_eyebrow_points = np.array(left_eyebrow_points, np.int32).reshape((-1, 1, 2))
    
    cv2.polylines(canvas, [right_eyebrow_points], True, (0, 255, 0), 2)  
    cv2.polylines(canvas, [left_eyebrow_points], True, (0, 255, 0), 2)   

def draw_lip_outline(canvas, face_landmarks):
    upper_lip_landmarks = [61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
    lower_lip_landmarks = [78,95,88,178,87,14,317,402,318,324,308,291,375,321,405,314,17,84,181,91,146,61]
    
    lip_landmarks = upper_lip_landmarks + lower_lip_landmarks
    
    points = []
    for idx in lip_landmarks:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * canvas.shape[1])
        y = int(landmark.y * canvas.shape[0])
        points.append((x, y))
    
    points = np.array(points, np.int32)
    points = points.reshape((-1, 1, 2))
    
    cv2.polylines(canvas, [points], True, (0, 0, 255), 3)  
    cv2.fillPoly(canvas, [points], (0, 0, 200))    

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
height, width, _ = frame.shape

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    canvas = np.zeros_like(image)
    
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            draw_eye_outline(canvas, face_landmarks)
            draw_iris_outline(canvas, face_landmarks)  
            draw_eyebrow_outline(canvas, face_landmarks)  
            draw_lip_outline(canvas, face_landmarks)
            
            mp_drawing.draw_landmarks(
                canvas,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,  
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                canvas,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
            ) 

    cv2.imshow('RalphDGAF', canvas)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
