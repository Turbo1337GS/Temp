import cv2
import mediapipe as mp
import pyautogui
import numpy as np

# Inicjalizacja MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Współczynnik wygładzania ruchu kursora
smoothing = 5
prev_x, prev_y = 0, 0

def is_fist_closed(hand_landmarks):
    tip_of_middle_finger = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    base_of_palm = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

    distance = np.sqrt((tip_of_middle_finger.x - base_of_palm.x) ** 2 +
                       (tip_of_middle_finger.y - base_of_palm.y) ** 2)

    return distance < 0.1
def check_thumb_index_up(hand_landmarks):
    # Pobieranie punktów orientacyjnych dla palców
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
    middle_pip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y
    ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y
    pinky_pip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y
    
    # Sprawdzenie, czy koniec kciuka i palca wskazującego są wyżej niż środkowe punkty pozostałych palców
    if (thumb_tip < middle_pip and index_tip < middle_pip and
        index_tip < ring_pip and index_tip < pinky_pip):
        return True
    else:
        return False
def calculate_finger_distance(hand_landmarks):
    # Współrzędne punktów orientacyjnych kciuka (THUMB_TIP) i palca wskazującego (INDEX_FINGER_TIP)
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Obliczanie odległości euklidesowej między dwoma punktami
    distance = np.sqrt((index_finger_tip.x - thumb_tip.x) ** 2 +
                       (index_finger_tip.y - thumb_tip.y) ** 2 +
                       (index_finger_tip.z - thumb_tip.z) ** 2)
    print(f'distance ',{distance})    
    return distance


def convert_coordinates(hand_landmarks, image_width, image_height):
    landmark = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    x = int(landmark.x * image_width)
    y = int(landmark.y * image_height)
    screen_width, screen_height = pyautogui.size()
    scaled_x = np.interp(x, (0, image_width), (0, screen_width))
    scaled_y = np.interp(y, (0, image_height), (0, screen_height))
    return scaled_x, scaled_y

# Uruchomienie kamery internetowej
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Nie można odczytać obrazu z kamery.")
        break

    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Rysowanie wykrytych dłoni na obrazie
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Sterowanie kursorem
            cursor_position = convert_coordinates(hand_landmarks, image_width, image_height)
            if cursor_position and is_fist_closed(hand_landmarks) < 0.1 :
                new_x = prev_x + (cursor_position[0] - prev_x) / smoothing
                new_y = prev_y + (cursor_position[1] - prev_y) / smoothing
                pyautogui.moveTo(new_x, new_y)
                prev_x, prev_y = new_x, new_y

            # # Kliknięcie lewym przyciskiem myszy, jeśli pięść jest zamknięta
            if is_fist_closed(hand_landmarks):
                pyautogui.click()
                print('click')
                # Dodanie opóźnienia, aby uniknąć wielokrotnych kliknięć
                cv2.waitKey(200)
            if check_thumb_index_up(hand_landmarks):
                calculate_finger_distance(hand_landmarks)
    cv2.imshow('Hand Tracking', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
