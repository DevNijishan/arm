import cv2
import mediapipe as mp
import pyttsx3
import time

# Initialize
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

gesture_dict = {0: "PUNCH ", 1: "UP ", 5: "HI ", 2: "DOWN "}  # 0=fist,1=thumb up,5=palm
last_gesture = None
gesture_history = []
last_time = 0

def count_fingers(hand):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand.landmark[tips[0]].x < hand.landmark[tips[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers
    for id in range(1, 5):
        fingers.append(1 if hand.landmark[tips[id]].y <
                        hand.landmark[tips[id]-2].y else 0)
    return fingers.count(1)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms,
                                       mp_hands.HAND_CONNECTIONS)
                finger_count = count_fingers(handLms)
                if finger_count in gesture_dict:
                    text = gesture_dict[finger_count]
                    if text != last_gesture and time.time() - last_time > 1:
                        engine.say(text)
                        engine.runAndWait()
                        last_gesture = text
                        gesture_history.append(text)
                        last_time = time.time()

        disp = "".join(gesture_history[-10:])
        cv2.putText(frame, f"Text: {disp}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture to Text (Fast + Accurate)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
