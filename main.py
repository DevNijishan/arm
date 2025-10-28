# --- AI Hand Gesture Controlled Robotic Arm ---
# Works with: Raspberry Pi 5 + USB Webcam + SG90 Servos (no PCA9685)

import cv2
import mediapipe as mp
import RPi.GPIO as GPIO
import time

# --- GPIO setup ---
servo_pins = [17, 27, 22, 23]  # Base, Shoulder, Elbow, Gripper
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)

servos = [GPIO.PWM(pin, 50) for pin in servo_pins]  # 50Hz PWM
for s in servos:
    s.start(0)
    time.sleep(0.2)

def set_angle(servo, angle):
    """Convert angle (0–180) to duty cycle for SG90"""
    duty = 2 + (angle / 18)
    servo.ChangeDutyCycle(duty)

# --- Hand detection setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# --- Open webcam ---
cap = cv2.VideoCapture(0)

print("✅ System Ready: Show your hand to the camera!")
print("🖐️  Thumb + Index distance = Gripper control")
print("✋ Move hand vertically = Arm Up/Down")
print("Press 'q' to quit.")

base_angle = 90
shoulder_angle = 90
elbow_angle = 90
gripper_angle = 90

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # --- Get landmark positions ---
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

                # --- Calculate gesture metrics ---
                distance = abs(index_tip.y - thumb_tip.y)
                height = wrist.y  # hand height (0 top, 1 bottom)

                # --- Gripper control (open/close) ---
                if distance < 0.05:
                    gripper_angle = 40  # close
                else:
                    gripper_angle = 90  # open
                set_angle(servos[3], gripper_angle)

                # --- Arm up/down control (wrist height) ---
                if height < 0.4:
                    shoulder_angle = min(shoulder_angle + 2, 160)
                elif height > 0.6:
                    shoulder_angle = max(shoulder_angle - 2, 30)
                set_angle(servos[1], shoulder_angle)

                # --- Optional base rotation using x-position ---
                hand_x = wrist.x
                if hand_x < 0.4:
                    base_angle = min(base_angle + 2, 160)
                elif hand_x > 0.6:
                    base_angle = max(base_angle - 2, 30)
                set_angle(servos[0], base_angle)

        # --- Display status on screen ---
        status = f"Base:{base_angle}  Shoulder:{shoulder_angle}  Gripper:{gripper_angle}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("AI Robotic Arm Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
for s in servos:
    s.stop()
GPIO.cleanup()
print("👋 Program Ended. Servos Released.")
