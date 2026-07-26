import os
import csv
import cv2
import mediapipe as mp
from tqdm import tqdm

# ==========================
# Paths
# ==========================
DATASET_PATH = "data/raw/asl_alphabet_train"
OUTPUT_CSV = "data/processed/landmarks.csv"

# ==========================
# MediaPipe
# ==========================
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# ==========================
# CSV Header
# ==========================
header = []

for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])

header.append("label")

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

processed = 0
skipped = 0

classes = sorted(os.listdir(DATASET_PATH))

for label in classes:

    class_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)

    for image_name in tqdm(images, desc=f"Processing {label}"):

        image_path = os.path.join(class_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]

            row = []

            for lm in hand.landmark:
                row.extend([lm.x, lm.y, lm.z])

            row.append(label)

            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            processed += 1

        else:
            skipped += 1

hands.close()

print("\n==============================")
print("Finished!")
print("==============================")
print(f"Processed : {processed}")
print(f"Skipped   : {skipped}")
print(f"CSV Saved : {OUTPUT_CSV}")