def extract_palm(image_path, output_path, margin=20):
    import mediapipe as mp
    import cv2

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            h, w, _ = image.shape
            landmark_ids = [0, 1, 5, 9, 13, 17]
            hand_landmarks = results.multi_hand_landmarks[0]
            coords = [(int(hand_landmarks.landmark[i].x * w),
                       int(hand_landmarks.landmark[i].y * h)) for i in landmark_ids]

            x_vals, y_vals = zip(*coords)
            x_min = max(min(x_vals) - margin, 0)
            x_max = min(max(x_vals) + margin, w)
            y_min = max(min(y_vals) - margin, 0)
            y_max = min(max(y_vals) + margin, h)

            palm_crop = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(output_path, palm_crop)
            return palm_crop
        else:
            raise ValueError(f"No hand detected in image: {image_path}")
