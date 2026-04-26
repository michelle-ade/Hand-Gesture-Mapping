import mediapipe as mp
import cv2 as cv
import numpy as np

def test_calc_bounding_rect():

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    image = cv.flip(cv.imread("test_hand_img.png"), 1)
    results = hands.process(image)
    landmarks = results.multi_hand_landmarks[0]

    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for landmark in landmarks.landmark: 

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    assert [x, y, x + w, y + h] == [102, 136, 249, 341], "Bounding Rectangle Points should be [102, 136, 249, 341]"
