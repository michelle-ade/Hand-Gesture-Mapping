import mediapipe as mp
import cv2 as cv
import itertools

def test_pre_process_landmark():
    #Calculate Landmark List------------------------------

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
    
    landmark_list = []
    
    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        print("\n Landmark: ")
        print([landmark_x, landmark_y])
        landmark_list.append([landmark_x, landmark_y])
    
    processed_landmark_list = landmark_list

    #Process Landmarks------------------
    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(processed_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        processed_landmark_list[index][0] = processed_landmark_list[index][0] - base_x
        processed_landmark_list[index][1] = processed_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    processed_landmark_list = list(
        itertools.chain.from_iterable(processed_landmark_list))

    # Normalization
    max_value = max(list(map(abs, processed_landmark_list)))

    def normalize_(n):
        return n / max_value

    processed_landmark_list = list(map(normalize_, processed_landmark_list))

    for i in range(len(processed_landmark_list)):
        assert(processed_landmark_list[i]) >= -1, "Processed Point should be >= -1"
        assert(processed_landmark_list[i]) <= 1, "Processed Point should be <= 1"
