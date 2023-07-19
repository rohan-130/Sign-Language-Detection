import pickle
import time
import cv2
from tracker import HandTracker
from words import Words


def train_word(word_name, wait_time=3, train_time=4, num_iterations=3, pickle_file="words.pkl"):
    with open(pickle_file, 'rb') as file:
        words = pickle.load(file)
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    start_time = None
    new_result = []
    i = 0
    result = []
    while i < num_iterations:
        success, image = cap.read()
        image = tracker.hands_finder(image)
        lmList = tracker.position_finder(image)
        cv2.putText(image, "Train " + word_name, (50, 200), 0, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if len(lmList) == 0:
            start_time = None
        else:
            start_time = time.time() if start_time is None else start_time
        if start_time is None:
            cv2.putText(image, "Hand not detected", (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Video", image)
            cv2.waitKey(1)
            continue
        duration = int(time.time() - start_time) if start_time is not None else 0
        if duration <= wait_time:
            cv2.putText(image, str(wait_time - duration), (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif duration >= train_time + wait_time:
            if start_time is not None:
                start_time = None
                i += 1
                for j in range(len(new_result) - 1):
                    new_result[j] = (new_result[j + 1][0] - new_result[j][0], new_result[j + 1][1] - new_result[j][1])
                new_result = new_result[:-1]
                try:
                    answer = str(words.check_word(new_result))
                except:
                    print("Not detected")
                result.append(new_result)
                new_result = []
        else:
            val = lmList  # lmList[10][1], lmList[10][2]
            new_result.append(val)
            cv2.putText(image, str(val), (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Video", image)
        cv2.waitKey(1)
    print(result)
    words.update_word({word_name: result})
    print(words.all_words)
    with open(pickle_file, 'wb') as file1:
        pickle.dump(words, file1, pickle.HIGHEST_PROTOCOL)


def check_words(wait_time=3, train_time=4, pickle_file="words.pkl"):
    with open(pickle_file, 'rb') as file:
        words = pickle.load(file)
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    start_time = None
    new_result = []
    i = 0
    answer = "None"
    while True:  # i < num_iterations:
        success, image = cap.read()
        image = tracker.hands_finder(image)
        lmList = tracker.position_finder(image)
        cv2.putText(image, str(answer), (50, 200), 0, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if len(lmList) == 0:
            start_time = None
        else:
            start_time = time.time() if start_time is None else start_time
        if start_time is None:
            cv2.putText(image, "Hand not detected", (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow("Video", image)
            cv2.waitKey(1)
            continue
        duration = int(time.time() - start_time) if start_time is not None else 0
        if duration <= wait_time:
            cv2.putText(image, str(wait_time - duration), (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
        elif duration >= train_time + wait_time:
            if start_time is not None:
                start_time = None
                i += 1
                for j in range(len(new_result) - 1):
                    new_result[j] = (new_result[j + 1][0] - new_result[j][0], new_result[j + 1][1] - new_result[j][1])
                new_result = new_result[:-1]
                answer = words.check_word(new_result)  # [0][:-1]
                new_result = []
        else:
            val = lmList  # lmList[4][1], lmList[4][2]
            new_result.append(val)
            cv2.putText(image, str(val), (50, 100), 0, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("Video", image)
        cv2.waitKey(1)


# train_word('sleep', wait_time=2, train_time=3, num_iterations=2, pickle_file="words.pkl")
# train_word('hello', wait_time=2, train_time=3, num_iterations=2, pickle_file="words.pkl")
# train_word('gift', wait_time=2, train_time=3, num_iterations=2, pickle_file="words.pkl")
# train_word('dizzy', wait_time=2, train_time=3, num_iterations=2, pickle_file="words.pkl")
check_words(wait_time=3, train_time=4, pickle_file="words.pkl")
