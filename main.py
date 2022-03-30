import cv2
import imutils
import numpy as np
import pytesseract
import pandas as pd
import os
import time
from tracker import *
from _collections import deque
from datetime import datetime
import csv

# 檔案路徑
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
video_path = "vidoes/test.MOV"
save_video = "saved_videos/test.mp4"
file_path = "saved_data/license_plate.csv"
yaml_file_path = "saved_data/license_plate.yml"

picture_list = []
picture_time_list = []
picture_current_time = {}
csv_list = []
csv_time_list = []
csv_current_time = {}

cap = cv2.VideoCapture(video_path)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
tracker = EuclideanDistTracker()
pts = [deque(maxlen=10000) for _ in range(10000)]


# 選擇是否儲存影片
# 選擇是否儲存車牌圖片
# 選擇是否儲存車牌資料
config = {
    "save_video": False,
    "save_license_plate_picture": False,
    "save_data": False
}

# 儲存影片
if config["save_video"]:
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps_cur = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_video, fourcc, fps_cur, size, True)
    except:
        print("[INFO] could not determine in video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    local_time = time.time()
    current_time_string = datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    csv_current_date_string = datetime.now().strftime('%Y-%m-%d')
    csv_current_time_string = datetime.now().strftime('%H-%M-%S')

    # 車牌偵測
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    detections = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            rect = cv2.minAreaRect(approx)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            x_max, max_x = max(box, key=lambda item: item[0])
            x_min, min_x = min(box, key=lambda item: item[0])
            max_y, y_max = max(box, key=lambda item: item[1])
            min_y, y_min = min(box, key=lambda item: item[1])

            if ((x_max - x_min) / (y_max - y_min)) > 2 and ((x_max - x_min) / (y_max - y_min)) < 5.5:

                src = [[x_min, (y_min - 10)], [x_max, (y_min - 10)], [x_min, (y_max + 10)], [x_max, (y_max + 10)]]
                dst = [[0, 0], [(x_max - x_min), 0], [0, (y_max - y_min)], [(x_max - x_min), (y_max - y_min)]]

                src_list = np.float32(src)
                dst_list = np.float32(dst)
                M = cv2.getPerspectiveTransform(src_list, dst_list)
                warped_frame = cv2.warpPerspective(frame, M, ((x_max - x_min), (y_max - y_min)))

                if warped_frame is not None:
                    predicted_warped_roi = pytesseract.image_to_string(warped_frame, lang='eng', config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    if predicted_warped_roi != "" and len(predicted_warped_roi) >= 5:
                        screenCnt = approx
                        x, y, w, h = cv2.boundingRect(screenCnt)
                        detections.append([x, y, w, h])

                        roi = gray[(y - 10):(y + h + 10), x:(x + w)]
                        warped_roi = cv2.warpPerspective(frame, M, ((x_max - x_min), (y_max - y_min)))

    boxes_ids = tracker.update(detections)

    # 車牌辨識
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x + w, y - 20), (0, 255, 0), -1)
        indexID = int(box_id[4])

        if warped_roi is not None and screenCnt is not None:
            predicted_result = pytesseract.image_to_string(warped_roi, lang='eng', config='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
            filter_new_predicted_result = "".join(predicted_result.split()).replace(":", "").replace("-", "")

            if filter_new_predicted_result != "" and len(filter_new_predicted_result) >= 5:
                cv2.putText(frame, filter_new_predicted_result, (x + 10, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                pts[indexID].append(filter_new_predicted_result)
                df = pd.value_counts(pts[indexID]).rename_axis('license_plate').reset_index(name='counts')

                # 儲存車牌照片
                if config["save_license_plate_picture"]:
                    for index, i in enumerate(df['counts']):
                        if df['license_plate'][index] not in picture_list and i == 6:
                            picture_current_time[indexID] = time.time()
                            picture_list.append(df['license_plate'][index])
                            picture_time_list.append(picture_current_time[indexID])
                            cv2.imwrite(os.path.join("saved_license_plate", df['license_plate'][index]) + "_" + current_time_string + ".jpg", roi)

                        if picture_list != [] and picture_time_list != []:
                            for j in picture_time_list:
                                if (local_time - j) > 30:
                                    picture_time_index = picture_time_list.index(j)
                                    picture_time_list.remove(j)
                                    picture_list.remove(picture_list[picture_time_index])

                        if df['license_plate'][index] in picture_list and i == 6:
                            pass

                # 儲存車牌資料
                if config["save_data"]:
                    with open(file_path, 'a', encoding='UTF8', newline="") as csv_file:
                        for index, i in enumerate(df['counts']):
                            if df['license_plate'][index] not in csv_list and i == 6:
                                csv_current_time[indexID] = time.time()
                                csv_list.append(df['license_plate'][index])
                                csv_time_list.append(csv_current_time[indexID])
                                writer = csv.writer(csv_file)
                                writer.writerow([df['license_plate'][index], csv_current_date_string, csv_current_time_string])

                            if csv_list != [] and csv_time_list != []:
                                for j in csv_time_list:
                                    if (local_time - j) > 30:
                                        csv_time_index = csv_time_list.index(j)
                                        csv_time_list.remove(j)
                                        csv_list.remove(csv_list[csv_time_index])

                            if df['license_plate'][index] in csv_list and i == 6:
                                pass

    cv2.imshow("frame", frame)

    if config["save_video"]:
        writer.write(frame)

    # 按下鍵盤 Esc 關閉影片
    if cv2.waitKey(1) == 27:
        break

if config["save_video"]:
    writer.release()

cap.release()
cv2.destroyAllWindows()