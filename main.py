import cv2
import pandas as pd
from datetime import datetime

first_frame = None
status_list = [None, None]
opening_times = []
closing_times = []
count = 0
video = cv2.VideoCapture(0)
times_df = pd.DataFrame(columns=["start", "end"])

while True:

    check, frame = video.read()
    status = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if count > 10:
        if first_frame is None:
            first_frame = gray
            continue

        delta_frame = cv2.absdiff(first_frame, gray)

        thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        (contours,_) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cont in contours:
            if cv2.contourArea(cont) < 10000:
                continue

            status = 1

            (x, y, w, h) = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        status_list.append(status)
        if status_list[-1] == 1 and status_list[-2] == 0:
            opening_times.append(datetime.now())
        if status_list[-1] == 0 and status_list[-2] == 1:
            closing_times.append(datetime.now())

        cv2.imshow("Gray Frame", gray)
        cv2.imshow("Delta Frame", delta_frame)
        cv2.imshow("Threshold Frame", thresh_frame)
        cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    count += 1

    if key == ord("q"):
        if status == 1:
            closing_times.append(datetime.now())
        break

times_df["start"] = opening_times
times_df["end"] = closing_times

times_df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows()