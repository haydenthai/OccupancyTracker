# Python 3.11.9
from collections import defaultdict

import cv2
import supervision as sv
from flask import Flask
from flask import Response
from ultralytics import YOLO

app = Flask(__name__)


# @app.route('/webcam')
def webcam():
    # Use a breakpoint in the code line below to debug your script.

    model = YOLO('weights/best2.pt')
    print('loaded the model')

    # Set up video capture
    cap = cv2.VideoCapture("testing/best_video_maybe.mp4")

    # Define the line coordinates
    START = sv.Point(0, 240)
    END = sv.Point(600, 240)

    # Store the track history and initial positions
    track_history = defaultdict(lambda: [])
    initial_positions = defaultdict(lambda: None)

    # Create a dictionary to keep track of objects that have crossed the line
    crossed_objects = defaultdict(lambda: False)

    cnt = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:

            results = list(
                model.track(cv2.resize(frame, (600, 480)), persist=True, tracker="bytetrack.yaml", stream=True,
                            save_conf=True))

            annotated_frame = results[0].plot()
            # write annoted fram to a file
            # cv2.imwrite('annotated_frame.jpg', annotated_frame)
            detections = sv.Detections.from_ultralytics(results[0])

            coords = track_ids = []
            if detections.xyxy is not None:
                coords = detections.xyxy

            if detections.tracker_id is not None:
                track_ids = detections.tracker_id

            # Plot the tracks and count objects crossing the line
            for coord, track_id in zip(coords, track_ids):
                x1, y1, x2, y2 = coord
                track = track_history[track_id]

                center_y = float((y1 + y2) / 2)  # y center point
                track.append(center_y)
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                # Store the initial position of the object
                if initial_positions[track_id] is None:
                    initial_positions[track_id] = center_y
                    continue

                # Check if the object has crossed the line
                if len(track) > 1:
                    prev_y = initial_positions[track_id]
                    curr_y = track[-1]
                    if prev_y < START.y and curr_y > START.y:
                        cnt += 1
                    elif prev_y > START.y and curr_y < START.y:
                        cnt -= 1

                # Annotate the object as it crosses the line
                cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                              (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

            # Write the count of objects on each frame
            count_text = f"Objects crossed: {cnt}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            # cv2.imshow('Frame', annotated_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', annotated_frame)[1].tobytes() + b'\r\n')
            # Wait for a key press and close the window after a small delay
        else:
            break

        if cv2.waitKey(75) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/

@app.route('/')
def webcam_display():
    return Response(webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
