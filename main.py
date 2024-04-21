# This is a sample Python script.


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

# Python 3.11.9

def main():
    # Use a breakpoint in the code line below to debug your script.
    from collections import defaultdict
    from ultralytics import YOLO
    import cv2
    import supervision as sv

    model = YOLO('weights/best.pt')
    print('loaded the model')

    # Set up video capture
    cap = cv2.VideoCapture("testing/test.mp4")

    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # print(f'width: {width}, height: {height}')
    # 640 x 480

    # Define the line coordinates
    START = sv.Point(320, 0)
    END = sv.Point(320, 480)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Create a dictionary to keep track of objects that have crossed the line
    crossed_objects = defaultdict(lambda: False)

    cnt = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = list(model.track(frame, persist=True, tracker="bytetrack.yaml", stream=True, save_conf=True))

            annotated_frame = results[0].plot()
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

                # with open('detections.txt', 'a') as f:
                #     f.write(f'{track_id}, {x1}, {y1}, {x2}, {y2}'
                #             f'track: {track}'
                #             f'\n')
                track.append((float((x1 + x2) / 2), float((y1 + y2) / 2)))  # x, y center point
                if len(track) > 30:  # retain 30 tracks for 30 frames
                    track.pop(0)

                if len(track) > 1 and ((track[-2][0] < START.x and track[-1][0] > START.x) or (
                        track[-2][0] > START.x and track[-1][0] < START.x)):
                    print(f'track_id: {track_id}, crossed_objects: {crossed_objects}')
                    if track_id not in crossed_objects:
                        print('counted')
                        crossed_objects[track_id] = True

                    cnt += 1
                    # Annotate the object as it crosses the line
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                                  (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw the line on the frame
            cv2.line(annotated_frame, (START.x, START.y), (END.x, END.y), (0, 255, 0), 2)

            # Write the count of objects on each frame
            # count_text = f"Objects crossed: {len(crossed_objects)}"
            count_text = f"Objects crossed: {cnt}"
            cv2.putText(annotated_frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write the frame with annotations to the output video
            # sink.write_frame(annotated_frame)

            # Display the frame
            cv2.imshow('Frame', annotated_frame)

            # Wait for a key press and close the window after a small delay
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture
    cap.release()
    cv2.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    print('finished!')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
