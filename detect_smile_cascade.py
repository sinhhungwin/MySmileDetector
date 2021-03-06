import cv2

face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('cascade/haarcascade_smile.xml')


def detect(g, f):
    faces = face_cascade.detectMultiScale(g, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(f, (x, y), ((x + w), (y + h)), (255, 0, 0), 2)
        roi_gray = g[y:y + h, x:x + w]
        roi_color = f[y:y + h, x:x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), ((sx + sw), (sy + sh)), (0, 0, 255), 2)
    return f


video_capture = cv2.VideoCapture(0)


# Grab width and height from video feed
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer = cv2.VideoWriter('haar.mp4', cv2.VideoWriter_fourcc(*'VIDX'), 25, (width, height))

while video_capture.isOpened():
    # Captures video_capture frame by frame
    _, frame = video_capture.read()

    # To capture image in monochrome
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calls the detect() function
    canvas = detect(gray, frame)

    writer.write(canvas)

    # Displays the result on camera feed
    cv2.imshow('Video', canvas)

    # The control breaks once q key is pressed
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the capture once all the processing is done.
video_capture.release()
writer.release()
cv2.destroyAllWindows()
