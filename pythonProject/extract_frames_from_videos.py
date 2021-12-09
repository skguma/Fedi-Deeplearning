import cv2

# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
i = 0
frame_num = 1

while(cap.isOpened() and i<=frame_num):

    ret, frame = cap.read()

    # This condition prevents from infinite looping
    # incase video ends.
    if ret == False:
        break

    cv2.imshow("preview",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Save Frame by Frame into disk using imwrite method
    #if (i>10):
    cv2.imwrite("test/img7.png", frame)
    i += 1

cap.release()
cv2.destroyAllWindows()