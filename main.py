import os
from yolodetector import imgyolo3, detectionyolo3
import pandas as pd
import cv2
import numpy as np
from time import sleep

main_path = (
    r"E:\DER-MG\1. CRGs\CRG 23\Videos\773LMG0040-C-1-0-27\CAMpp"
)
network, layers_names_output, colours, labels = imgyolo3().load_network()

object_list_end = []
for file in os.listdir(main_path):
    if file.endswith(".jpg"):
        print(file)
        img_path = os.path.join(main_path, file)
        image, object_list = detectionyolo3(img_path, 
                                            network, 
                                            0.10, 
                                            0.10, 
                                            colours, 
                                            layers_names_output, 
                                            labels
                                            ).run_detection()
        image = cv2.resize(image,(600,600))
        cv2.imshow("image",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(object_list) != 0:
            print(object_list)
            sleep(1)


# print(object_list_end)

# main_path = (
#     r"C:\Users\lucas\OneDrive\Documentos\TCC\Repos\yolo-detector\Videos\panrem-teste.asf"
# )
# network, layers_names_output, colours, labels = imgyolo3().load_network()

# # cap = cv2.VideoCapture(main_path)
# # while(cap.isOpened()):
# #   # Capture frame-by-frame
# #   ret, frame = cap.read()
# #   if ret == True:
# #     img, object_list = detectionyolo3(frame, 
# #                                         network, 
# #                                         0.30, 
# #                                         0.15, 
# #                                         colours, 
# #                                         layers_names_output, 
# #                                         labels
# #                                         ).run_detection()
# #     # Display the resulting frame
# #     cv2.imshow('Frame',img)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #       break

# cap = cv2.VideoCapture(main_path)
# while cap.isOpened():
#     ret, frame = cap.read()

#     aspect_ratio = 1280 / 446
#     points = [(150, 155), (320, 155), (360, 210), (70, 210)]
#     adjusted_points = [(int(x * aspect_ratio), int(y * aspect_ratio)) for x, y in points]

#     src_points = np.array(adjusted_points, dtype=np.float32)

#     height, width, _ = frame.shape

#     # Adjust the destination points for a proper perspective transformation
#     dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

#     perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#     per_frame = cv2.warpPerspective(frame, perspective_matrix, (width, height))  # Keep the same dimensions as the original frame
#     per_frame = cv2.resize(per_frame, (600, 600))

#     if not ret:
#         print("End of video")
#         break
    

#     img, object_list = detectionyolo3(per_frame, 
#                                     network, 
#                                     0.30, 
#                                     0.15, 
#                                     colours, 
#                                     layers_names_output, 
#                                     labels
#                                     ).run_detection()
    
#     # Draw points on the original frame
#     for point in adjusted_points:
#         cv2.circle(frame, (point[0], point[1]), 5, (0, 0, 255), -1)  # Red circle at each point

#     # Resize frame_with_points to match per_frame
#     frame_with_points = cv2.resize(frame, (600, 600))

#     # Concatenate the original frame with points and per_frame vertically
#     stacked_frames = np.vstack([frame_with_points, img])

#     cv2.imshow("Combined Frames", stacked_frames)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

#     if len(object_list) != 0:
#         sleep(2)

# cv2.destroyAllWindows()