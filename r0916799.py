import sys

import cv2
import numpy as np
from matplotlib import pyplot as plt

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# Write text
def writeText(frame, text, x, y):
    cv2.putText(frame, text,
                (x, y),                        # cordinates
                cv2.FONT_HERSHEY_SIMPLEX, 1,   # font, fontScale
                (0, 255, 255), 2 )             # color, thickness

# Gaussian Filter
def gaussianFilter(frame, size_kernel):
    kernel = np.ones((size_kernel,size_kernel),np.float32)/size_kernel**2
    return cv2.filter2D(frame,-1,kernel)

# Bilateral Filter
def bilateralFilter(frame, kernelSize):
    return cv2.bilateralFilter(frame,kernelSize,95,95)

def main(input_video_file: str, output_video_file: str) -> None:
    cap = cv2.VideoCapture(input_video_file)

    # Metada of Video
    frame_width = int(cap.get(3))   # 1280
    frame_height = int(cap.get(4))  # 720
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(round(cap.get(5)))
    len_vid = num_frames/fps

    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # For placing text
    font = cv2.FONT_HERSHEY_SIMPLEX

    while cap.isOpened():
        ret, frame = cap.read()
        l = frame_width;

        if ret == True:

            if between(cap, 0, 4 *1000):
                grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

            if between(cap, 4 *1000, 12 *1000):
                # sys.exit()
                if between(cap, 4 *1000, 8 *1000):
                    kernelSize = 5; kernelSize2 = 20
                    frame1 = gaussianFilter(frame, kernelSize)
                    frame2 = gaussianFilter(frame, kernelSize2)
                    frame[:, 0:int(l/2)] = frame1[:, 0:int(l/2)];
                    frame[:, int(l/2):l] = frame2[:, int(l/2):l];
                    writeText(frame, f'gaussianFilter: {kernelSize}', 50, 100 )
                    writeText(frame, f'gaussianFilter: {kernelSize2}', 690, 100 )

                if between(cap, 8 *1000, 12 *1000):
                    kernelSize = 15; kernelSize2 = 45
                    frame1 = bilateralFilter(frame,kernelSize)
                    frame2 = bilateralFilter(frame,kernelSize2)
                    frame[:, 0:int(l/2)] = frame1[:, 0:int(l/2)];
                    frame[:, int(l/2):l] = frame2[:, int(l/2):l];
                    writeText(frame,f'bilateralFilter: {kernelSize}', 50, 100 )
                    writeText(frame,f'bilateralFilter: {kernelSize2}', 690, 100 )


            # inserting current_time on video
            current_time = round(cap.get(cv2.CAP_PROP_POS_MSEC)/1000, 2)
            writeText(frame, f'Time: {current_time}sec', 50,50)
            cv2.imshow('Video Player', frame)
            out.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    cap.release() #closes the Video
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main('Video/Input/in3.MOV', 'Video/Output/out3.mp4')
