"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)

    # checking if video is loaded
    # print(cap.isOpened())
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4

    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))


    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break
            if between(cap, 0, 500):
                # do something using OpenCV functions (skipped here so we simply write the input frame back to output)
                pass
            # ...

            # write frame that you processed to output
            out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Video Player', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release() # closes the video
    out.release()
    # Closes all the frames
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(1)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='OpenCV video processing')
    # parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    # parser.add_argument('-o', "--output", help='full path for saving processed video output')
    # args = parser.parse_args()
    #
    # if args.input is None or args.output is None:
    #     sys.exit("Please provide path to input and output video files! See --help")

    # main(args.input, args.output)
    main('Video/Input/in.mp4', 'Video/Output/out.mp4')
