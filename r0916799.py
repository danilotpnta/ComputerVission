import cv2
import numpy as np

# Constants   # (B, G, R)
blue_color    = (250,0,35)
purple_color  = (210,99,188)
orange2_color = (0, 128, 255)
orange_colour = (54,121,194)
green_colour  = (65,255,48)
red_colour    = (76,0,255)
rosy_colour   = (255,20,238)
brown_colour  = (26,29,54)

# HSV min and max values for color range
minHSV = np.array([20, 125, 100])       # Yellow
maxHSV = np.array([30, 255, 255])
minHSV_b = np.array([0, 0, 0])        # Brown 2snd
maxHSV_b = np.array([255, 255, 106])
minHSV_iris = np.array([75, 0, 0])    # Iris 2nd
maxHSV_iris = np.array([153, 255, 255])

# Creates Image 3 Channel Solid color
blue_solid_Color = np.zeros((720, 1280, 3), dtype = np.uint8)
blue_solid_Color[:,:] = blue_color
purple_solid_Color = np.zeros((720, 1280, 3), dtype = np.uint8)
purple_solid_Color[:,:] = purple_color
orange_solid_Color = np.zeros((720, 1280, 3), dtype = np.uint8)
orange_solid_Color[:,:] = orange_colour
green_solid_Color = np.zeros((720, 1280, 3), dtype = np.uint8)
green_solid_Color[:,:] = green_colour
black_solid_Color_uint8 = np.zeros((720, 1280), dtype = np.uint8)
black_solid_Color_float32 = np.zeros((720, 1280), dtype = np.float32)


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper

# Write text
def writeText(frame, text, x, y, fontScale = None, thickness = None):
    if fontScale == None:
        fontScale = 1
    else:
        fontScale = fontScale

    if thickness == None:
        thickness = 2
    else:
        thickness = thickness

    cv2.putText(frame, text,
                (x, y),                                # cordinates
                cv2.FONT_HERSHEY_SIMPLEX, fontScale,   # font, fontScale
                (0, 255, 255), thickness )             # color, thickness

# Gaussian Filter
def gaussianFilter(frame, size_kernel):
    kernel = np.ones((size_kernel,size_kernel),np.float32)/size_kernel**2
    return cv2.filter2D(frame,-1,kernel)

# Bilateral Filter
def bilateralFilter(frame, kernelSize):
    return cv2.bilateralFilter(frame,kernelSize,95,95)

def main(input_video_file: str, output_video_file: str, input_imageTemplate: str) -> None:
    cap = cv2.VideoCapture(input_video_file)
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

    # Reading Template image
    template_gray = cv2.imread(input_imageTemplate, cv2.IMREAD_GRAYSCALE)

    # Codinates eyes
    last_x_iris, last_y_iris, last_r_iris = 0, 0, 0
    last_x_pupil, last_y_pupil, last_r_pupil = 0, 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        l = frame_width

        if ret == True:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            if between(cap, 0, 4 *1000):
                if between(cap, 0, 2 *1000):
                    writeText(frame,f'BGR Frame', 50, 100 )

                if between(cap, 2 *1000, 4 *1000):
                    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)
                    writeText(frame,f'GRAY Frame', 50, 100 )

            if between(cap, 4 *1000, 12 *1000):

                if between(cap, 4 *1000, 8 *1000):
                    kernelSize = 5; kernelSize2 = 50
                    frame1 = gaussianFilter(frame, kernelSize)
                    frame2 = gaussianFilter(frame, kernelSize2)
                    frame[:, 0:int(l/2)] = frame1[:, 0:int(l/2)];
                    frame[:, int(l/2):l] = frame2[:, int(l/2):l];
                    writeText(frame, f'gaussianFilter: {kernelSize} KernelSize', 50, 100 )
                    writeText(frame, f'gaussianFilter: {kernelSize2} KernelSize', 690, 100 )
                    writeText(frame, f'Edges are not conserved', 690, 650 )

                if between(cap, 8 *1000, 12 *1000):
                    kernelSize = 7; kernelSize2 = 70
                    frame1 = bilateralFilter(frame,kernelSize)
                    frame2 = bilateralFilter(frame,kernelSize2)
                    frame[:, 0:int(l/2)] = frame1[:, 0:int(l/2)];
                    frame[:, int(l/2):l] = frame2[:, int(l/2):l];
                    writeText(frame,f'bilateralFilter: {kernelSize} KernelSize', 50, 100 )
                    writeText(frame,f'bilateralFilter: {kernelSize2} KernelSize', 690, 100 )
                    writeText(frame, f'Edges are conserved', 690, 650 )
                    writeText(frame, f'while noise is reduced', 690, 685 )

            if between(cap, 12 *1000, 20 *1000):

                if between(cap, 12*1000, 14*1000):
                    writeText(frame,f'Original Frame', 50, 100 )

                if between(cap, 14*1000, 16.5*1000):

                    # transform frame to HSV
                    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Select only desired 'COLOUR'
                    mask = cv2.inRange(frame_HSV, minHSV, maxHSV)

                    # Selecting only what is primary color
                    foreground = cv2.bitwise_or(frame, frame, mask=mask)

                    # Create 1280x700 White Solid Rectangle
                    background = np.full(frame_HSV.shape, 255, dtype=np.uint8)

                    frame_without_morp = cv2.bitwise_or(foreground, background, mask=mask)
                    frame = frame_without_morp
                    writeText(frame,f'Grabbing without Morphological Op.', 50, 100 )

                if between(cap, 16.5*1000, 20*1000):
                    # transform frame to HSV
                    frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # Select only desired 'COLOUR'
                    mask = cv2.inRange(frame_HSV, minHSV, maxHSV)

                    # Selecting only what is primary color
                    foreground = cv2.bitwise_or(frame, frame, mask=mask)

                    # Create 1280x700 White Solid Rectangle
                    background = np.full(frame_HSV.shape, 255, dtype=np.uint8)

                    frame_without_morp = cv2.bitwise_or(foreground, background, mask=mask)

                    # Applying morphological operations
                    # Erosion
                    kSize = 3
                    kernel = np.ones((kSize,kSize,), np.uint8)
                    erosion = cv2.erode(frame_without_morp, kernel, iterations = 1)

                    # Closing
                    kSize = 3
                    kernel = np.ones((kSize,kSize,), np.uint8)
                    closing = cv2.morphologyEx(frame_without_morp, cv2.MORPH_CLOSE, kernel)

                    # Opening
                    kSize = 3
                    kernel = np.ones((kSize,kSize,), np.uint8)
                    opening = cv2.morphologyEx(frame_without_morp, cv2.MORPH_OPEN, kernel)

                    frame_plus_morp = cv2.bitwise_or(blue_solid_Color,opening, mask=mask)
                    frame = frame_plus_morp
                    writeText(frame,f'Grabbing + Morphological Op.', 50, 100 )
                    writeText(frame, f'* Erosion: Noise is reduced bottom-left corner Corn Label', 50, 650, 0.8 )
                    writeText(frame, f'* Dilatation: Fill holes in Choco-Circles & Cutter Handel', 50, 685, 0.8 )

            offset = 20

            if between(cap, (offset)*1000, (offset+1)*1000):
                writeText(frame, 'Original Frame', 50,100)

            if between(cap, (offset+1)*1000, (offset+5)*1000):

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kSize = 15

                # Removing noise
                frame = bilateralFilter(gray,kSize)

                if between(cap, (offset+1)*1000, (offset+2.5)*1000):
                    kSizeSobel = 1

                if between(cap, (offset+2.5)*1000, (offset+5)*1000):
                    kSizeSobel = 3

                grad_x = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=kSizeSobel)
                grad_y = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=kSizeSobel)

                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                mask1Ch_combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

                abs_grad_x  = cv2.cvtColor(abs_grad_x, cv2.COLOR_GRAY2BGR)
                abs_grad_y  = cv2.cvtColor(abs_grad_y, cv2.COLOR_GRAY2BGR)

                frame_coloured_x = cv2.bitwise_and(green_solid_Color,abs_grad_x)
                frame_coloured_y = cv2.bitwise_and(blue_solid_Color,abs_grad_y)


                combined = cv2.bitwise_or(frame_coloured_x,frame_coloured_y,mask=mask1Ch_combined)
                frame = combined
                writeText(frame, f'Sobel Edge Dectector', 50,100)
                writeText(frame, f'KernelSize: {kSizeSobel}', 50,150)

            Hough_time = 5

            if between(cap, (Hough_time+offset)*1000, (Hough_time+offset+10)*1000):
                frame = cv2.medianBlur(frame,3)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if between(cap, (Hough_time+offset)*1000, (Hough_time+offset+3.5)*1000):
                    minDist = 5
                    param1 = 100
                    param2 = 40
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2)

                    if circles is not None:
                        detected_circles = np.uint16(np.around(circles))

                        for x, y, r in detected_circles[0,:]:
                            frame = cv2.circle(frame, (x, y), r, (0, 0, 255), 2)

                    writeText(frame, f'Hough Transform', 50,100)
                    writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                    writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                    writeText(frame, f'param2: {param2}', 50,200,0.7,2)

                if between(cap, (Hough_time+offset+3.5)*1000, (Hough_time+offset+7)*1000):
                    minDist = 5
                    param1 = 130
                    param2 = 100
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2)

                    if circles is not None:
                        detected_circles = np.uint16(np.around(circles))

                        for x, y, r in detected_circles[0,:]:
                            frame = cv2.circle(frame, (x, y), r, (255, 0, 0), 2)

                    writeText(frame, f'Hough Transform', 50,100)
                    writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                    writeText(frame, f'param1: {param1} (*)', 50,170,0.7,2)
                    writeText(frame, f'param2: {param2} (*)', 50,200,0.7,2)
                    writeText(frame, f'(*) Change in value', 50, 650, 0.7,2)

                if between(cap, (Hough_time+offset+7)*1000, (Hough_time+offset+10)*1000):
                    minDist = 20
                    param1 = 130
                    param2 = 100
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2)

                    if circles is not None:
                        detected_circles = np.uint16(np.around(circles))

                        for x, y, r in detected_circles[0,:]:
                            frame = cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

                    writeText(frame, f'Hough Transform', 50,100)
                    writeText(frame, f'minDist: {minDist} (*)', 50,140,0.7,2)
                    writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                    writeText(frame, f'param2: {param2}', 50,200,0.7,2)
                    writeText(frame, f'(*) Change in value', 50, 650, 0.7,2)

            deel3 = 10

            if between(cap, (deel3 + Hough_time + offset)*1000, (deel3 + Hough_time+offset+5)*1000):
                frame = cv2.medianBlur(frame,3)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if between(cap, (deel3+Hough_time+offset)*1000, (deel3 + Hough_time+offset+2)*1000):
                    minDist = 20
                    param1 = 130
                    param2 = 90
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=minDist, param1=param1, param2=param2)

                    if circles is not None:
                        detected_circles = np.uint16(np.around(circles))

                        for x, y, r in detected_circles[0,:]:
                            frame = cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), rosy_colour, 3)

                    writeText(frame, f'Object Detection', 50,100)

                if between(cap, (deel3+Hough_time+offset + 2)*1000, (deel3 + Hough_time+offset+5)*1000):

                    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                    frame = result

                    x , y = result.shape
                    offset_x  = int((720-x)/2) - 10
                    offset_y = int((1280-y)/2)
                    black_solid_Color_float32[offset_x:x+offset_x, offset_y:y+offset_y] = frame[0:x, 0:y]
                    frame = black_solid_Color_float32
                    frame = (frame*255)
                    frame = np.where(frame > 0, frame, 0).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                    writeText(frame, f'Template Matching', 50,100)
                    writeText(frame, f'Center of Object: White 100%', 690,100)


            deel4 = 40
            if between(cap, (deel4 + 0)*1000, (deel4 + 60)*1000):
                frame_copy = frame.copy()
                original_frame = frame

                # Transform frame to HSV
                frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Select only desired 'COLOUR'
                mask_pupil = cv2.inRange(frame_HSV, minHSV_b, maxHSV_b)
                mask_iris = cv2.inRange(frame_HSV, minHSV_iris, maxHSV_iris)
                mask = cv2.subtract(mask_pupil, mask_iris)

                # Selecting only what is primary color
                foreground = cv2.bitwise_or(frame, frame, mask=mask)
                foreground_copy = foreground.copy()

                # Opening
                kSize = 1
                kernel = np.ones((kSize,kSize,), np.uint8)
                opening = cv2.morphologyEx(foreground_copy, cv2.MORPH_OPEN, kernel)
                foreground_copy = opening

                # Closing
                kSize = 13
                kernel = np.ones((kSize,kSize,), np.uint8)
                closing = cv2.morphologyEx(foreground_copy, cv2.MORPH_CLOSE, kernel)

                # Dilatation
                kSize = 13
                kernel = np.ones((kSize, kSize), np.uint8)
                dilatation = cv2.dilate(closing, kernel, iterations=1)

                foreground_copy_gray = cv2.cvtColor(dilatation, cv2.COLOR_BGR2GRAY)

                # Closing
                kSize = 5
                kernel = np.ones((kSize,kSize,), np.uint8)
                closing = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
                # cv2.imshow('closing', closing)

                # Removing noise
                frame_blur = cv2.medianBlur(closing,7)
                gray = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2GRAY)
                # cv2.imshow('gray', gray)

                # Draw a circle around Pupil
                minDist = 300
                param1 = 30
                param2 = 0.8
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, minDist=minDist,
                                           param1=param1, param2=param2,
                                           minRadius=80,maxRadius=90)  # brown pupil

                if circles is not None:
                    detected_circles = np.uint16(np.around(circles))
                    for x, y, r in detected_circles[0,:]:
                        if between(cap, (deel4 + 4)*1000, (deel4 + 7)*1000):
                            frame = cv2.circle(original_frame, (x, y), r, rosy_colour, 3)
                            writeText(frame, f'*Pupil Detection', 50, 650,)
                            writeText(frame, f'Hough Transform', 50,100)
                            writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                            writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                            writeText(frame, f'param2: {param2}', 50,200,0.7,2)
                        foreground = cv2.circle(foreground, (x, y), r, (0, 255, 0), 2)
                        last_x_pupil = x; last_y_pupil = y; last_r_pupil = r

                else:
                    if between(cap, (deel4 + 4)*1000, (deel4 + 7)*1000):
                        frame = cv2.circle(original_frame, (last_x_pupil, last_y_pupil), last_r_pupil, rosy_colour, 3)
                        writeText(frame, f'*Pupil Detection', 50, 650)
                        writeText(frame, f'Hough Transform', 50,100)
                        writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                        writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                        writeText(frame, f'param2: {param2}', 50,200,0.7,2)
                    foreground = cv2.circle(foreground, (last_x_pupil, last_y_pupil), last_r_pupil, (0, 255, 0), 2)

                # created mask for displaying pupil only
                offset_radius = 9
                mask1 = np.zeros_like(gray)
                mask1 = cv2.circle(mask1, (last_x_pupil,last_y_pupil), last_r_pupil + offset_radius, (255,255,255), -1)
                cutted_pupil = cv2.bitwise_or(foreground_copy, foreground_copy, mask=mask1)

                # removing Noise
                cutted_pupil = cv2.medianBlur(cutted_pupil,9)

                # Dilatation
                kSize = 3
                kernel = np.ones((kSize, kSize), np.uint8)
                cutted_pupil = cv2.dilate(cutted_pupil, kernel, iterations=2)
                cutted_pupil_gray = cv2.bitwise_or(foreground_copy_gray, foreground_copy_gray, mask=mask1)

                hsv_cutPupil = cv2.cvtColor(cutted_pupil, cv2.COLOR_BGR2HSV)
                h,s,v = cv2.split(hsv_cutPupil)
                hsv_new = cv2.merge([h+57,s,v+3])  # green
                bgr_new = cv2.cvtColor(hsv_new, cv2.COLOR_HSV2BGR)
                bgr_new = cv2.bitwise_or(bgr_new, bgr_new, mask=cutted_pupil_gray)
                bgr_new_pupil = bgr_new

                bgr_new = cv2.bitwise_or(frame_copy, bgr_new_pupil)
                kSize = 3
                bgr_new = bilateralFilter(bgr_new,kSize)

                if between(cap, (deel4 + 0)*1000, (deel4 + 4)*1000):

                    # Draw a circle around Iris
                    minDist = 700
                    param1 = 40
                    param2 = 0.8
                    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT_ALT, 1, minDist=minDist,
                                               param1=param1, param2=param2,
                                               minRadius=18,maxRadius=29)  # iris

                    if circles is not None:
                        detected_circles = np.uint16(np.around(circles))
                        for x, y, r in detected_circles[0,:]:
                            frame = cv2.circle(frame_copy, (x, y), r, (0, 255, 0), 2)
                            last_x_iris = x; last_y_iris = y; last_r_iris = r
                            writeText(frame, f'*Iris Detection', 50, 650)
                            writeText(frame, f'Hough Transform', 50,100)
                            writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                            writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                            writeText(frame, f'param2: {param2}', 50,200,0.7,2)
                    else:
                        frame = cv2.circle(frame_copy, (last_x_iris, last_y_iris), last_r_iris, (0, 255, 0), 2)
                        writeText(frame, f'*Iris Detection', 50, 650)
                        writeText(frame, f'Hough Transform', 50,100)
                        writeText(frame, f'minDist: {minDist}', 50,140,0.7,2)
                        writeText(frame, f'param1: {param1}', 50,170,0.7,2)
                        writeText(frame, f'param2: {param2}', 50,200,0.7,2)


                # Saving ouput frames to VideoWriter
                if between(cap, (deel4 + 7)*1000, (deel4 + 9)*1000):
                    frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    writeText(frame, f'Object Detection', 50,100)
                    writeText(frame, f'(*) Isolation of brown_colour using inRange', 50, 650, 0.7,2)
                if between(cap, (deel4 + 9)*1000, (deel4 + 11)*1000):
                    frame = foreground
                    writeText(frame, f'Masking Pupil + brown_colour: green circle', 50,100)
                    writeText(frame, f'(*) MedianBlur filter applied to reduce noise', 50, 650, 0.7,2)
                if between(cap, (deel4 + 11)*1000, (deel4 + 13)*1000):
                    frame = cutted_pupil
                    writeText(frame, f'Morphological op. after masking Pupil only', 50,100)
                    writeText(frame, f'(*) Opening -> Closing -> Dilatation', 50, 650, 0.7,2)
                if between(cap, (deel4 + 13)*1000, (deel4 + 15)*1000):
                    frame = bgr_new_pupil
                    writeText(frame, f'Changing Hue value of Pupil', 50,100)
                    writeText(frame, f'(*) To be mixed with Original frame', 50,650)
                if between(cap, (deel4 + 15)*1000, (deel4 + 18)*1000):
                    frame = bgr_new
                    writeText(frame, f'End Result!', 50,100,1,2)
                    writeText(frame, f'* Manipulation of Pupil color', 50,650)


            # Inserting current_time on video
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
    inVid_location       = 'Video/Input/in.mov'
    outVid_location      = 'Video/Output/out.mp4'
    imgTemplate_location = 'Video/Input/template3.png'
    main(inVid_location, outVid_location, imgTemplate_location)
