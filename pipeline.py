#!/usr/bin/python

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy.ma as ma
from moviepy.editor import VideoFileClip

image_size = ()

def imageShow(img):
    fig, ax = plt.subplots()
    cax = ax.imshow(img)
    fig.colorbar(cax)
    plt.show()

def slidingWindowPolyfit(binary_warped, Minv, undistorted):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    #plt.imshow(out_img)
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')
    #plt.xlim(0, 1280)
    #plt.ylim(720, 0)
    #plt.show()

    #print(ploty.shape, leftx.shape, rightx.shape)
    

    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(ploty)

    ym_per_pix = 30/720.0 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undistorted.shape[1], undistorted.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    cv2.putText(result ,'Left curvature: {}'.format(left_curverad),(10,100), \
            cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result ,'Right curvature: {}'.format(right_curverad),(10,130), \
            cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2,cv2.LINE_AA)

    #plt.imshow(result)
    #plt.show()
    return result


def calibrateCamera():
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    imgPoints = []
    objPoints = []
    img_size = None
    for img in os.listdir('camera_cal'):
        print('Processing: {}'.format(img))
        cal_img = cv2.imread(os.path.join('camera_cal', img))
        gray = cv2.cvtColor(cal_img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        img_size = (cal_img.shape[1], cal_img.shape[0])

        if success:
            print(objp.shape)
            print(corners.shape)
            objPoints.append(objp)
            imgPoints.append(corners)
            #cv2.drawChessboardCorners(gray, (9,6), corners, success)
            #cv2.imshow('img', gray)
            #cv2.waitKey(500)
    #cv2.destroyAllWindows()

    print(len(objPoints))
    print(len(imgPoints))
    print(objPoints[0].shape)
    print(imgPoints[0].shape)
    # imgPoints = np.array(imgPoints)
    # objPoints = np.array(objPoints)
    #print(imgPoints.shape, objPoints.shape)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, img_size,None,None) 
    return (ret, mtx, dist)

def saturationThresholding(hls_img):
    saturationThreshold = (80, 150)
    s_channel = hls_img[:,:,2]
    #imageShow(s_channel)
    # Normalize s channel 
    #s_channel = np.absolute(s_channel)
    #s_channel = np.uint8(255 * (s_channel / np.max(s_channel)))

    # Thresholding
    mask = np.zeros_like(s_channel)
    mask [(s_channel >= saturationThreshold[0]) & (s_channel <= saturationThreshold[1])] = 1
    return mask

def gradientThresholding(rgb_img):
    gradient_direction_threshold = (10 , 150)
    sobel_kernel_size = 3
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    sobel_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel_size))

    sobel_x = np.uint8(255 * sobel_x / np.max(sobel_x))

    img_mask = np.zeros_like(sobel_x)
    img_mask[(sobel_x >= gradient_direction_threshold[0]) & (sobel_x <=
        gradient_direction_threshold[1])] = 1
    return img_mask

def roiTransform():
    global image_size
    src_roi = np.float32(np.array([[575, 450],\
            [725, 450],\
            [1200, 720],\
            [200, 720]]))
    dst_roi = np.float32(np.array([[300, 0],\
            [950, 0],\
            [950, 720],\
            [300, 720]]))

    M = cv2.getPerspectiveTransform(src_roi, dst_roi)
    MInv = cv2.getPerspectiveTransform(dst_roi, src_roi)
    return (M, MInv, src_roi)

def calcCurvatures(left_fit, right_fit, y_coordinate):
    left_curverad = ((1 + (2*left_fit[0]*y_coordinate + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_coordinate + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    return(left_curverad, right_curverad)

# def testPipeline(rgb_img, C, distCoeff):
def testPipeline(rgb_img):
    global C
    global distCoeff

    # Undistort image
    # imageShow(rgb_img)
    undistorted = cv2.undistort(rgb_img, C, distCoeff)
    #imageShow(undistorted)
    
    # Convert color space
    hls_img = cv2.cvtColor(undistorted, cv2.COLOR_RGB2HLS)
    #imageShow(hls_img)

    # Saturation thresholding
    saturation_mask = saturationThresholding(hls_img)
    #imageShow(saturation_mask)

    # Gradient thresholding
    gradient_mask = gradientThresholding(rgb_img)
    #imageShow(saturation_mask)

    # Combine masks
    combined_scaled_mask = 255 * np.dstack((np.zeros_like(gradient_mask), saturation_mask, gradient_mask))
    #imageShow(combined_scaled_mask)

    combined_binary_mask = np.zeros_like(gradient_mask)
    combined_binary_mask[(saturation_mask == 1) | (gradient_mask == 1)] = 1
    #imageShow(combined_binary_mask)

    M, MInv, src_roi = roiTransform()

    polygon_image = rgb_img.copy()
    cv2.polylines(polygon_image, np.int32([src_roi]), True, (255, 0, 0), thickness = 2)
    #imageShow(polygon_image)

    warped_image = cv2.warpPerspective(combined_binary_mask, M, (1280, 720))
    #warped_image = cv2.warpPerspective(rgb_img, M, (1280, 720))
    #imageShow(warped_image)

    return slidingWindowPolyfit(warped_image, MInv, undistorted)
    #return combined_scaled_mask


if __name__ == '__main__':
    global image_size
    global C
    global distCoeff

    success, C, distCoeff = calibrateCamera()
    if success:
        # Run on testing imagesb
        #for img in os.listdir('test_images'):
        #    rgb_img = mpimg.imread(os.path.join('test_images', img))
        #    image_size = rgb_img.shape
        #    testPipeline(rgb_img)
        #    #break;

        # Run on video
        input_video = os.path.join('.', 'project_video.mp4')
        output_video = os.path.join('.', 'project_video_output.mp4')
        clip = VideoFileClip(input_video)
        output_clip = clip.fl_image(testPipeline)
        output_clip.write_videofile(output_video, audio=False)

