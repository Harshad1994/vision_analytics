import cv2
import os
import numpy as np


#controls whether to compare with all previous distinct images or with just latest
compare_all=True

##SSIM structural similarity index
ssim_score_threshold=0.92

def get_ssim(imageA, imageB):
  """Calculates the structural similarity index (SSIM) between two images.

  Args:
    imageA: The first image.
    imageB: The second image.

  Returns:
    A similarity score between 0 and 1, where 1 means the images are identical
    and 0 means the images are completely different.
  """

  # Convert the images to grayscale.
  imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

  # Calculate the mean (mu) and standard deviation (sigma) of each image.
  mu1 = np.mean(imageA_gray)
  mu2 = np.mean(imageB_gray)
  sigma1 = np.std(imageA_gray)
  sigma2 = np.std(imageB_gray)

  # Calculate the covariance (sigma12) of the two images.
  sigma12 = np.cov(imageA_gray.ravel(), imageB_gray.ravel())[0][1]

  # Calculate the SSIM between the two images.
  ssim = ((2 * mu1 * mu2 + 0.01) * (2 * sigma12 + 0.03)) / ((mu1**2 + mu2**2 + 0.01) * (sigma1**2 + sigma2**2 + 0.03))

  return ssim



def get_distinct_frames_between_times(filepath,time_frames, resolution=None):

    '''
    Finds distinct frames between time stamps

    Args:
        filepath: type str, path to file location
        time_frames: list of tuple [(a1,b1),(a2,b2)]
        resolution: type float, gap between two consecutive frames in seconds


    Returns:
        dictionary with keys are timestamps provided and values as 
        list of distinct frames between time stamps

    '''


    video = cv2.VideoCapture(filepath)

    if resolution==None:
        #0.5 seconds default resolution
        resolution=0.5

    distinct_frames_dict={}

    #compare this image with all previous images
    
    for item in time_frames:

        distinct_in_timespan=[]

        start=item[0]
        stop=item[1]

        time_sec=start


        while time_sec<stop:

            t_msec = 1000*time_sec
            video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
            ret, frame = video.read()

            if ret!=True:
                time_sec+=resolution
                continue

            if len(distinct_in_timespan)==0:
                previous=frame
                distinct_in_timespan.append(frame)
                time_sec+=resolution
                continue
            

            if compare_all:
                ssim_score=compare_with_all_previous(frame,distinct_in_timespan)
            else:
                ssim_score = get_ssim(frame, previous)

            if ssim_score<ssim_score_threshold:
                previous=frame
                distinct_in_timespan.append(frame)
                # #dump this frame with its time stand
                # name=str(time_sec)+"_sec_new_frame.png"

                # filepath=os.path.join('/home/harshad/workspace/dtp_proj/image_dump',name)
                # cv2.imwrite(filepath, new_frame)
                
            time_sec+=resolution

        distinct_frames_dict[item]=distinct_in_timespan


    return distinct_frames_dict


def compare_with_all_previous(frame,distinct_in_timespan):

    '''
    Function compares this frame with all previous distinct images in this time span
    returns ssim=0.95 if it matches with any of the previous so as to not consider distinct
    '''

    for item in distinct_in_timespan[::-1]:
        ssim_score = get_ssim(frame, item)

        if ssim_score > ssim_score_threshold:
            return ssim_score_threshold+0.2
    
    return ssim_score_threshold-0.2


def get_video_duration(video):
    '''
    returns the video duration in seconds
    '''

    # Get the length of the video
    fps = video.get(cv2.CAP_PROP_FPS)
    totalNoFrames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    durationInSeconds = totalNoFrames / fps

    return durationInSeconds

def dump_images(frames_dict,save_location):

    '''
    function to save all distinct frames in a folder
    '''

    counter=0

    for _,v in frames_dict.items():

        for frame in v:
            name=str(counter)+"_image.png"
            savepath=os.path.join(save_location,name)
            cv2.imwrite(savepath, frame)

            counter+=1
          
    return "Saved Successfully"



if __name__=="__main__":

    file=r'/home/harshad/workspace/video.mp4'
    save_location=r'/home/harshad/workspace/temp_dump'

    times=[(115,126),(200,228)]
    output_frames=get_distinct_frames_between_times(file,times)

    print(dump_images(output_frames,save_location))
