from cam_backend import *

print('Starting the download loop')
last_time_fetched = time.time() # get the current time
first_run = True # set a flag to indicate that this is the first run of the loop (for the first run, we will download rss feeds for all the buoys)
duplicate_removal_flag = True # set this flag to true if we want to remove duplicated images with difPy
#note: bugs are present in difPy, so this flag is set to false
exper_1 = False # flag for dupe detect in panels
verbose_wait = False # flag that makes waiting show a progress bar.

# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import glob
rotating = True # flag to indicate if the tapestry is rotating
panel_mode = False # flag to indicate if we want to use panels for color detection
def is_recent(file, minutes):
    # get the time the image was taken
    image_time = os.path.getmtime(file)
    # get the current time
    current_time = time.time()
    # get the difference between the two times
    time_difference = current_time - image_time
    # if the time difference is less than minutes, return true
    if time_difference < minutes*60:
        return True
    else:
        return False

def crop_the_bottom_off(images):
    # for each of the images crop the bottom off (20 pixels)
    for image in images:
        try:
            # get the image size
            img_width, img_height = get_image_size(image)
            # crop the bottom off
            cropped_image = image.crop((0, 0, img_width, img_height-20))
            # save the image
            cropped_image.save(image)
        except Exception as e:
            print("Error cropping the bottom off of the image: " + str(e))

def check_colors_of_six_panels(image):
    # there are six panels in the image (side by side) and we want to check the colors of each panel
    # get the image size
    # the input image should be an image object

    img_width, img_height = get_image_size(image)
    # get the width of each panel
    panel_width = img_width/6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height-20
    # get the colors of each panel
    panel_1 = image.getpixel((panel_width/2, panel_height/2))
    panel_2 = image.getpixel((panel_width*1.5, panel_height/2))
    panel_3 = image.getpixel((panel_width*2.5, panel_height/2))
    panel_4 = image.getpixel((panel_width*3.5, panel_height/2))
    panel_5 = image.getpixel((panel_width*4.5, panel_height/2))
    panel_6 = image.getpixel((panel_width*5.5, panel_height/2))
    # return the number of panels that meet the color criteria:
    # criteria:
    # red = less than 250 and greater than 170
    # green = less than 250 and greater than 170
    # blue = less than 250 and greater than 170
    # orange = less than 250 and greater than 170

    # check panel 1
    # if panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'white'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 170:
    #     panel_1_color = 'yellow'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'green'
    # elif panel_1[0] < 170 and panel_1[1] < 250 and panel_1[1] > 170 and panel_1[2] < 250 and panel_1[2] > 170:
    #     panel_1_color = 'blue'
    # elif panel_1[0] < 250 and panel_1[0] > 170 and panel_1[1] < 170 and panel_1[2] < 170:
    #     panel_1_color = 'red'
    # else:
    #     panel_1_color = 'other'
    # this is an interesting code but let's do something simpler, I want to isolate images that have sunsets in them, so let's just check the red and blue values
    if panel_1[0] > 200 and panel_1[2] > 200: # if the red and blue values are greater than 200, then it's a sunset?
        panel_1_result = True # set the result to true
    else:
        panel_1_result = False
    # check panel 2
    if panel_2[0] > 200 and panel_2[2] > 200:
        panel_2_result = True
    else:
        panel_2_result = False
    # check panel 3
    if panel_3[0] > 200 and panel_3[2] > 200:
        panel_3_result = True
    # check panel 4
    if panel_4[0] > 200 and panel_4[2] > 200:
        panel_4_result = True
    # check panel 5
    if panel_5[0] > 200 and panel_5[2] > 200:
        panel_5_result = True
    # check panel 6
    if panel_6[0] > 200 and panel_6[2] > 200:
        panel_6_result = True
    # return the results
    panels_collection = [panel_1, panel_2, panel_3, panel_4, panel_5, panel_6] # put the panels into a list
    return panel_1, panel_2, panel_3, panel_4, panel_5, panel_6, panel_1_result, panel_2_result, panel_3_result, panel_4_result, panel_5_result, panel_6_result


def get_panel_segments(image):
    """
    get_panel_segments takes an image and returns the segments of the image that are the panels

    :param image: the image to be segmented

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the segments of the image that are the panels
    :rtype: list of image objects
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the width of each panel
    panel_width = img_width/6
    # get the height of each panel (remove botttom 20 pixels)
    panel_height = img_height-20
    # cast them to integers
    panel_width = int(panel_width)
    panel_height = int(panel_height)

    # get the segments of the image that are the panels
    panel_1 = image[0:panel_height, 0:panel_width] # to overcome the error: slice indices must be integers or None or have an __index__ method
    # we have to convert the panel_1 to an image object
    panel_2 = image[0:panel_height, panel_width:panel_width*2]
    panel_3 = image[0:panel_height, panel_width*2:panel_width*3]
    panel_4 = image[0:panel_height, panel_width*3:panel_width*4]
    panel_5 = image[0:panel_height, panel_width*4:panel_width*5]
    panel_6 = image[0:panel_height, panel_width*5:panel_width*6]
    # return the segments
    panels = [panel_1, panel_2, panel_3, panel_4, panel_5, panel_6]
    return panels

def get_average_color(image):
    """
    get_average_color takes an image and returns the average color of the image

    :param image: the image to be segmented
    :type image: image object (from PIL) or numpy array (from OpenCV)
    :return: the average color of the image
    :rtype: tuple of integers
    """
    # get the image size
    img_width, img_height = image.shape[0], image.shape[1]
    # get the average color of the image
    # to do this we have to convert the image from a numpy array to a PIL image
    image = Image.fromarray(image)
    average_color = image.getpixel((img_width//2, img_height//2))
    # return the average color
    return average_color

def generate_one_full_day(buoy_id, day_number, month_number, year_number):
    # concatenate one image for each hour of the day and save it as a new image called 'full_day.png'
    # get the images from the buoy_id folder and sort them
    images = os.listdir(buoy_id)
    images.sort()
    # the names of the images are in the format: '2019-01-01_00.png'
    # images/buoys/41004/2022_11_5_15_44.jpg
    # where the day is the 3rd element in the list
    # and the hour is the 4th element in the list
    # so we can use a list comprehension to get the images for each hour of the day if we sort the images first.
    # go through the folder and add the filenames to the list if the day is the same as the day_number
    # and the month is the same as the month_number
    # and the year is the same as the year_number

    # get the images for the day
    images_for_day = [image for image in images if image.split('_')[2] == str(day_number) and image.split('_')[1] == str(month_number) and image.split('_')[0] == str(year_number)]
    # get the images for the day and sort them
    images_for_day.sort()
    # get one image for each hour of the day
    images_for_day = [image for image in images_for_day if image.split('_')[3] == '00' or image.split('_')[3] == '01' or image.split('_')[3] == '02' or image.split('_')[3] == '03' or image.split('_')[3] == '04' or image.split('_')[3] == '05' or image.split('_')[3] == '06' or image.split('_')[3] == '07' or image.split('_')[3] == '08' or image.split('_')[3] == '09' or image.split('_')[3] == '10' or image.split('_')[3] == '11' or image.split('_')[3] == '12' or image.split('_')[3] == '13' or image.split('_')[3] == '14' or image.split('_')[3] == '15' or image.split('_')[3] == '16' or image.split('_')[3] == '17' or image.split('_')[3] == '18' or image.split('_')[3] == '19' or image.split('_')[3] == '20' or image.split('_')[3] == '21' or image.split('_')[3] == '22' or image.split('_')[3] == '23']
    # get the images for the day and sort them
    images_for_day.sort()
    # add the images one after the other by concatenating them vertically and save the result as a new image called 'full_day.png' in the buoy_id folder
    # get the first image
    first_image = images_for_day[0]
    # open the first image
    first_image = Image.open(buoy_id + '/' + first_image)
    # get the width and height of the first image
    width, height = first_image.size
    # create a new image with the same width as the first image and the height of the first image multiplied by the number of images
    new_image = Image.new('RGB', (width, height*len(images_for_day)))
    # paste the first image in the new image
    new_image.paste(first_image, (0, 0))
    # go through the rest of the images and add them to the bottom of the new image one after the other
    for i in range(1, len(images_for_day)):
        # get the image
        image = images_for_day[i]
        # open the image
        image = Image.open(buoy_id + '/' + image)
        # paste the image in the new image
        new_image.paste(image, (0, height*i))
    # save the new image
    new_image.save(buoy_id + '/full_day.png')



### Testing To find Red

def finding_red_version_two(image):
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_height = image.shape[0]
    image_width = image.shape[1]
    # https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    image_result = np.zeros((image_height,image_width,3),np.uint8)
    for i in range(image_height):  #those are set elsewhere
        for j in range(image_width): #those are set elsewhere
            if img_hsv[i][j][1]>=50 \
                and img_hsv[i][j][2]>=50 \
                and (img_hsv[i][j][0] <= 10 or img_hsv[i][j][0]>=170):
                image_result[i][j]=img_hsv[i][j] # this is the red (above is saturation, value, and hue)
    return image_result

def finding_red_version_three(image_path):
    """
    finding_red_version_three takes an image and returns the red pixels in the image

    :param image_path: the path to the image to be segmented

    :param image_path: the path to the image to be segmented
    :type image_path: string
    :return: the red pixels in the image
    :rtype: image object
    """
    img=cv2.imread(image_path)
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    # set my output img to zero everywhere except my mask
    #output_img = img.copy()
    #output_img[np.where(mask==0)] = 0

    # or your HSV image, which I *believe* is what you want
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0

    # save the output_img to 'test.png' in images folder
    #cv2.imwrite('images/test.png', output_img)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite('images/test_hsv.png', output_hsv)
    # why is the output_hsv image all black?
    # because the output_hsv image is in HSV format and not RGB format
    # so we have to convert it to RGB format
    # https://stackoverflow.com/questions/15007348/convert-hsv-to-rgb-using-python-and-opencv
    # converting the image from HSV to RGB
    output_hsv = cv2.cvtColor(output_hsv, cv2.COLOR_HSV2RGB)
    # save the output_hsv to 'test_hsv.png' in images folder
    cv2.imwrite('images/test_hsv.png', output_hsv)


def detect_red_v4(image):
    # Red color
    if type(image) == str:
        image = cv2.imread(image) # read the image
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    red_mask = cv2.inRange(image, low_red, high_red)
    percent_pixels_red = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    return percent_pixels_red


def detect_red(img):
    """
    detect_red _summary_

    _extended_summary_

    :param image: _description_
    :type image:
    :return: _description_
    :rtype: _type_
    """
    try:
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # creating a mask to catch the color red in the image
        # Here, you define your target color as
        # a tuple of three values: RGB
        # red = [130, 158, 0]
        red = [0, 0, 255] # this is the color of the red in the image


        # You define an interval that covers the values
        # in the tuple and are below and above them by 20
        diff = 20

        # Be aware that opencv loads image in BGR format,
        # that's why the color values have been adjusted here:
        boundaries = [([red[2], red[1]-diff, red[0]-diff],
                [red[2]+diff, red[1]+diff, red[0]+diff])]

        # Scale your BIG image into a small one:
        scalePercent = 0.3

        # Calculate the new dimensions
        width = int(img.shape[1] * scalePercent)
        height = int(img.shape[0] * scalePercent)
        newSize = (width, height)

        # Resize the image:
        img = cv2.resize(img, newSize, None, None, None, cv2.INTER_AREA)

        # check out the image resized:
        #!cv2.imshow("img resized", img)
        #!cv2.waitKey(0)


        # for each range in your boundary list:
        for (lower, upper) in boundaries:

            # You get the lower and upper part of the interval:
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)

            # cv2.inRange is used to binarize (i.e., render in white/black) an image
            # All the pixels that fall inside your interval [lower, uipper] will be white
            # All the pixels that do not fall inside this interval will
            # be rendered in black, for all three channels:
            mask = cv2.inRange(img, lower, upper)

            # Check out the binary mask:
            #!cv2.imshow("binary mask", mask)
            #cv2.waitKey(0)

            # Now, you AND the mask and the input image
            # All the pixels that are white in the mask will
            # survive the AND operation, all the black pixels
            # will remain black
            output = cv2.bitwise_and(img, img, mask=mask)

            # Check out the ANDed mask:
            #!cv2.imshow("ANDed mask", output)
            #cv2.waitKey(0)

            # You can use the mask to count the number of white pixels.
            # Remember that the white pixels in the mask are those that
            # fall in your defined range, that is, every white pixel corresponds
            # to a red pixel. Divide by the image size and you got the
            # percentage of red pixels in the original image:
            ratio_red = cv2.countNonZero(mask)/(img.size/3)

            # This is the color percent calculation, considering the resize I did earlier.
            colorPercent = (ratio_red * 100) / scalePercent

            # Print the color percent, use 2 figures past the decimal point
            print('red pixel percentage:', np.round(colorPercent, 2))

            # numpy's hstack is used to stack two images horizontally,
            # so you see the various images generated in one figure:
            #!cv2.imshow("images", np.hstack([img, output]))
            # save the image as 'test.png' in the images folder
            cv2.imwrite('images/test.png', np.hstack([img, output]))
            #cv2.waitKey(0)
    except Exception as e:
        print("Error in detect_red: ", e)

def deal_with_white_images_and_populate_tapestry():
    sunsets_found = 0 # keep track of how many sunsets we find
    files =glob.glob('images/buoys/*/*')
    # without glob
    #files = []
    #for file in os.listdir('images/buoys/'):
    #   files.append('images/buoys/' + file)
    #print(files)
    height, width, channels = cv2.imread(files[0]).shape
    # blank_image = np.zeros((height*10, width, channels), np.uint8)
    # get the ten images that have the most orange in them
    # make the blank image the same size as the images


    # shuffle the files so we don't always get the same ten images
    #!random.shuffle(files) #note: this could be a problem later

    add_list = []


    for file in tqdm(files):
        # read the image
        try:
            image = cv2.imread(file)
            if not is_recent(file, 300): # 300 minutes
                continue

            if panel_mode:
                # get the image details for panels 1-6
                panel_1, panel_2, panel_3, panel_4, panel_5, panel_6, = get_panel_segments(image)
                # explanation of results:
                panels_collection = [panel_1, panel_2, panel_3, panel_4, panel_5,panel_6] # put the panels into a list


                # put True into panel_results once for each panel (1/6 of the width of the image) that has an average red value greater than 180 and an average blue value greater than 180

                # the image passed to get_panel_segments should be a cv2 image
                assert(type(image) == np.ndarray)

                panel_segments = get_panel_segments(image)

                positive_panel_count = 0 # keep track of how many panels have a sunset in them
                # get the average color of each panel
                for panel in panel_segments:
                    panel_average_color = get_average_color(panel)
                    #check if the panel is a sunset
                    if panel_average_color[0] > 200 and panel_average_color[2] > 200:
                        # increment the positive_panel_count by 1
                        positive_panel_count += 1

                # now check if the positive_panel_count is greater than 3 (i.e. more than half of the panels have a sunset in them)
                if positive_panel_count > 3:
                    add_list.append(file)
                    sunsets_found += 1
                else:
                    continue # if the positive_panel_count is not greater than 3, then continue on to the next image

                # what is the average amount of red in the image?
                # what is the average amount of blue in the image?
                # what is the average amount of green in the image?
                # what is the average amount of orange in the image?


                #x = [[panel_1_result, panel_2_result, panel_3_result, panel_4_result, panel_5_result, panel_6_result],panels_collection] # return the results
                #panel_results - a list of true or false values for each panel (true if the panel is orange, false if not).
                #panels_collection - a list of the colors of each panel (in RGB format) (this is for debugging purposes)
                #if the image has at least 4 panels that are orange, then we want to add it to the tapestry

                #note: uncomment below if the check_colors_of_six_panels function is not working
                # # get the average orange value
                # orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel
                # red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image

                # get the median orange value across the panels
                panels_collection = panel_segments # put the panels into a list
                orange_value = np.median([panels_collection[0][2], panels_collection[1][2], panels_collection[2][2], panels_collection[3][2], panels_collection[4][2], panels_collection[5][2]])
                # # get the median red value across the panels
                red_value = np.median([panels_collection[0][0], panels_collection[1][0], panels_collection[2][0], panels_collection[3][0], panels_collection[4][0], panels_collection[5][0]])

            # if the average amount of orange is greater than 200, then add the image to the add_list

            # if the average amount of orange is greater than 200:
                # add_list.append(file)
                # sunsets_found += 1
            # else:
                # continue # if the average amount of orange is not greater than 200, then continue on to the next image

            #* just check the image to see if red is less than 20, green is less than 20, and blue is less than 20
            #* if so then skip the image

            red_score = np.mean(image[:,:,0])
            green_score = np.mean(image[:,:,1])
            blue_score = np.mean(image[:,:,2])
            if red_score < 20 and green_score < 20 and blue_score < 20:
                # print('Night image detected')
                continue
            else:
                #print('Day image detected')
                red_val = detect_red_v4(image)
                if red_val > 2.5:
                    print(' ---- > Sunset detected? ', red_val) # print the sunset detected message
                    # save the image to the sunset folder under the appropriate buoy
                    buoy_name = file.split('/')[2]
                    buoy_folder = 'images/sunsets/' + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + '/' + file.split('/')[3], image)
                    red_flag = True
                elif red_val > 15:
                    print(' ---- > super sunset detected? ', red_val) # print the sunset detected message
                    # save the image to the keepers folder under the appropriate buoy
                    buoy_name = file.split('/')[2]
                    buoy_folder = 'images/keepers/' + buoy_name
                    if not os.path.exists(buoy_folder):
                        os.makedirs(buoy_folder)
                    cv2.imwrite(buoy_folder + '/' + file.split('/')[3], image)
                    red_flag = True
                else:
                    red_flag = False
                if red_flag: # if the image has more than 10% red in it
                    add_list.append(file)
                    sunsets_found += 1



            #blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # save the filename to a list if the image is to be added to the tapestry


        except Exception as e:
            print(e)
            continue


        blank_image = np.zeros((height*len(add_list), width, channels), np.uint8)
        try:
            cv2.imwrite('images/tapestry.png', blank_image)
        except Exception as e:
            print(e)
            print('Could not write blank image')
            print('line 322')
            continue




    for file in tqdm(add_list):
        # read the image
        try:
            image = cv2.imread(file)
            # get the average orange value
            # print(np.mean(image[:,:,2]))
            #orange_value = np.mean(image[:,:,2]) # the orange channel is the third channel

            #red_value = np.mean(image[:,:,0]) # get the average red value when over 147 then save the image


            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            # if orange_value < 10 and red_value < 10: # higher than 250 for all means it is a white imag
            #     continue
            # # if the values are all higher than 250 then it is a white image and we want to remove it
            # if orange_value > 250 and red_value > 250:
            #     os.remove(file)
            #     print("Removed white image")
            #     continue
            # # if the image was not taken in the last x hours, skip it
            # if not is_recent(file, 60): # 60 minutes
            #     continue

            #blue_value = np.mean(image[:,:,1]) # blue value
            # print(orange_value, red_value, blue_value)
            # show the image and annotate it with the orange, red, and blue values
            # plt.imshow(image)
            # plt.title("Orange: " + str(orange_value) + " Red: " + str(red_value) + " Blue: " + str(blue_value))
            # plt.show()

            # if we reached this point the image can be added to the tapestry unless the tapestry has already been filled then just keep going without adding the image
            if rotating: # if the tapestry is rotating, we take an image and add it to the tapestry as well as remove the oldest image otherwise we just add the image to the tapestry
                if sunsets_found == 10:
                    # remove the top image from the tapestry
                    # get the image at the top of the tapestry which has a height of total_height/10
                    top_image = blank_image[0:height, 0:width]
                    # crop the image to remove the top 1/10th of the image
                    blank_image = blank_image[height:height*10, 0:width]
                    # add the new image to the bottom of the tapestry
                    blank_image = np.concatenate((blank_image, image), axis=0)
                    cv2.imwrite('images/tapestry.png', blank_image)
                else:
                    blank_image[sunsets_found*height:(sunsets_found+1)*height, 0:width] = image
                    # show progress by printing out the blank image
                    cv2.imwrite('images/tapestry.png', blank_image)
                    #print("Sunset found!")
                    sunsets_found += 1 # increment the number of sunsets found

            else:
                blank_image[sunsets_found*height:(sunsets_found+1)*height, 0:width] = image
                # show progress by printing out the blank image
                cv2.imwrite('images/tapestry.png', blank_image)
                #print("Sunset found!")
                sunsets_found += 1 # increment the number of sunsets found
        except:
            print("Error reading image")
            print('line 386')
            pass


def stitched_panoramas(panel1, panel2, panel3, panel4, panel5, panel6):
    # get the image size
    img_width, img_height = panel1.shape[1], panel1.shape[0]
    # get the ratio of the width to height
    r = float(img_width)/float(img_height)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-panel1.get_top(), -panel1.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((img_width/2, img_height/2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(panel1, m, (img_width, img_height))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width/2, new_im.shape.height/2))
    # crop the image to the correct size
    new_im = new_im.copy()
    #!cropped_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(rotated_im, (int(round(ar*img_width)), int(round(ar*img_height))))
    return resized_im

def get_image_size(image):
    """
    get_image_size returns the width and height of an image

    _extended_summary_

    :param image: the image to get the size of
    :type image: cv2 image
    :return: the width and height of the image
    :rtype: tuple
    """
    # get the image width and height
    w, h = image.shape[:2]
    # I am getting Exception has occurred: ValueError
    # too many values to unpack (expected 2)
    # the way to fix this is...
    # w, h = image.shape[:2]

    # get the ratio of the width to height
    r = float(w)/float(h)
    # get the aspect ratio of the image
    ar = round(r, 2)
    # calculate the rotation angle
    rot = math.degrees(math.atan2(-image.get_top(), -image.get_left()))
    # get the rotation matrix for this angle
    m = cv2.getRotationMatrix2D((w/2, h/2), rot, 1)
    # multiply the two matrices together to get the final transformation
    new_im = cv2.warpAffine(image, m, (w, h))
    # find the top-left corner of the image
    pos = np.array((new_im.shape.width/2, new_im.shape.height/2))
    # crop the image to the correct size
    new_im = new_im.crop(pos)
    # rotate the image back
    rotated_im = cv2.rotate(new_im, 90, pos, 0, 0, 360)
    # resize the image to the right size
    resized_im = cv2.resize(rotated_im, (int(round(ar*w)), int(round(ar*h))))
    return resized_im



# Implementing the trained h5 model in real-time using OpenCV and Python to detect scene elements or types in panel images from the cameras on the buoys.
# The model is already trained and saved in the models/buoy_model/keras_model.h5 file.

# # import the necessary packages
# from imutils import paths
# import numpy as np
# import cv2

# # import the load_model function from keras
# from keras.models import load_model

# # load the model from disk
# print("[INFO] loading model...")
# model = load_model('models/buoy_model/keras_model.h5')
# Labels for this model are shown below
# 0 Direct Sun
# 1 Stormy Weather
# 2 Interesting
# 3 Object Detected
# 4 Sunset
# 5 Clouds
# 6 Night



# def check_image(image):
#     global model # load the model from disk
#     # load the image and show it
#     image = cv2.imread('images/panels/44020/2022_11_6_10_54/panel_1.jpg')
#     cv2.imshow("image", image)
#     cv2.waitKey(0) # show image
#     # feed the image to the model and get the prediction
#     prediction = model.predict(image) # predict the image
#     print(prediction) # print the prediction

#     # Return the predicted class of the image (0-6)
#     return np.argmax(prediction) # return the predicted class i.e. the type of scene in the image 'Stormy Weather' or 'Direct Sun' etc.
#     # We return the np.argmax of the prediction because the prediction is a list of probabilities for each class. The class with the highest probability is the predicted class.



# # Do a test run of the check_image function to see if it works on a single image
# test_image = 'images/panels/42002/2022_11_6_18_30/panel_3.png'
# validation_class = 'Sunset' # we know that this image is a sunset image, so we can use this to test the model
# # we can use this to test the model

# # Testing Model:
# print("Testing Keras Model")
# assert(os.path.exists(test_image)) # check if file exists
# pred = check_image(test_image) # get the predicted class of the image
# if pred != 4:
#     print("The image is not a sunset image")
#     print(pred) # print the predicted class
# else:
#     print("The image is a sunset image")



















def buoy_links():
    global ids
    links = ["https://www.ndbc.noaa.gov/buoycam.php?station=42001","https://www.ndbc.noaa.gov/buoycam.php?station=46059","https://www.ndbc.noaa.gov/buoycam.php?station=41044","https://www.ndbc.noaa.gov/buoycam.php?station=46071","https://www.ndbc.noaa.gov/buoycam.php?station=42002","https://www.ndbc.noaa.gov/buoycam.php?station=46072","https://www.ndbc.noaa.gov/buoycam.php?station=46066","https://www.ndbc.noaa.gov/buoycam.php?station=41046","https://www.ndbc.noaa.gov/buoycam.php?station=46088","https://www.ndbc.noaa.gov/buoycam.php?station=44066","https://www.ndbc.noaa.gov/buoycam.php?station=46089","https://www.ndbc.noaa.gov/buoycam.php?station=41043","https://www.ndbc.noaa.gov/buoycam.php?station=42012","https://www.ndbc.noaa.gov/buoycam.php?station=42039","https://www.ndbc.noaa.gov/buoycam.php?station=46012","https://www.ndbc.noaa.gov/buoycam.php?station=46011","https://www.ndbc.noaa.gov/buoycam.php?station=42060","https://www.ndbc.noaa.gov/buoycam.php?station=41009","https://www.ndbc.noaa.gov/buoycam.php?station=46028","https://www.ndbc.noaa.gov/buoycam.php?station=44011","https://www.ndbc.noaa.gov/buoycam.php?station=41008","https://www.ndbc.noaa.gov/buoycam.php?station=46015","https://www.ndbc.noaa.gov/buoycam.php?station=42059","https://www.ndbc.noaa.gov/buoycam.php?station=44013","https://www.ndbc.noaa.gov/buoycam.php?station=44007","https://www.ndbc.noaa.gov/buoycam.php?station=46002","https://www.ndbc.noaa.gov/buoycam.php?station=51003","https://www.ndbc.noaa.gov/buoycam.php?station=46027","https://www.ndbc.noaa.gov/buoycam.php?station=46026","https://www.ndbc.noaa.gov/buoycam.php?station=51002","https://www.ndbc.noaa.gov/buoycam.php?station=51000","https://www.ndbc.noaa.gov/buoycam.php?station=42040","https://www.ndbc.noaa.gov/buoycam.php?station=44020","https://www.ndbc.noaa.gov/buoycam.php?station=46025","https://www.ndbc.noaa.gov/buoycam.php?station=41010","https://www.ndbc.noaa.gov/buoycam.php?station=41004","https://www.ndbc.noaa.gov/buoycam.php?station=51001","https://www.ndbc.noaa.gov/buoycam.php?station=44025","https://www.ndbc.noaa.gov/buoycam.php?station=41001","https://www.ndbc.noaa.gov/buoycam.php?station=51004","https://www.ndbc.noaa.gov/buoycam.php?station=44027","https://www.ndbc.noaa.gov/buoycam.php?station=41002","https://www.ndbc.noaa.gov/buoycam.php?station=42020","https://www.ndbc.noaa.gov/buoycam.php?station=46078","https://www.ndbc.noaa.gov/buoycam.php?station=46087","https://www.ndbc.noaa.gov/buoycam.php?station=51101","https://www.ndbc.noaa.gov/buoycam.php?station=46086","https://www.ndbc.noaa.gov/buoycam.php?station=45002","https://www.ndbc.noaa.gov/buoycam.php?station=46053","https://www.ndbc.noaa.gov/buoycam.php?station=46047","https://www.ndbc.noaa.gov/buoycam.php?station=46084","https://www.ndbc.noaa.gov/buoycam.php?station=46085","https://www.ndbc.noaa.gov/buoycam.php?station=45003","https://www.ndbc.noaa.gov/buoycam.php?station=45007","https://www.ndbc.noaa.gov/buoycam.php?station=46042","https://www.ndbc.noaa.gov/buoycam.php?station=45012","https://www.ndbc.noaa.gov/buoycam.php?station=42019","https://www.ndbc.noaa.gov/buoycam.php?station=46069","https://www.ndbc.noaa.gov/buoycam.php?station=46054","https://www.ndbc.noaa.gov/buoycam.php?station=41049","https://www.ndbc.noaa.gov/buoycam.php?station=45005"]

    #note: undo this to go with the established buoy list
    # links_2 = create_buoy_links(ids)

    # # append the links_2 to links if they are not already in links
    # for link in links_2:
    #     if link not in links:
    #         links.append(link)

    return links


# ids from cam_backend

def create_buoy_links(ids):
    # for each id in ids, create a link
    links = []
    for id in ids:
        link = "https://www.ndbc.noaa.gov/buoycam.php?station=" + id
        links.append(link)
    return links

# Notes:
# Buoy 42002 Has good sunsets

def check_buoy_image_ifwhite(image):
    """
    check_buoy_image_ifwhite checks if the image is white

    This function checks if the image is white. If the image is white, then the image is not valid and should be deleted.

    :param image: the image to check
    :type image: result of requests library get request for image url
    :return: True if the image is white, False if the image is not white
    :rtype: bool
    """
    # some buoys do not have a camera or the camera is not working. In these cases the image is white with only the text "No Image Available"
    # determine if the image is white
    # get the image in numpy array format
    img = np.asarray(bytearray(image.content), dtype="uint8")
    # convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # check if the image is white
    if np.mean(gray) > 250:
        return True
    else:
        return False

def ocean_stitching(imagePaths, pano_path):
    images = []
    # loop over the image paths, load each one, and add them to our
    # images to stitch list
    # open each image with cv2 and append to images list
    for imagePath in imagePaths:
        try:
            # add the full path to the image
            # '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/images/panels/44020/2022_11_6_10_54/panel_1.jpg'
            # read the image
            full_path =  '/Volumes/Backups of Grahams IMAC/PythonProjects/PySeas_Master_Folder/PySeas/' + imagePath
            imagePath = full_path # this is the full path to the image
            assert(os.path.exists(imagePath)) # check if file exists
            image = cv2.imread(imagePath, cv2.IMREAD_COLOR) # read image
            #cv2.imshow("image", image)
            #cv2.waitKey(0) # show image
            images.append(image) # append to list
            if image is None:
                print("Error loading image: " + imagePath)
        except:
            print("Error reading image: " + imagePath)
            continue
    # getting error: OpenCV(4.6.0) /Users/runner/work/opencv-python/opencv-python/opencv/modules/imgproc/src/resize.cpp:4052: error: (-215:Assertion failed) !ssize.empty() in function 'resize'
    # how to fix this:
    # The error is saying that the image is empty. This is because you are trying to read an image that doesn't exist. Check the path to the image and make sure it is correct.

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    (status, stitched) = stitcher.stitch(images)
    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # write the output stitched image to disk
        cv2.imwrite(pano_path, stitched) # save the stitched image
        # display the output stitched image to our screen
        #!cv2.imshow("Stitched", stitched)
        #!cv2.waitKey(0)
    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))


def refine_view(stitched_image):
    stitched = cv2.copyMakeBorder(stitched, 10, 10, 10, 10,
			cv2.BORDER_CONSTANT, (0, 0, 0))
    # convert the stitched image to grayscale and threshold it
    # such that all pixels greater than zero are set to 255
    # (foreground) while all others remain 0 (background)
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    # find all external contours in the threshold image then find
    # the *largest* contour which will be the contour/outline of
    # the stitched image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # allocate memory for the mask which will contain the
    # rectangular bounding box of the stitched image region
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)


# given a list of images, create a pano image
def create_pano_image(image_list, pano_path):
    # create a pano image from the images in image_list
    # image_list is a list of image paths
    # pano_path is the path to save the pano image
    # create the pano image
    ocean_stitching(image_list, pano_path)
    # refine the pano image
    #refine_view(pano_path)



def chunk_images(buoy_id,foldername):
    """
    chunk_images takes a folder of images and splits them into sets of 6 images

    _extended_summary_

    :param buoy_id: The id of the buoy
    :type buoy_id: int
    :param foldername: The name of the folder containing the images
    :type foldername: str
    # """
    # #buoy_id = str(foldername)
    # # There are many images in the folder. We want to parse these in sets of six.
    # # images/panels/41002/2022_11_5_15_44_panel_1.png is an example of the naming convention for the images.
    # # the first number (folder under panels) is the buoy id, then the filename begins with the year, the second number is the month, the third number is the day, the fourth number is the hour, the fifth number is the minute, and the sixth number is the panel number.
    # # we want to select the images that have the same buoy id, year, month, day, and hour AND end in the panel numbers: 1,2,3,4,5,6.

    # # for the example file above 2022_11_5_15_44_panel_1.png
    # # we want to put it into a subfolder named 2022_11_5_15 and put all the images that match the naming convention into that folder. There should only be 6.
    # # then return the list of lists of images that are in the subfolders.

    # # if the name of the

    # # foldername[0] --> '2022_11_5_15_44.jpg_panel_1.png'
    # # this is the kind of dirty names we have to clean.

    # # we want to get the first 4 numbers and the last number.
    # # we want to put the images into a folder named 2022_11_5_15
    # # we want to return a list of lists of images that are in the subfolders.

    # # we want to get the first 4 numbers and the last number.
    # # if type(foldername) == list:
    # #     foldername = foldername[0]
    # # else:
    # #     foldername = foldername
    # for file_name in tqdm(foldername):

    #     first_four = file_name.split('_')[0:4] # i.e. ['2022', '11', '5', '15']
    #     last_num = file_name.split('_')[-1] # i.e. '1.png'
    #     last_num = last_num.split('.')[0] # i.e. '1'


    #     # we want to put the images into a folder named 2022_11_5_15
    #     new_folder_name = '_'.join(first_four) # i.e. '2022_11_5_15'

    #     # the photos that belong with the new folder name (2022_11_5_15) contain the foldername's numbers.
    #     # Sort the images in the folder by the date they were added to the folder.
    #     #sorted_foldername = sorted(foldername, key=lambda x: os.path.getmtime(x)) # sort the images by the date they were added to the folder.

    #     # find all the files with the first 4 numbers (2022_11_5_15) and .jpg only once in the name
    #     files_with_last_four = [x for x in foldername if new_folder_name in x and 'jpg_panel' not in x]
    #     # find all the files with the last number (1)
    #     files_with_last_num = [x for x in files_with_last_four if str('panel_')+last_num in x]


    #     # are the images in the foldername in the correct order?
    #     # we want to check if the images are in the correct order.
    #     # They should now have the word panel 1, in the first image, panel 2 in the second image, etc. down to panel 6 and then the next image should be panel 1 again.
    #     # now we can just take each six images and put them into a folder.

    #     # logic
    #     # as long as the image contains "panel_n" where n is the next number in the order of the panels we add the image to the folder name that matches the first four numbers in their name.

    #      # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.


    #     # make a folder in the panels folder with the name of the first four numbers.
    #     # if the folder doesn't exist, make it.
    #     print('new_folder_name',new_folder_name)
    #     try:
    #         os.mkdir(f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #     except Exception as e:
    #         print(e)
    #         pass

    #     # go through the images in the foldername and put them into the correct folder.
    #     # we want to put the images into a folder named 2022_11_5_15 or whatever the first four numbers are.
    #     for image in files_with_last_num:
    #         # move the image into the directory f'images/panels/{buoy_id}/{new_folder_name}'
    #         # full path of the image
    #         full_path = f'images/panels/{buoy_id}/{image}'
    #         shutil.move(full_path, f'images/panels/{buoy_id}/batches/{new_folder_name}')
    #         pass



    #     # now we want to delete the images that are in the foldername but not in the subfolder.
    #     # we want to delete the images that are in the foldername but not in the subfolder.

    #     # get the list of images in the foldername
    #     images_in_foldername = [x for x in foldername if buoy_id in x]
    #     # get the list of images in the subfolder
    #     images_in_subfolder = [x for x in os.listdir(f'images/panels/{buoy_id}/batches/{new_folder_name}') if buoy_id in x]
    #     # get the list of images that are in the foldername but not in the subfolder
    #     #!images_to_delete = [x for x in images_in_foldername if x not in images_in_subfolder]
    #     # delete the images that are in the foldername but not in the subfolder
    #     #for image in images_to_delete:
    #     #    os.remove(f'images/panels/{buoy_id}/{image}')





    #     # img1 = os.path.join('images','panels',str(buoy_id),file_name) # the image we want to move
    #     # img2 = os.path.join('images','batches',str(buoy_id),new_folder_name) # the folder we want to move the image to.
    #     # # i.e. move the image from images/panels/41002/2022_11_5_15_44_panel_1.png to images/panels/41002/2022_11_5_15/2022_11_5_15_44_panel_1.png
    #     # shutil.copy(img1, img2) # copy the image to the folder.
    pass


# 51101,51000,51003,51002,51004 - Hawaii
# 46006 - Amazing Sunset
# 46089 - Tillamook Oregon.

from difPy import dif
cam_urls = buoy_links() # get the links to the cameras
all_buoy_urls = create_buoy_links(ids)
stitch_switch = False # make false if you don't want to stitch the images.

# open the blacklist file

from ratelimit import limits, sleep_and_retry

# @limits(calls=1, period=4) # limit the number of calls to the function to 1 every 4 seconds.
@sleep_and_retry
def pull_data(cam_url, buoy_id, now):
    img = requests.get(cam_url) # get the image
    if img.status_code == 200:
        return img
    else:
        print("status code", img.status_code, "for buoy", buoy_id)
    return img




### Testing the code
# detect red in an image
# load the image
image = cv2.imread('images/buoys/46072/2022_11_5_19_27.jpg')
image_path = 'images/buoys/46072/2022_11_5_19_27.jpg'
#* Test 2.
# result = finding_red_version_two(image) # find the red in the image
# print(result)

#* Test 3. hsv and npwhere
output_img = finding_red_version_three(image_path) # find the red in the image
print(output_img)
#print(output_hsv)


#Notes to self: remove functions for tests up to this point.
#* Test 4. Just red percent
#& Successful!
percent_red = detect_red_v4(image_path)
print(percent_red)


# test with the function to see if it detects red.

detect_red_v4(image_path)# returns True if it detects red, False if it doesn't.










do_loop = True

if do_loop:
    pass
else:
    exit() # exit the program if do_loop is False.


while True:
    try:
        # turn on at 4 am CST and turn off at 11 pm CST
        if datetime.datetime.now().hour < 3 or datetime.datetime.now().hour > 24: # if it is before 3 am or after 12 am
            # wait to turn on until 4 am CST
            # keep the computer awake
            print('The computer is sleeping')
            time.sleep(240) # sleep for 4 minutes
            continue

        # updated blacklist file
        blacklist = open('data/blacklisted_buoy_ids.csv').read().splitlines() # get the list of buoy ids that are blacklisted.
        # parse blacklist to remove extra ' and " characters
        blacklist = [x.replace('"','') for x in blacklist]
        blacklist = [x.replace("'",'') for x in blacklist]
        # create a blacklist list of strings from blacklist
        blacklist = [str(x) for x in blacklist][0].replace(' ','').split(',')




        # # if the time is between 4 am and 11 am pacific time, then your wait_period is 100 seconds
        # if datetime.datetime.now().hour >= 4 and datetime.datetime.now().hour < 11:
        #     wait_period = 100
        # # if the time is between 11 am and 11 pm pacific time, then your wait_period is 600 seconds
        # if datetime.datetime.now().hour >= 11 and datetime.datetime.now().hour < 13:
        #     wait_period = 600 # 10 minutes
        # wait for 15 minutes
        wait_period = 600 # 10 minutes
        start_time = datetime.datetime.now() # use this to calculate the next time to download images (every ten minutes)
        #!print('Starting the download loop at {}'.format(start_time))
        # print('I can still see things! Downloading images...')
        chunk_size = 30 # download 30 images at a time then pause for 10 seconds.
        chunk_size_current = 0 # the current number of images downloaded in the current chunk.
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search('station=(.*)', cam_url).group(1)
            if buoy_id in blacklist: # if the buoy id is in the blacklist, then skip it.
                continue # skip this buoy id
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists('images/buoys/{}'.format(buoy_id)):
                os.makedirs('images/buoys/{}'.format(buoy_id))
                ##logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            ##logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if 'images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute) not in os.listdir('images/buoys/{}'.format(buoy_id)): # if the image has not already been downloaded
                time.sleep(0.15) # wait 0.25 seconds to avoid getting blocked by the server
                if chunk_size_current < chunk_size: # if we have not downloaded 30 images yet
                    chunk_size_current += 1 # add one to the chunk size
                else:
                    time.sleep(15) # wait 15 seconds
                    chunk_size_current = 0 # reset the chunk size

                wait = True # set the wait variable to true
                while wait: # while we are waiting
                    try: # try to get the image
                        img = pull_data(cam_url, buoy_id, now) # download the image
                        wait = False
                    except Exception as e:
                        # print(e)
                        wait = True
                        time.sleep(1)
                        continue

                # check if the image is white


                # Print the name of the image we are downloading
                print('Downloading image: {}'.format('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)))
                # save the image
                with open('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), 'wb+') as f:
                    f.write(img.content) # write the image to the file
                # check if the image is daytime or nighttime
                # ##logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                #if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                    # then we will delete the image
                    #*print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                    #*os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass

                #^ check image to see if it is just a white screen or not. If it is then we want to add this buoy id to the blacklist so that we don't download images from it anymore.

            else:
                print('Image already exists: {}'.format('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)))
                pass # if the image already exists, don't download it again

        ##logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir('images/buoys') # get the list of buoy ids by their directory names

        # sample a random 20 extras from the
        print('Creating panels...')
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            #print(f'Paneling images for buoy {buoy_id}')
            if buoy_id != '.DS_Store' and '.' not in buoy_id: # if the buoy id is not a hidden file
                images = os.listdir('images/buoys/{}'.format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                ##logging.info("Saving panels for buoy {}".format(buoy_id))
                for image in images:
                    # print(f'    Paneling image {image}')
                    # if the image is not None
                    if image == '.DS_Store' or image != 'None':
                        continue
                    # If the panels directory for the buoy doesn't exist, create it.
                    if not os.path.exists('images/panels/{}'.format(buoy_id)):
                        os.makedirs('images/panels/{}'.format(buoy_id))
                    if 'images/buoys/{}/{}'.format(buoy_id, image) in os.listdir('images/panels/{}'.format(buoy_id)) and image == '.DS_Store' and buoy_id != '.DS_Store':
                        print('This image has already been used to create panels. Or it is a hidden file.')
                    else:
                        # get the panels
                        panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(buoy_id, 'images/buoys/{}/{}'.format(buoy_id, image))

                    #print('Processing image: {}'.format(image))

                    ##logging.info("Saved panels for buoy {}".format(buoy_id))
                    # print('Saving panels...')
                    # save the panels to the images/panels directory
                    # now, stitch these images together (correcting for the misalignment of the cameras) and save the result to the images/panoramas directory
                    # print('Stitching panels...')
                    # stitch the panels together using stitched_panoramas

                    # try:
                    #     stitched = stitched_panoramas(panel_1, panel_2, panel_3, panel_4, panel_5, panel_6)
                    #     # save the stitched panorama to the images/panoramas directory
                    #     # print('Saving stitched panorama...')
                    #     if not os.path.exists('images/panoramas/{}'.format(buoy_id)):
                    #         os.makedirs('images/panoramas/{}'.format(buoy_id))
                    #     # print('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                    #     cv2.imwrite('images/panoramas/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), stitched)
                    #     # move the image to the images/panels directory
                    #     os.rename('images/buoys/{}/{}'.format(buoy_id, image), 'images/panels/{}/{}'.format(buoy_id, image))
                    #     ##logging.info("Moved image to panels directory for buoy {}".format(buoy_id))
                    # except Exception as e:
                    #     print(e)
                    #     pass


        # Stage 4: save buoy_update_rates_dict to a csv file
        buoy_update_rates_dict_df = pd.DataFrame.from_dict(buoy_update_rates_dict, orient='index')
        buoy_update_rates_dict_df.to_csv('data/buoy_update_rates_dict.csv')

        # Stage 5: Using DifPy, find any images that are similar 'normal' to white_blank.jpg and delete them.
        # parse the buoy folders and their images


        try:
            buoy_folders = os.listdir('images/buoys')
            for buoy_folder in buoy_folders:
                if buoy_folder != '.DS_Store':
                    images = os.listdir('images/buoys/{}'.format(buoy_folder))
                    for image in images:
                        if image != '.DS_Store' and image != 'None':
                            # get the image path
                            image_path = 'images/buoys/{}/{}'.format(buoy_folder, image)
                            # get the image
                            image = cv2.imread(image_path)
                            white_image = cv2.imread('images/white_blank.jpg')
                            #we need these images to be the same size, so we will resize the white image to the size of the image
                            white_image = cv2.resize(white_image, (image.shape[1], image.shape[0]))
                            # are they ndarrays?
                            #print(type(image))
                            #print(type(white_image))

                            #get the difference between the image and the white_blank.jpg image
                            #calculate the difference between pixel values of the image and a pure white image using numpy
                            diff = np.sum(np.abs(image - white_image)) # get the sum of the absolute difference between the two images
                            #if the difference is less than 1000, then we will delete the image
                            if diff < 1000:
                                print('Deleting image: {}'.format(image_path))
                                #move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                if not os.path.exists('images/deleted_images'):
                                   os.makedirs('images/deleted_images')
                                os.rename(image_path, 'images/deleted_images/{}_{}'.format(image_path.split('/')[-1].split('.')[0], buoy_folder))
                                os.remove(image_path)
                            # get the difference score from the difference image
                            #difference_score = dif.get_difference_score()
                            # if the difference score is less than 0.1, then delete the image

                            # if the difference is less than 0.1, then delete the image
                            # if difference_score < 0.1:
                            #     os.remove(image_path)
                            #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
        except Exception as e:
            print("Error with White Image Detection: {}".format(e))
            pass

        if exper_1:
            # run DifPy on the images in the images/panels directory
            try:
                buoy_folders = os.listdir('images/panels')
                for buoy_folder in buoy_folders:
                    if buoy_folder != '.DS_Store':
                        images = os.listdir('images/panels/{}'.format(buoy_folder))
                        for image in images:
                            if image != '.DS_Store':
                                # get the image path
                                image_path = 'images/panels/{}/{}'.format(buoy_folder, image)
                                # get the image
                                image = cv2.imread(image_path)
                                white_image = cv2.imread('images/white_blank.jpg')
                                # we need these images to be the same size, so we will resize the white image to the size of the image
                                white_image = cv2.resize(white_image, (image.shape[1], image.shape[0]))
                                # are they ndarrays?
                                # print(type(image))
                                # print(type(white_image))

                                # get the difference between the image and the white_blank.jpg image
                                # calculate the difference between pixel values of the image and a pure white image using numpy
                                diff = np.sum(np.abs(image - white_image)) # get the sum of the absolute difference between the two images
                                # if the difference is less than 1000, then we will delete the image
                                if diff < 1000:
                                    print('Deleting image: {}'.format(image_path))
                                    # move the images instead of literally deleting them to a folder called 'deleted_images' in the images directory
                                    if not os.path.exists('images/deleted_images'):
                                        os.makedirs('images/deleted_images')
                                    os.rename(image_path, 'images/deleted_images/{}_{}'.format(image_path.split('/')[-1].split('.')[0], buoy_folder))
                                    # os.remove(image_path)
                                # dif.
                                # # get the difference score from the difference image
                                # difference_score = dif.get_difference_score()
                                # # if the difference score is less than 0.1, then delete the image

                                # # if the difference is less than 0.1, then delete the image
                                # if difference_score < 0.1:
                                #     os.remove(image_path)
                                #     print('Deleted image {} because it was too similar to white_blank.jpg'.format(image_path))
            except Exception as e:
                print("Error with White Image Detection: {}".format(e))
                pass


        # Remove duplicate images (preferably before paneling but for now after)
        if duplicate_removal_flag == True:

            for folder in os.listdir('images/buoys'):
                if folder == '.DS_Store':
                    continue
                # get the list of images in the folder
                # sort the images by date
                # make folder_path variable from relative path
                folder_path = 'images/buoys/{}'.format(folder)
                search = dif(folder_path, similarity='high', show_output=False, show_progress=True) # returns a list of lists of similar images
                # for each list of similar images, move all but the first image to the deleted_images folder
                file_results_dict = search.result # get the list of file names
                # {20220824212437767808 : {"filename" : "image1.jpg",
                #                         "location" : "C:/Path/to/Image/image1.jpg"},
                #                         "duplicates" : ["C:/Path/to/Image/duplicate_image1.jpg",
                #                                         "C:/Path/to/Image/duplicate_image2.jpg"]},
                # This is the format of the dictionary returned by the dif.search() method
                # I want to the filename, location, and duplicates
                # I want to move the duplicates to the deleted_images folder





                # make the deleted_images folder if it doesn't exist
                if not os.path.exists('images/deleted_images'):
                    os.makedirs('images/deleted_images')

                # counter should be how many files are in the deleted folder before we start
                counter = len(os.listdir('images/deleted_images'))
                # move the duplicates to the deleted_images folder
                for key in file_results_dict: # iterate through the keys in the dictionary
                    # get the duplicates
                    value = file_results_dict[key]
                    duplicates = value['duplicates']
                    for duplicate in duplicates:
                        try:
                            # move the duplicate to the deleted_images folder
                            # os.rename(duplicate, 'images/deleted_images/{}_{}'.format(counter,duplicate.split('/')[-1]))
                            # remove the duplicate
                            # full dupe path
                            #full_dupe_path = 'images/buoys/{}/{}'.format(folder, duplicate.split('/')[-1])

                            # first add "duplicate_" to the beginning of the file name
                            new_name = duplicate.split('/')[-1] # get the file name
                            new_name = 'duplicate_{}'.format(new_name) # add duplicate_ to the beginning of the file name
                            # then rename it in the same directory as the original
                            os.rename(duplicate, 'images/buoys/{}/{}'.format(folder, new_name))
                            # then move the file to the deleted_images folder
                            print('Renamed {} to {}'.format(duplicate, new_name))
                            # os.rename(duplicate, str(duplicate).replace('images/buoys', 'images/deleted_images'))
                            counter += 1
                        except Exception as e:
                            print("Error moving duplicate image: {}".format(e))
                            pass

        ignoring_panel_optimimal = True # note: this is a temporary fix to the problem of the panel images not being generated
        # final step: make sure that all the previous buoy images have been panelled and saved to the images/panels directory
        for folder in tqdm(os.listdir('images/buoys')):
            #print('Checking if all images have been panelled for buoy {}'.format(folder))
            try:
                if folder == '.DS_Store':
                    continue
                images = os.listdir('images/buoys/{}'.format(folder))
                # if the folder is not in the images/panels directory, then we need to panel the images
                # if not os.path.exists('images/panels/{}'.format(folder)):
                if ignoring_panel_optimimal:
                    #print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                    # panelling the images
                    try:
                        os.mkdir('images/panels/{}'.format(folder))
                        print('made directory for buoy {}'.format(folder) + ' in images/panels')
                    except:
                        pass
                    batch_id = 1
                    for image in images:
                        # make a folder for the batch that has the same name as the image without the extension
                        try:
                            i_name = image[:-4]
                            directory_save_path = f'images/panels/{folder}/{i_name}' # make the directory path
                            os.mkdir(directory_save_path)
                        except FileExistsError:
                            pass
                        # get the panels
                        # if the folder is not empty skip it
                        if len(os.listdir(directory_save_path)) > 0:
                            continue
                        try:
                            if image == '.DS_Store':
                                continue
                            # get the panels and save them to directory_save_path
                            panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(folder, 'images/buoys/{}/{}'.format(folder, image))
                            # save the panels to the directory_save_path
                            # panel_1.save(f'{directory_save_path}/panel_1.jpg')
                            # panel_2.save(f'{directory_save_path}/panel_2.jpg')
                            # panel_3.save(f'{directory_save_path}/panel_3.jpg')
                            # panel_4.save(f'{directory_save_path}/panel_4.jpg')
                            # panel_5.save(f'{directory_save_path}/panel_5.jpg')
                            # panel_6.save(f'{directory_save_path}/panel_6.jpg')
                            ##logging.info('Saved panels for image: {}'.format(image))

                        except:
                            ##logging.ERROR('Error while saving panels for image: {}'.format(image))
                            # print('Could not create panels for image: {}'.format(image))
                            continue

                        #note: trying to add in the vincent code here
                        # stitch the images together
                        if stitch_switch:
                            files_to_stitch = [f'{directory_save_path}/panel_1.jpg', f'{directory_save_path}/panel_2.jpg', f'{directory_save_path}/panel_3.jpg', f'{directory_save_path}/panel_4.jpg', f'{directory_save_path}/panel_5.jpg', f'{directory_save_path}/panel_6.jpg'] # list of files to stitch

                            # Stitch the images together with OpenCV and save the stitched image to the panoramas directory
                            print('Stitching images...')
                            try:
                                ocean_stitching(files_to_stitch, f'images/panoramas/{folder}/{i_name}.jpg') # stitch the images together and save the stitched image to the panoramas directory
                            except Exception as f:
                                print(f)
                                print('Could not stitch images for image: {}'.format(image))
                            # > Overload resolution failed:
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Can't parse 'images'. Sequence item with index 0 has a wrong type
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # >  - Stitcher.stitch() missing required argument 'masks' (pos 2)
                            # fix: https://stackoverflow.com/questions/6380057/python-cv2-error-215-overload-resolution-failed











                        # try:
                        #     print('Stitching images for image set {}'.format(files_to_stitch))
                        #     panorama = vincent.make_panorama(images)
                        #     #remove the .jpg from the image name
                        #     image_name = image[:-4] # remove the .jpg from the image name

                        #     cv2.imwrite(f'images/panoramas/{folder}_{image_name}_panorama.png', panorama)
                        # except Exception as e:
                        #     print(e)
                        #     print('Could not stitch images together for buoy: {}'.format(folder))
                        #     continue


                        batch_id += 1
            except Exception as e:
                print(e)
                #print('Could not create panels for buoy: {}'.format(folder))
                #print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                #print('line 139') # line 139
                pass
        # for each folder in the images/panels folder, stitch the images together and save them to the images/panoramas folder with the same name as the folder + panorama.png

        #//: the for loop below does not account for the fact that there are multiple captures with 6 frames per capture. This means that the images will be stitched together incorrectly. This is a problem that needs to be fixed. Find a way to select only the sets of 6 images that go together to stitch together.

        print('stage 5 complete')
        # Stage 6: Create the buoy dataframes
            # if it has been ten minutes since the last time the data was fetched, fetch the data again
        if time.time() - last_time_fetched > 600 or first_run:
            latest_data = get_latest_data() # get the latest data from the RSS feed (updates every 10 minutes)
            # save the master dataframe to a csv file
            run_date = time.strftime("%Y%m%d_%H%M%S")
            latest_data.to_csv(f'data/rss/rss_buoy_data_{run_date}.csv', index=False)
            print('Done with this run')
            time_last_fetched_rss = time.time() # get the time of the last fetch
        print('stage 6 complete')

        #* ======= show the last buoy image captured in this run
        try:
            display_last_image(list_of_buoys) # display the last image captured in this run
        except:
            pass
        #* ====== End Show Buoy Image Snippet


        print('Running White Elimiination and populating the Tapestry')
        deal_with_white_images_and_populate_tapestry() # run the white elimination and populate the tapestry




        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # wait until the time elapsed is 15 minutes from the start of the loop
        print("Waiting for the remainder of the minutes...")

        if verbose_wait:
            # wait_period = 100 # was 900 (15 minutes)
            for i in tqdm(range(wait_period - time_elapsed.seconds)):
                time.sleep(1)
            iteration_counter += 1
        else:
            print('Waiting for the remaining {} seconds'.format(wait_period - time_elapsed.seconds))
            time.sleep(wait_period - time_elapsed.seconds)
            iteration_counter += 1

    except Exception as e:
        print(e)
        print('Error occurred.')
        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        #* wait till the ten minute mark is reached.
        for i in tqdm(range(wait_period - time_elapsed.seconds)):
            time.sleep(1)
        iteration_counter += 1








    # # now we have all the images in the subfolders
    # # we want to return a list of lists of images that are in the subfolders

    # list_of_lists = []
    # for folder in os.listdir(f'images/batches/{buoy_id}'):
    #     try:
    #         # Get the list of images in the folder, and append it to the list of lists (these are the relative paths to the images)
    #         list_of_lists.append(os.listdir(f'images/batches/{buoy_id}/{folder}')) # this will be a list of 6 images
    #     except NotADirectoryError:
    #         # there is a file in the folder
    #         continue
    #     except FileNotFoundError:
    #         # there is a missing folder
    #         print(f'FileNotFoundError: {folder}')
    #         continue
    #     except Exception as e:
    #         print(e)
    #         continue
    # return list_of_lists # return a list of sets of images that have the same buoy id, year, month, day, hour, and panel numbers 1,2,3,4,5,6



# Next Steps

"""

The next steps are to:
Use the Keras model saved in the models folder to predict the content of each panel image as it is created, and save the predictions to a csv file.
These predictions will determine which buoys the model will follow closest and which buoys it will eventually ignore. This will be done by finding which of the buoy cameras record the most interesting content (i.e. the most interesting content will be the content that the model predicts as being the most interesting).



            # daytime images always have higher values than 10 for all three channels
            # values less than 10 are usually night
            # skip the image if it is night
            #if orange_value < 10 and red_value < 10: # if the image is night
            #    continue
            # if the values are all higher than 250 then it is a white image and we want to remove it
            #if orange_value > 250 and red_value > 250:
            #    os.remove(file)
            #    print("Removed white image")
            #    continue
            # interesting photos have high red values
            #if red_value > 147 or (orange_value > 147 and #red_value > 100): # if the image is interesting
            #    add_list.append(file)
            # if the image was not taken in the last x hours, skip it

"""