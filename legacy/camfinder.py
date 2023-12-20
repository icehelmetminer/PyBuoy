from cam_finder_raw import *


print('Starting the download loop')
last_time_fetched = time.time() # get the current time
first_run = True # set a flag to indicate that this is the first run of the loop (for the first run, we will download rss feeds for all the buoys)


while True:
    try:
        # turn on at 4 am CST and turn off at 11 pm CST
        if datetime.datetime.now().hour < 4 or datetime.datetime.now().hour > 22: # if it is before 4 am or after 11 pm
            # wait to turn on until 4 am CST
            # keep the computer awake
            print('The computer is sleeping')
            time.sleep(240) # sleep for 4 minutes
            continue
        # # if the time is between 4 am and 11 am pacific time, then your wait_period is 100 seconds
        # if datetime.datetime.now().hour >= 4 and datetime.datetime.now().hour < 11:
        #     wait_period = 100
        # # if the time is between 11 am and 11 pm pacific time, then your wait_period is 600 seconds
        # if datetime.datetime.now().hour >= 11 and datetime.datetime.now().hour < 13:
        #     wait_period = 600 # 10 minutes
        # wait for 15 minutes
        wait_period = 600 # 10 minutes
        start_time = datetime.datetime.now() # use this to calculate the next time to download images (every ten minutes)
        logging.INFO("Beginning to go through images (line 24) at {}".format(start_time))
        # print('I can still see things! Downloading images...')
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search('station=(.*)', cam_url).group(1)
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists('images/buoys/{}'.format(buoy_id)):
                os.makedirs('images/buoys/{}'.format(buoy_id))
                logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if 'images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute) not in os.listdir('images/buoys/{}'.format(buoy_id)): # if the image has not already been downloaded
                time.sleep(0.25) # wait 0.25 seconds to avoid getting blocked by the server
                img = requests.get(cam_url) # get the image
                # save the image
                with open('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), 'wb+') as f:
                    f.write(img.content) # write the image to the file
                # check if the image is daytime or nighttime
                logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                #if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                    # then we will delete the image
                    #*print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                    #*os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass
            else:
                pass # if the image already exists, don't download it again


        print("Paneling images...")
        logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir('images/buoys') # get the list of buoy ids by their directory names
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            if buoy_id != '.DS_Store':
                images = os.listdir('images/buoys/{}'.format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                logging.info("Saving panels for buoy {}".format(buoy_id))
                for image in images:
                    # if the image is not None
                    if image == '.DS_Store' or image != 'None':
                        continue
                    # If the panels directory for the buoy doesn't exist, create it.
                    if not os.path.exists('images/panels/{}'.format(buoy_id)):
                        os.makedirs('images/panels/{}'.format(buoy_id))
                    if 'images/buoys/{}/{}'.format(buoy_id, image) in os.listdir('images/panels/{}'.format(buoy_id)):
                        print('This image has already been used to create panels.')
                        continue
                    if image == '.DS_Store' and buoy_id != '.DS_Store':
                        continue # skip the .DS_Store file
                    #print('Processing image: {}'.format(image))

                    # get the panels
                    panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(buoy_id, 'images/buoys/{}/{}'.format(buoy_id, image))
                    logging.info("Saved panels for buoy {}".format(buoy_id))
                    print('Saving panels...')
                    # save the panels to the images/panels directory


        # Stage 4: save buoy_update_rates_dict to a csv file
        buoy_update_rates_dict_df = pd.DataFrame.from_dict(buoy_update_rates_dict, orient='index')
        buoy_update_rates_dict_df.to_csv('data/buoy_update_rates_dict.csv')

        # Stage 5: Remove any duplicate images in the images/buoys directory with DifPy



        # Remove duplicate images (preferably before paneling but for now after)
        for folder in os.listdir('images/buoys'):
            if folder == '.DS_Store':
                continue
            # get the list of images in the folder
            # sort the images by date
            # make folder_path variable from relative path
            folder_path = 'images/buoys/{}'.format(folder)
            search = dif(folder_path, similarity='high', show_output=False, show_progress=True, silent_del=True, delete=True)

        # final step: make sure that all the previous buoy images have been panelled and saved to the images/panels directory
        for folder in tqdm(os.listdir('images/buoys')):
            if folder == '.DS_Store':
                continue
            if not os.path.exists('images/panels/{}'.format(folder)):
                print('The images in the images/buoys/{} directory have not been panelled yet.'.format(folder))
                # panelling the images
                os.mkdir('images/panels/{}'.format(folder))
                images = os.listdir('images/buoys/{}'.format(folder))
                for image in images:
                    try:
                        if image == '.DS_Store':
                            continue
                        # get the panels
                        panel_1, panel_2, panel_3, panel_4, panel_5, panel_6 = divide_into_panels(folder, 'images/buoys/{}/{}'.format(folder, image))
                        # save the panels to the images/panels directory
                        # remove the .jpg from the image name first so that we can save the panels with the same name as the image
                        image.replace('.jpg', '') # remove the .jpg from the image name
                        # save the panels
                        cv2.imwrite('images/panels/{}/{}_panel_1.jpg'.format(folder, image), panel_1)
                        cv2.imwrite('images/panels/{}/{}_panel_2.jpg'.format(folder, image), panel_2)
                        cv2.imwrite('images/panels/{}/{}_panel_3.jpg'.format(folder, image), panel_3)
                        cv2.imwrite('images/panels/{}/{}_panel_4.jpg'.format(folder, image), panel_4)
                        cv2.imwrite('images/panels/{}/{}_panel_5.jpg'.format(folder, image), panel_5)
                        cv2.imwrite('images/panels/{}/{}_panel_6.jpg'.format(folder, image), panel_6)
                        logging.info('Saved panels for image: {}'.format(image))
                    except:
                        logging.ERROR('Error while saving panels for image: {}'.format(image))
                        # print('Could not create panels for image: {}'.format(image))
                        continue

        # for each folder in the images/panels folder, stitch the images together and save them to the images/panoramas folder with the same name as the folder + panorama.png

        #note: the for loop below does not account for the fact that there are multiple captures with 6 frames per capture. This means that the images will be stitched together incorrectly. This is a problem that needs to be fixed. Find a way to select only the sets of 6 images that go together to stitch together.

        # for folder in os.listdir('images/panels'):
        #     if folder == '.DS_Store':
        #         continue
        #     images = []
        #     for image in tqdm(os.listdir(f'images/panels/{folder}')):
        #         if image == '.DS_Store':
        #             continue
        #         if artist_eval(f'images/panels/{folder}/{image}'): # if the artist decides to make a panorama (True)
        #             images.append(cv2.imread(f'images/panels/{folder}/{image}'))
        #             try:
        #                 panorama = vincent.make_panorama(images)
        #                 cv2.imwrite(f'images/panoramas/{folder}_panorama.png', panorama)
        #             except:
        #                 ##print(f'Could not create panorama for {folder}')
        #                 pass
        #         else:
        #             #print(f'Could not create panorama for {folder}')
        #             pass


        # Stage 6: Create the buoy dataframes
            # if it has been ten minutes since the last time the data was fetched, fetch the data again
        if time.time() - last_time_fetched > 600 or first_run:
            latest_data = get_latest_data() # get the latest data from the RSS feed (updates every 10 minutes)
            # save the master dataframe to a csv file
            run_date = time.strftime("%Y%m%d_%H%M%S")
            latest_data.to_csv(f'data/rss/rss_buoy_data_{run_date}.csv', index=False)
            print('Done with this run')
            time_last_fetched_rss = time.time() # get the time of the last fetch



        # how much time has passed since the start of the loop?
        time_elapsed = datetime.datetime.now() - start_time
        # wait until the time elapsed is 15 minutes from the start of the loop
        print("Waiting for the remainder of the minutes...")
        # wait_period = 100 # was 900 (15 minutes)
        for i in tqdm(range(wait_period - time_elapsed.seconds)):
            time.sleep(1)
        iteration_counter += 1
    except Exception as e:
        print(e)
        print('Error occurred. Restarting loop...')
        continue