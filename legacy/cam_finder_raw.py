# %% [markdown]
# # Goal: determine which cameras (on buoys) from NOAA have webcams that are functional and can be used for the project.
#

# %%
import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm
import time
import shutil
from difPy import dif
import cv2
import os
import requests
import json
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from tqdm import tqdm
# import #logging

import time
import random
import imutils
# import pillow
from PIL import Image

# the list of buoys to check for cameras are in the file data/buoy_ids.csv
with open('data/buoys.csv', 'r') as f:
    buoy_ids = f.read().splitlines()

# How can we know if a buoy has a camera?
# We can check the text of the page for the buoy on the NOAA website for the word "Camera"
# If the word "Camera" is in the text, then we know there is a camera

# We can use the requests library to get the text of the page
# We can use the BeautifulSoup library to parse the text of the page
# We can use the re library to search for the word "Camera" in the text of the page

# We will store the buoy ids that have cameras in the list cam_buoys
cam_buoys = []


# %%
# Set up #logging file
#logging.basicConfig(filename='cam_finder.log', level=#logging.INFO)


# %% [markdown]
# Stage One

# %%
ids = ["21414","21415","21416","21417","21418","21419","32301","32302","32411","32412","32413","41001","41002","41003","41004","41007","41008","41009","41010","41011","41012","41016","41017","41018","41021","41022","41023","41036","41040","41041","41043","41044","41046","41049","41420","41421","41424","41425","42001","42002","42003","42004","42007","42008","42009","42010","42011","42012","42017","42018","42019","42020","42025","42035","42038","42039","42040","42041","42042","42053","42056","42057","42058","42059","42060","42065","42408","42409","42429","42501","42503","42534","44001","44003","44004","44005","44006","44007","44010","44011","44012","44013","44014","44015","44019","44020","44023","44025","44026","44027","44066","44070","44071","44401","44402","44403","45002","45003","45004","45005","45006","45007","45010","45011","45012","46001","46002","46003","46007","46008","46009","46010","46011","46012","46015","46016","46017","46018","46019","46020","46023","46024","46025","46026","46027","46028","46031","46032","46033","46034","46035","46037","46040","46041","46042","46043","46045","46047","46051","46053","46054","46059","46060","46061","46066","46069","46070","46071","46072","46073","46077","46078","46079","46080","46081","46082","46085","46086","46087","46088","46089","46090","46107","46115","46270","46290","46401","46402","46405","46406","46407","46408","46409","46410","46413","46414","46415","46416","46419","46490","46779","46780","46781","46782","46785","51000","51001","51002","51003","51004","51005","51028","51100","51101","51406","51407","51425","52009","52401","52402","52403","52404","52405","91204","91222","91251","91328","91338","91343","91356","91365","91374","91377","91411","91442","46265","41670","41852","41904","41933","48916","48917","52838","52839","52840","52841","52842","52843","52862","55012","55013","55015","55016","55023","55042","58952","31052","31053","41052","41053","41056","41058","41115","41121","41030","44042","44043","44057","44058","44059","44061","44064","44068","45016","45017","45018","45019","45177","45202","45203","45204","45205","45206","45207","46116","46117","46127","42014","42021","42022","42023","42024","42026","32404","41029","41033","41037","41038","41064","41065","41110","41119","41159","32488","41193","44138","44139","44140","44141","44142","44150","44176","44235","44251","44255","44258","44488","45132","45135","45136","45137","45138","45139","45142","45143","45144","45145","45147","45148","45151","45152","45154","45155","45158","45159","46036","46131","46132","46084","46134","46138","46139","46147","46181","46183","46184","46185","46204","46207","46208","46303","46304","48021","45162","45163","45195","23219","23227","32067","32068","42087","42088","42089","42090","46109","46110","46111","46112","21346","21347","21348","21595","21597","21598","21637","21640","22102","22103","22104","22105","22106","22107","45029","45164","45165","45168","45169","45176","46091","46092","62091","62092","62093","62094","41097","41098","41100","41101","41300","61001","45025","45175","44039","44040","44060","23220","23223","23225","46261","46263","48901","48908","48909","48912","44024","44029","44030","44031","44032","44033","44036","44037","45172","45173","46118","46119","46531","46534","46538","46565","44075","44076","44077","44078","46097","46098","51046","51201","51202","51203","51204","51205","51208","51209","51210","51211","51212","51213","52202","52211","13002","13008","13009","13010","15001","15002","31001","31002","31003","31004","31005","31006","62121","62124","62125","62126","62127","62130","62144","62145","62146","62147","62148","62149","62165","62166","63105","63110","63112","63113","14041","14043","14047","23001","23003","23004","23008","23009","23010","23011","23012","23013","23016","23017","53005","53006","53009","53040","56053","01506","01507","01518","01537","48904","48907","01521","01522","01523","01524","01526","01531","01535","01536","01538","01909","01910","31201","41112","41113","41114","41116","41118","41120","42084","42091","42094","42099","44088","44094","44099","44100","44172","46114","46211","46212","46215","46216","46217","46218","46219","46220","46223","46224","46225","46226","46227","46228","46231","46232","46234","46235","46236","46237","46240","46241","46242","46243","46244","46245","46249","46250","46251","46253","46254","46256","46262","46267","46268","46269","46273","46274","51200","48212","48213","48214","48677","48678","48679","48680","48911","42044","42045","42046","42047","42048","42049","42078","42079","42093","42095","42097","44056","45180","46259","46266","62028","62029","62030","62050","62081","62103","62108","62163","62170","62298","62301","62303","62442","64045","44098","46121","46122","46123","46124","28902","28903","28904","28906","28907","28908","58900","58902","58903","58904","58905","58906","58909","68900","78900","45014","45184","44053","01517","32012","41060","41061","21D20","32D12","32D13","41A46","41S43","41S46","46B35","ALSN6","AMAA2","AUGA2","BLIA2","BURL1","BUSL1","CDRF1","CHLV2","CLKN7","CSBF1","DBLN6","DESW1","DRFA2","DRYF1","DSLN7","DUCN7","EB01","EB10","EB33","EB35","EB36","EB43","EB52","EB53","EB70","EB90","EB91","EB92","FARP2","FBIS1","FPSN7","FWYF1","GBCL1","GDIL1","GLLN6","IOSN3","LONF1","LPOI1","MDRM1","MISM1","MLRF1","MPCL1","PILA2","PILM4","PLSF1","POTA2","PTAC1","PTAT2","SANF1","SAUF1","SBIO1","SGNW3","SGOF1","SISW1","SPGF1","SRST2","STDM4","SUPN6","SVLS1","THIN6","VENF1","HBXC1","MYXC1","TDPC1","FSTI2","DMNO3","GPTW1","HMNO3","PRTO3","SEFO3","SETO3","SRAW1","SRFW1","TANO3","ANMF1","ARPF1","BGCF1","CAMF1","CLBF1","EGKF1","NFBF1","PTRF1","SHPF1","MBIN7","MBNN7","OCPN7","BSCA1","CRTA1","DPHA1","KATA1","MBLA1","MHPA1","SACV4","BBSF1","BDVF1","BKYF1","BNKF1","BOBF1","BSKF1","CNBF1","CWAF1","DKKF1","GBIF1","GBTF1","GKYF1","JBYF1","JKYF1","LBRF1","LBSF1","LMDF1","LMRF1","LSNF1","MDKF1","MNBF1","MUKF1","NRRF1","PKYF1","TCVF1","THRF1","TPEF1","TRRF1","WIWF1","WPLF1","APNM4","CHII2","MCYI3","SRLM4","SVNM4","TBIM4","THLO1","LCIY2","LLBP7","FWIC3","MISC3","MISN6","NCSC3","NOSC3","OFPN6","ILDL1","MRSL1","SIPM6","SLPL1","LUML1","TAML1","AKXA2","APMA2","BEXA2","CDXA2","CPXA2","DHXA2","DPXA2","ERXA2","GBXA2","GEXA2","GIXA2","GPXA2","HMSA2","ICYA2","JLXA2","JMLA2","JNGA2","KEXA2","KNXA2","KOZA2","LIXA2","MIXA2","MRNA2","MRYA2","NKLA2","NKXA2","NLXA2","NMXA2","NSXA2","PAUA2","PEXA2","PGXA2","PPXA2","PTLA2","RIXA2","SCXA2","SIXA2","SKXA2","SLXA2","SPXA2","SRXA2","STXA2","SXXA2","TKEA2","TPXA2","UQXA2","VDXA2","WCXA2","MSG10","MSG12","ACQS1","ACXS1","ANMN6","ANRN6","APQF1","APXA2","BILW3","BRIM2","BSLM2","BVQW1","CHNO3","CHQO3","CWQT2","DBQS1","DEQD1","DRSD1","EAZC1","EHSC1","EVMC1","FFFC1","GBHM6","GBQN3","GBRM6","GDQM6","GGGC1","GTQF1","GTXF1","HBMN6","HMRA2","HUQN6","JCTN4","JOBP4","JOQP4","JOXP4","KCHA2","LTQM2","MIST2","MQMT2","MWQT2","NAQR1","NAXR1","NIQS1","NOXN7","NPQN6","NPXN6","OWDO1","OWQO1","OWSO1","PBLW1","PKBW3","RKQF1","RKXF1","RYEC1","SAQG1","SCQC1","SCQN6","SEQA2","SFXC1","SKQN6","SLOO3","TCSV2","TIQC1","TIXC1","TKPN6","WAQM3","WAXM3","WELM1","WEQM1","WEXM1","WKQA1","WKXA1","WYBS1","NLMA3","SBBN2","SLMN2","BAXC1","BDRN4","BDSP1","BGNN6","BKBF1","BLIF1","BRND1","CHCM2","CHYV2","COVM2","CPMW1","CPNW1","CRYV2","DELD1","DMSF1","DOMV2","DPXC1","EBEF1","FMOA1","FRVM3","FRXM3","FSKM2","FSNM2","GCTF1","LNDC1","LQAT2","LTJF1","MBPA1","MCGA1","MHBT2","MRCP1","MTBF1","MZXC1","NBLP1","NFDF1","NWHC3","OMHC1","OPTF1","PDVR1","PEGF1","PFDC1","PFXC1","PPTM2","PPXC1","PRJC1","PRUR1","PSBC1","PSXC1","PTOA1","PVDR1","PXAC1","PXOC1","PXSC1","QPTR1","RPLV2","RTYC1","SEIM1","SJSN4","SKCF1","SWPM4","TCNW1","TLVT2","TPAF1","TSHF1","TXVT2","UPBC1","WDSV2","ACYN4","ADKA2","AGCM4","ALIA2","ALXN6","AMRL1","APAM2","APCF1","APRP7","ASTO3","ATGM1","ATKA2","BEPB6","BFTN7","BHBM3","BISM2","BKTL1","BLTM2","BYGL1","BZBM3","CAMM2","CAPL1","CARL1","CASM1","CECC1","CFWM1","CHAO3","CHAV3","CHBV2","CHSV3","CHYW1","CLBP4","CMAN4","CMTI2","CNDO1","CRVA2","DILA1","DKCM6","DTLM4","DUKN7","DULM5","EBSW1","ERTF1","ESPP4","FAIO1","FCGT2","FMRF1","FOXR1","FPTT2","FRCB6","FRDF1","FRDW1","FREL1","FRPS1","FTPC1","GBWW3","GCVF1","GDMM5","GISL1","GNJT2","GTOT2","GWPM6","HBYC1","HCGN7","HLNM4","HMDO3","ICAC1","IIWC1","ILOH1","ITKA2","JMPN7","JNEA2","KECA2","KGCA2","KLIH1","KPTN6","KPTV2","KWHH1","KYWF1","LABL1","LAMV3","LAPW1","LCLL1","LDTM4","LOPW1","LPNM4","LTBV3","LTRM4","LWSD1","LWTV2","MBRM4","MCGM4","MCYF1","MEYC1","MGIP4","MGZP4","MOKH1","MQTT2","MRHO1","MROS1","MTKN6","MTYC1","NEAW1","NIAN6","NJLC1","NKTA2","NLNC3","NMTA2","NTBC1","NTKM3","NUET2","NWCL1","NWPR1","NWWH1","OCIM2","OHBC1","OLSA2","OOUH1","ORIN7","OSGN6","PCBF1","PCLF1","PCOC1","PGBP7","PHBP1","PLXA2","PNLM6","PORO3","PRDA2","PRYC1","PSBM1","PSLC1","PTAW1","PTIM4","PTIT2","PTWW1","RARM6","RCKM4","RCYF1","RDDA2","RDYD1","SAPF1","SBEO3","SBLF1","SDBC1","SDHN4","SHBL1","SJNP4","SKTA2","SLIM2","SNDP5","SWLA2","SWPV2","TESL1","THRO1","TLBO3","TRDF1","TXPT2","ULAM6","ULRA2","UNLA2","VAKF1","VDZA2","WAHV2","WAKP8","WASD2","WAVM6","WLON7","WPTW1","WYCM6","YATA2","BLTA2","CDEA2","EROA2","LCNA2","PBPA2","PRTA2","SDIA2","AGMW3","BHRI3","BIGM4","BSBM4","CBRW3","CLSM4","FPTM4","GBLW3","GRMM4","GSLM4","GTLM4","GTRM4","KP53","KP58","KP59","LSCM4","MEEM4","NABM4","PCLM4","PNGW3","PRIM4","PSCM4","PWAW3","SBLM4","SPTM4","SXHW3","SYWW3","TAWM4","WFPM4","BARN6","CBLO1","CHDS1","CMPO1","GELO1","HHLO1","LORO1","NREP1","OLCN6","RPRN6","WATS1","AUDP4","FRDP4","PLSP4","VQSP4","CGCL1","SKMG1","SPAG1","AVAN4","BRBN4","OCGN4","AWRT2","BABT2","BZST2","CLLT2","CPNT2","EMAT2","GRRT2","HIST2","IRDT2","LUIT2","LYBT2","MGPT2","NWST2","PACT2","PCGT2","PCNT2","PMNT2","PORT2","RSJT2","RTAT2","RTOT2","SDRT2","SGNT2","TAQT2","BTHD1","FRFN7","JPRN7","18CI3","20CM4","GDIV2","32ST0","41NT0"]
buoy_ids = ids

buoy_ids_with_cameras = [] # blank list to store buoy ids with cameras

# %%
# Loop through the buoy ids
for buoy_id in tqdm(buoy_ids):
    # if the page text needs to be retrieved then do so, else pull from the text files
    if not os.path.exists('data/buoy_pages/{}.txt'.format(buoy_id)):
        # get the page text
        time.sleep(1)
        page = requests.get('https://www.ndbc.noaa.gov/station_page.php?station={}'.format(buoy_id))
        # write the page text to a file
        with open('data/buoy_pages/{}.txt'.format(buoy_id), 'w+') as f:
            f.write(page.text)
        if 'Buoy Camera' in page.text:
            # save the buoy id to the list of buoy ids with cameras
            buoy_ids_with_cameras.append(buoy_id)
            with open('data/buoy_ids_with_cameras.txt', 'a+') as f:
                f.write(buoy_id + ',')
    else:
        # read the page text from a file
        # with open('data/buoy_pages/{}.txt'.format(buoy_id), 'r') as f:
        #     page = f.read()
        pass

# %%
# parse the page text
cam_buoys = [] # blank list to store buoy ids with cameras

# %%
# Purpose: parse the page text to get the camera urls and save them to a file. The page is parsed using BeautifulSoup and the camera urls are extracted using regular expressions.

# only do this once to get the camera urls and save them to a file for later use
# if the file does not exist then do so, else pull from the file
if not os.path.exists('data/buoy_ids_with_cameras.txt'):
    # open each buoy page text file in data/buoy_pages and parse the page text
    for buoy_page in tqdm(os.listdir('data/buoy_pages')):
        with open('data/buoy_pages/{}'.format(buoy_page), 'r') as f:
            page = f.read() # read the page text
        soup = BeautifulSoup(page, 'html.parser')
        # search for the word "Camera" in the page text
        if re.search('Camera', soup.text):
            # if the word "Camera" is in the page text, then the buoy has a camera.
            # save the name of the file (without the .txt extension) to the list of buoy ids with cameras (cam_buoys) array.
            cam_buoys.append(buoy_page.replace('.txt', ''))
else:
    # read the buoy ids with cameras from the file
    with open('data/buoy_ids_with_cameras.txt', 'r') as f:
        cam_buoys = f.read().split(',')
# save the buoy ids with cameras to a file


# %%
# Purpose: get the camera urls from the buoy pages and save them to a file
# open each buoy page text file in data/buoy_pages and parse the page text to get the camera urls

print(f'Found {len(cam_buoys)} buoys with functional cameras.')
# save them to the file data/buoys.csv
with open('data/buoys.csv', 'w+') as f:
    f.write("'buoy_id',\n") # could add lat and lng later
    for buoy_id in cam_buoys: # loop through the buoy ids with cameras
        f.write('{},\n'.format(buoy_id)) # write the buoy id to the file
print('These have been saved in the cam_buoys array.')

#46078 is functional and very interesting as well.


# %%
# building the urls for the cameras
cam_urls = [] # blank list to store the camera urls
for buoy_id in tqdm(cam_buoys):
    cam_url = 'https://www.ndbc.noaa.gov/buoycam.php?station={}'.format(buoy_id)
    cam_urls.append(cam_url) # add the camera url to the list of camera urls

# save the camera urls to the file data/camera_urls.csv
with open('data/camera_urls.csv', 'w+') as f:
    f.write("'cam_url',\n")
    for cam_url in cam_urls: # loop through the camera urls
        f.write('{},\n'.format(cam_url)) # write the camera url to the file

print('These have been saved in the cam_urls array.')

# %%
def panel_sorter():
    # Go into each panel directory and sort the images into folders by the date in their filename (if they haven't already been sorted)
    # example unsorted directory: 'images/panels/46078/2022_11_5_15_44_panel_1.png'
    # example sorted directory: images/panels/51000/2022_11_5_15_44/panel_1.png

    for buoy_id in os.listdir('images/panels'):
        if buoy_id != '.DS_Store' and '.' not in buoy_id:
            for image in os.listdir('images/panels/{}'.format(buoy_id)):
                if image != '.DS_Store' and '.' not in image:
                    try:
                        # find the 2022_11_5_15_44 (#_#_#_#_#) part of the filename and make a new folder with that name.
                        # move the image that contain the matching pattern in their filename into the new folder
                        # if the folder already exists, then move the image into the existing folder
                        # if the folder doesn't exist, then create the folder and move the image into the new folder
                        # if the image has already been moved into a folder, then skip it
                        # if the image is the .DS_Store file, then skip it
                        # if the image is None, then skip it

                        pattern = re.compile(r'\d{4}_\d{1,2}_\d{1,2}_\d{1,2}_\d{1,2}') # create a pattern to match the date in the filename
                        match = pattern.search(image) # search the filename for the date
                        if match:
                            date = match.group() # get the date from the match
                            if not os.path.exists('images/panels/{}/{}'.format(buoy_id, date)): # if the folder doesn't exist, then create it
                                os.makedirs('images/panels/{}/{}'.format(buoy_id, date)) # create the folder
                            shutil.move('images/panels/{}/{}'.format(buoy_id, image), 'images/panels/{}/{}/{}'.format(buoy_id, date, image)) # move the image into the new folder
                        else:
                            pass
                    except Exception as e:
                        pass
            else:
                pass
        else:
            pass

# %%
import cv2
import numpy as np
import urllib.request
import time

# Image Processing Functions
# we have the images in the images/buoys directory already downloaded.

def divide_into_panels(buoy_id, image_file):
    # divide the image into six panels, as dictated in the image processing pipeline for NOAA Buoy Cameras Comments above.

    # read the image
    img = cv2.imread(image_file)
    # get the name of the image file
    image_name = image_file.split('/')[-1]

    # get the dimensions of the image
    height, width, channels = img.shape

    # Before dividing into panels, crop the image to remove 30 pixels from the bottom of the image.
    # This is to remove the "Buoy Camera" text from the image.
    # img = img[0:height-30, 0:width]

    # divide the image into six panels named: image_name_panel_#.jpg
    image_name = image_name.replace('.jpg', '') # remove the .jpg extension for now
    panel_1 = img[0:height-30, 0:int(width/6)]
    cv2.imwrite('images/panels/{}/{}_panel_1.png'.format(buoy_id, image_name), panel_1)
    panel_2 = img[0:height-30, int(width/6):int(width/3)]
    cv2.imwrite('images/panels/{}/{}_panel_2.png'.format(buoy_id, image_name), panel_2)
    panel_3 = img[0:height-30, int(width/3):int(width/2)]
    cv2.imwrite('images/panels/{}/{}_panel_3.png'.format(buoy_id, image_name), panel_3)
    panel_4 = img[0:height-30, int(width/2):int(width*2/3)]
    cv2.imwrite('images/panels/{}/{}_panel_4.png'.format(buoy_id, image_name), panel_4)
    panel_5 = img[0:height-30, int(width*2/3):int(width*5/6)]
    cv2.imwrite('images/panels/{}/{}_panel_5.png'.format(buoy_id, image_name), panel_5)
    panel_6 = img[0:height-30, int(width*5/6):width]
    cv2.imwrite('images/panels/{}/{}_panel_6.png'.format(buoy_id, image_name), panel_6)
    return panel_1, panel_2, panel_3, panel_4, panel_5, panel_6



# %%
import math
from PIL import Image

def is_it_daytime(image_path):
    # This function will take an image path and return True if it is daytime and False if it is nighttime.
    # The image path should be a string.
    # The image path should be the path to the image that was used to create the panels.
    # The image path should be in the format 'images/buoys/46025/2019_12_12_12_12.jpg'

    # get the image
    img = Image.open(image_path)
    # get the image size
    width, height = img.size
    # get the pixel values for the center of the image
    pixel_values = img.getpixel((int(width/2), int(height/2)))
    # get the pixel values for the top left corner of the image
    upper_left = img.getpixel((0, 0))
    # get the pixel values for the top right corner of the image
    upper_right = img.getpixel((width-1, 0))

    # get the pixel values for the centers of the six panels in the image
    panel_1 = img.getpixel((int(width/12), int(height/2)))
    panel_2 = img.getpixel((int(width/4), int(height/2)))
    panel_3 = img.getpixel((int(width/2), int(height/2)))
    panel_4 = img.getpixel((int(width*3/4), int(height/2)))
    panel_5 = img.getpixel((int(width*5/6), int(height/2)))
    panel_6 = img.getpixel((int(width*11/12), int(height/2)))

    # put those pixel values into a list
    pixels_array = [pixel_values, upper_left, upper_right, panel_1, panel_2, panel_3, panel_4, panel_5, panel_6]
    brights = []

    # get the red, green, and blue values
    red, green, blue = pixel_values

    # for the panels
    for measurement in pixel_values[3:]:
        # calculate the brightness
        brightness = math.sqrt(0.241*(red**2) + 0.691*(green**2) + 0.068*(blue**2)) # this is the formula for brightness
        # add the brightness to the list
        brights.append(brightness) # add the brightness to the list of brightnesses
    # get the median brightness
    median_brightness = np.median(brights)

    # if the red value is greater than 200, then it is unusual and we want to return True

    # print('median brightness: {}'.format(median_brightness))
    # print('red: {}'.format(red))
    # print('green: {}'.format(green))
    # print('blue: {}'.format(blue))

    if red > 200 and median_brightness > 25:
        return True
    # if over 95% of the colors in the image are pure white it is a blank image and we will return False
    if median_brightness > 350:
        return False
    # if the brightness is greater than 100, it is daytime
    if median_brightness > 50:
        return True
    else:
        return False

# %%
import datetime

# %% [markdown]
# Now that we have a list of the buoys with cameras we can isolate these, create the urls, and download the latest images to the images folder. We will use the requests library to download the images and the os library to create the folder structure.

# %% [markdown]
# All of these photos are a series of six panels spread horizontally to create a panorama of the area around the buoy. The first step is to isolate the panels and then crop them to the size of the camera. We will use Python and CV2 to do this.
#
# # Image Processing Pipeline for NOAA Buoy Cameras
# 1. Get the latest image in the buoy's folder
# 2. Isolate the six panels and save them as individual images with the pattern "original_name + panel_number".png (e.g. 41001_2019-07-01_00:00:00_panel_1.png)
# 3. The height of the cropped image should be 270 pixels (measured from the top of the image)
# 4. The widths of each cropped panel should be 480 pixels (measured from the left of the image). The first panel should start at 0 pixels and the last panel should end at 2880 pixels.
#

# %%
buoy_update_rates_dict = {} # blank dictionary to store the update rates for each buoy (i.e. how often the buoy takes a picture (measured in seconds))
# fill the dictionary with blank update rate arrays for each buoy.
# these arrays will be averaged to get the average update rate for each buoy in real time.

for buoy_id in cam_buoys:
    buoy_update_rates_dict[buoy_id] = 1 # set the initial update rate to 600 seconds (10 minutes)

# %%
def image_has_changed(image_one,image_two):
    # looks for changes between two images and returns True if there are changes and False if there are not.
    # this function is used to determine what rate the images are being updated for each buoy.
    # if the images are not changing, then the buoy is not updating the images.
    # if the images are changing, then the buoy is updating the images.
    try:
        image_one = cv2.imread(image_one)
        image_two = cv2.imread(image_two)
        difference = cv2.subtract(image_one, image_two)
        result = not np.any(difference)
        return result
    except:
        return False

    # difference = cv2.subtract(image_one, image_two)
    # b, g, r = cv2.split(difference)
    # if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
    #     return False # images are the same
    # else:
    #     return True # there are changes between the two images

# %%
class Artist:
    def __init__(self):
        self.folder = 'images'
    def make_panorama(self, images):
        # this function will take a list of images and make a panorama out of them
        # resize image
        scale_percent = 70 # percent of original size
        images = [cv2.resize(images[i], (int(images[i].shape[1]*scale_percent/100), int(images[i].shape[0]*scale_percent/100)), interpolation= cv2.INTER_AREA) for i in range(len(images))]
        # create a list of the images
        # print('Making a panorama...')
        # images = [cv2.imread(image) for image in images]
        # # resize the images
        # print('images: {}'.format(images))
        # print(f'Resizing images...')
        # images = [cv2.resize(images[i], (int(images[i].shape[1]*scale_percent/100), int(images[i].shape[0]*scale_percent/100)), interpolation= cv2.INTER_AREA) for i in range(len(images))]
        # take a list of images and stitch them together with cv2
        stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create() # stitcher object
        # make stitcher with higher resolution
        stitcher.setPanoConfidenceThresh(0.1)
        stitcher.setRegistrationResol(0.6)
        stitcher.setSeamEstimationResol(0.1)
        stitcher.setCompositingResol(0.6)
        stitcher.setWaveCorrection(True)
        stitcher.setWaveCorrectKind(cv2.WAVE_CORRECT_HORIZ)
        stitcher.setFeaturesFinder(cv2.ORB_create())
        stitcher.setFeaturesMatcher(cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING))
        stitcher.setBundleAdjuster(cv2.detail_BundleAdjusterRay())
        stitcher.setExposureCompensator(cv2.detail.ExposureCompensator_createDefault(cv2.detail.ExposureCompensator_GAIN_BLOCKS))
        stitcher.setBlender(cv2.detail.Blender_createDefault(cv2.detail.Blender_NO))
        stitcher.setSeamFinder(cv2.detail.SeamFinder_createDefault(cv2.detail.SeamFinder_VORONOI_SEAM))
        stitcher.setWarper(cv2.PyRotationWarper(cv2.ROTATION_WARP_PERSPECTIVE))
        stitcher.setInterpolationFlags(cv2.INTER_LINEAR_EXACT)


        (status, stitched) = stitcher.stitch(images) # stitch the images together with cv2


        # print('stitched: {}'.format(stitched))


        if status == 0:
            # resize the stitched image to be a panorama size
            stitched = cv2.resize(stitched, (int(stitched.shape[1]*scale_percent/100), int(stitched.shape[0]*scale_percent/100)), interpolation= cv2.INTER_AREA)
            return stitched
        else:
            return None


def artist_eval(image_path):
    # get the image
    img = Image.open(image_path)
    # get the image size
    width, height = img.size
    # get the pixel values for the center of the image
    pixel_values = img.getpixel((int(width/2), int(height/2)))
    # get the pixel values for the top left corner of the image
    upper_left = img.getpixel((0, 0))
    # get the pixel values for the top right corner of the image
    upper_right = img.getpixel((width-1, 0))

    # get the pixel values for the centers of the six panels in the image
    panel_1 = img.getpixel((int(width/12), int(height/2)))
    panel_2 = img.getpixel((int(width/4), int(height/2)))
    panel_3 = img.getpixel((int(width/2), int(height/2)))
    panel_4 = img.getpixel((int(width*3/4), int(height/2)))
    panel_5 = img.getpixel((int(width*5/6), int(height/2)))
    panel_6 = img.getpixel((int(width*11/12), int(height/2)))

    # a panorama is best when the panels are similar in color
    # let's see how close the panels are to each other
    # we'll use the mean squared error
    mse_1 = np.mean((np.array(panel_1) - np.array(panel_2)) ** 2)
    mse_2 = np.mean((np.array(panel_2) - np.array(panel_3)) ** 2)
    mse_3 = np.mean((np.array(panel_3) - np.array(panel_4)) ** 2)
    mse_4 = np.mean((np.array(panel_4) - np.array(panel_5)) ** 2)
    mse_5 = np.mean((np.array(panel_5) - np.array(panel_6)) ** 2)

    # the mean squared error is a good measure of how similar the panels are
    # the lower the mse, the more similar the panels are
    # let's take the average of the mse's
    mse = (mse_1 + mse_2 + mse_3 + mse_4 + mse_5) / 5

    # we will make a panorama if the mse is less than 100
    # this is a pretty arbitrary number, but it seems to work well
    if mse < 100:
        return True
    else:
        return False

# %%
def check_for_updates(buoy_update_rates_dict):
    # Check the buoy_update_rates_dict to see if any of the buoys satistfy the update rate requirements:
    # Requirements: the current time minus the last time we downloaded an image for this buoy must be greater than the update rate for this buoy. If it is, then we will add the buoy id to the list of buoys that need to be updated and return it to the main function.
    # If the buoy_update_rates_dict is empty, then we will return an empty list.
    # If the buoy_update_rates_dict is not empty, then we will check the update rates for each buoy and return a list of the buoy ids that need to be updated.

    if len(buoy_update_rates_dict) == 0:
        return []
    else:
        buoys_to_update = []
        for buoy_id in buoy_update_rates_dict:
            #print(buoy_update_rates_dict[buoy_id])
            if (datetime.datetime.now() - buoy_update_rates_dict[buoy_id][1]).seconds > buoy_update_rates_dict[buoy_id][0]:
                # the current time minus the last time we downloaded an image for this buoy must be greater than the update rate for this buoy.
                buoys_to_update.append(buoy_id)
                # reset the last time we downloaded an image for this buoy to the current time.
                buoy_update_rates_dict[buoy_id][1] = datetime.datetime.now()

        return buoys_to_update, buoy_update_rates_dict # return a list of the buoy ids that need to be updated


# %%
# test check_for_updates function
buoy_update_rates_dict = {'46025': [600, datetime.datetime.now()]}
print(check_for_updates(buoy_update_rates_dict)) # should return an empty list
# wait 10 seconds
time.sleep(10)
print(check_for_updates(buoy_update_rates_dict)) # should return ['46025']


# %%
# import shutil
# from difPy import dif

# go through each buoy and check if there are duplicated images in the images/buoys folder
for folder in os.listdir('images/buoys'):
    if folder == '.DS_Store':
        continue
    # get the list of images in the folder
    # sort the images by date
    # make folder_path variable from relative path
    folder_path = 'images/buoys/{}'.format(folder)
    search = dif(folder_path, similarity='normal', show_output=False, show_progress=True, silent_del=True, delete=True)



# %%
# get the latest news from the RSS feed at https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt, and return the pandas dataframe of the data.
import pandas as pd
import requests
import io

def get_latest_data():
    url = "https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt"
    s=requests.get(url).content

    # the table contains two rows that have header data. combine them into one row.

    df = pd.read_csv(io.StringIO(s.decode('utf-8')), sep='\s+')
    df.columns = df.columns.str.strip()
    # df = df.dropna(axis=1, how='all')
    # df = df.dropna(axis=0, how='all')
    # df = df.dropna(axis=0, how='any')
    # df = df.reset_index(drop=True)
    # df = df.drop(df.index[0])

    return df



# %% [markdown]
# # The Artist Class Explained
#
# The goal of the artist is to identify which buoy photo sets are the most beautiful or interesting.
# The artist will be able to select a photo set and then rate it on a scale of 1-5.
#
# the photo sets are located in the images/panels directory and are named by the buoy id
# over time the artist will learn which times of day are the most beautiful and interesting for each buoy and will be able to teach the system to automatically raise the rate of photo capture during those times. For example, the most beautiful photos could be those that are taken at sunrise and sunset. The artist could also teach the system to take more photos when the weather is clear and less photos when the weather is cloudy. Unless cloudy weather is the most beautiful weather. That is up for debate.
#
# As the system collects data about the buoy's photos it will begin to add weights to each buoy which will inform a random.random.choice function with weighted probabilities. This will allow the system to take photos of the most beautiful buoys more often.
#
#
#
# The artist will use an ESRGAN on the photo sets that they think are most beautiful and enhance their quality. If they go well together then the artist will be able to create a video of the photo set or a panorama of the photo set.
#
# - Step One - Evaluation of the photo sets
# - Step Two - ESRGAN on the photo sets
# - Step Three - Video creation
# - Step Four - Panorama creation
#
#

# %%
# download the images
# until the current time is 10 pm Central Standard Time (CST) Keep downloading images from the cameras every 10 minutes
iteration_counter = 0
vincent = Artist() # create an instance of the Artist class

# %%
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
        #!print('Starting the download loop at {}'.format(start_time))
        # print('I can still see things! Downloading images...')
        for cam_url in tqdm(cam_urls):
            # get the buoy id from the camera url
            buoy_id = re.search('station=(.*)', cam_url).group(1)
            # get the current time
            now = datetime.datetime.now()
            # create a directory for the buoy id if it doesn't already exist
            if not os.path.exists('images/buoys/{}'.format(buoy_id)):
                os.makedirs('images/buoys/{}'.format(buoy_id))
                #logging.info("Created directory for buoy {}".format(buoy_id))

            # get the image
            #logging.info("Checking buoy {}".format(buoy_id)) # log the buoy id
            if 'images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute) not in os.listdir('images/buoys/{}'.format(buoy_id)): # if the image has not already been downloaded
                time.sleep(0.25) # wait 0.25 seconds to avoid getting blocked by the server
                img = requests.get(cam_url) # get the image
                # save the image
                with open('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute), 'wb+') as f:
                    f.write(img.content) # write the image to the file
                # check if the image is daytime or nighttime
                # #logging.WARNING("Skipped night detection model for buoy {}".format(buoy_id))
                #if not is_it_daytime('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute)): # if it is nighttime
                    # then we will delete the image
                    #*print(f'Deleting image for buoy {buoy_id} because it is nighttime.')
                    #*os.remove('images/buoys/{}/{}_{}_{}_{}_{}.jpg'.format(buoy_id, now.year, now.month, now.day, now.hour, now.minute))
                #    pass
            else:
                pass # if the image already exists, don't download it again


        print("Paneling images...")
        #logging.INFO("Beginning to panel images (line 24)") #! at {}".format(datetime.datetime.now()))
        # Save the panels to the images/panels directory
        list_of_buoys = os.listdir('images/buoys') # get the list of buoy ids by their directory names
        for buoy_id in tqdm(list_of_buoys):
            # get the list of images for the buoy
            if buoy_id != '.DS_Store':
                images = os.listdir('images/buoys/{}'.format(buoy_id))
                # if the image has not already been used to create panels, create the panels and save them to the images/panels directory
                #logging.info("Saving panels for buoy {}".format(buoy_id))
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
                    #logging.info("Saved panels for buoy {}".format(buoy_id))
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
                        #logging.info('Saved panels for image: {}'.format(image))
                    except:
                        #logging.ERROR('Error while saving panels for image: {}'.format(image))
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

# %% [markdown]
# This is designed to run continuously until we have a large enough dataset to train a model. We will use the time library to set the interval between downloads.

# %% [markdown]
# # Text Recognition Pipeline for NOAA Buoy Camera Images

# %% [markdown]
#

# %% [markdown]
#

# %% [markdown]
# 41088 - takes every hour
#

# %%



# %%
path = 'images/panels/46066/2022_11_5_15_45/2022_11_5_15_45_panel_1.png'
print(is_it_daytime(path))
