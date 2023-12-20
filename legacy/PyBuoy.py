class Buoy(): # this is the super parent class
    def __init__(self,buoy_id="none",temperature="none"):
        self.buoy_id = buoy_id
        self.temperature = temperature

    def check_buoy(self,buoy_id):
        self.check_data = 'defaults' # put defaults into the code here

class Chosen_Buoy(Buoy):
    def __init__(self,buoy_type="none"):
        super().__init__("Chosen_Buoy")
        self.buoy_lat = 10.0
        self.buoy_lng = 10.0
        self.buoy_depth = 10.0
        self.buoy_temp = 10.0
        self.buoy_atmpressure = 10.0

    def report_out(self,chosen_metric):
        super().report_out(chosen_metric)
        # do things here

from urllib.parse import urldefrag
import urllib.request
import json

import requests
import re



import pandas as pd

def main():



    # create a matrix 36 x 36 containing buoy ids
    # https://www.ndbc.noaa.gov/buoycam.php?station=xxxxx
    # this gets most recent photo from the buoy with the code xxxxx
    # https://tidesandcurrents.noaa.gov/api-helper/url-generator.html
    #https://www.ndbc.noaa.gov/rss/ndbc_obs_search.php?lat=40N&lon=73W&radius=100

    #https://www.ndbc.noaa.gov/data/realtime2/ (station id) .swdir

    #https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20130808 15:00&end_date=20130808 15:06&station=8454000&product=water_temperature&units=english&time_zone=gmt&application=ports_screen&format=json

    #The result "V" should be the desired variable
    product = 'air_temperature'
    station_id = str(8454000)
    start_year = str(2021)
    start_month = '11'
    start_day = str('11')

    end_year = str('2021')
    end_month = str('11')
    end_day = str(start_day)

    # a day
    start_time_military = str('00:00')
    end_time_military = str('23:59')

    #TODO will have issues with days that are two digits and months with one digit as well.
    #Valid Date Format is yyyyMMdd, MM/dd/yyyy, yyyyMMdd HH:mm, or MM/dd/yyyy HH:mm"
    # begin_date=20130808 15:00&end_date=20130808 15:06
    query_date = f'begin_date={start_year}{start_month}{start_day} {start_time_military}&end_date={end_year}{end_month}{end_day} {end_time_military}'

    unit_type = 'metric'
    #urlData = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{query_date}&station={str(station_id)}&product={str(product)}&units={str(unit_type)}&time_zone=gmt&application=ports_screen&format=json'
    query_date = 'range=24'
    urlData = f'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?{query_date}&station={str(station_id)}&product={str(product)}&units={str(unit_type)}&time_zone=gmt&application=ports_screen&format=json'

    print(urlData)
    urlData = urlData.replace(" ", "%20")
    print(urlData)
    #'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20210111%2015:00&end_date=20210112%2015:10&station=8454000&product=water_temperature&units=metric&time_zone=gmt&application=ports_screen&format=json'

    #webUrl = urllib.request.urlopen(urlData)
    #webUrl = urllib.request.urlopen(urlData)
    webUrl = requests.get(urlData,auth=('user','pass'))
    code = webUrl.status_code
    theJSON = webUrl.json()
    if code == 200:
        data = webUrl.text
    else:
        data = "na"

    # now we can access the contents of the JSON like any other Python object    if "title" in theJSON['metadata']:
        print(theJSON['metadata']['title'])
        # get air_temperature


    #! method 2

    #this file is created fresh every five minutes

    urlData_obs = 'https://www.ndbc.noaa.gov/data/latest_obs/latest_obs.txt'
    webUrl_obs = requests.get(urlData_obs,auth=('user','pass'))
    print(webUrl_obs)
    fh = webUrl_obs.text
    file_string = str(fh)
    found = True
    while found:
        # eliminate extra spaces
        if file_string.find("  ")>-1:
            file_string = file_string.replace("  "," ")
        else:
            found = False
    file_string = file_string.replace(" ",",")
    #file_string = file_string.replace("\n","\n,")
    import os
    file = open("tempfile.csv",'w+')
    file.write(file_string)

    # dictionary where the lines from
    # text will be stored
    dict1 = {}
    data2 = pd.read_csv("tempfile.csv", sep=",", header=0)
    file.close()
    # creating dictionary
    import csv

    mydict = {}

    '''with open('tempfile.csv', mode='r') as inp:
        reader = csv.reader(inp)
        #print(reader)
        dict_from_csv = {rows[0]:rows[1] for rows in reader}'''

    dict_from_file = pd.read_csv('tempfile.csv', header=0, index_col=0)
    dict_from_file.head()


    result.transpose().to_dict()



    #print(dict_from_csv)

    print(dict_from_csv.keys())




    # creating json file
    # the JSON file is named test1
    out_file = open("test1.json", "w")
    json.dump(dict_from_csv, out_file, indent = 4, sort_keys = False)
    out_file.close()




if __name__ == "__main__":
    main()
