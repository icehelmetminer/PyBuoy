'''Artem Russian Federation, http://www.insecam.org/en/view/892250/, +10:00
Launceston, Australia, http://45.248.50.164/view/viewer_index.shtml?id=4793, +11:00
Launceston, Australia, http://45.248.50.164:82/view/viewer_index.shtml?id=1231



+12:00

http://www.insecam.org/en/view/760911/
http://www.insecam.org/en/view/760913/
http://www.insecam.org/en/view/762012/
http://www.insecam.org/en/view/802079/
http://www.insecam.org/en/view/813107/
http://www.insecam.org/en/view/852204/


+13:00

http://www.insecam.org/en/view/1007354/
http://www.insecam.org/en/view/897822/
http://www.insecam.org/en/view/898463/
http://www.insecam.org/en/view/906144/
http://www.insecam.org/en/view/908954/
http://www.insecam.org/en/view/974484/
http://www.insecam.org/en/view/427325/
http://www.insecam.org/en/view/451528/
http://www.insecam.org/en/view/634466/
http://www.insecam.org/en/view/811693/
http://www.insecam.org/en/view/817960/
http://www.insecam.org/en/view/846271/
http://www.insecam.org/en/view/328598/

- could be anywhere in the world, but I think they are in the US
http://www.insecam.org/en/view/1007869/
http://www.insecam.org/en/view/881796/
http://www.insecam.org/en/view/882406/
http://www.insecam.org/en/view/882407/
http://www.insecam.org/en/view/882408/
http://www.insecam.org/en/view/884495/
http://www.insecam.org/en/view/869602/
http://www.insecam.org/en/view/870260/
http://www.insecam.org/en/view/877974/
http://www.insecam.org/en/view/879044/
http://www.insecam.org/en/view/881792/
http://www.insecam.org/en/view/881793/
http://www.insecam.org/en/view/798681/
http://www.insecam.org/en/view/809558/
http://www.insecam.org/en/view/817795/
http://www.insecam.org/en/view/851953/
http://www.insecam.org/en/view/864087/
http://www.insecam.org/en/view/869601/
http://www.insecam.org/en/view/635068/
http://www.insecam.org/en/view/635294/
http://www.insecam.org/en/view/724570/
http://www.insecam.org/en/view/724594/
http://www.insecam.org/en/view/742881/
http://www.insecam.org/en/view/746928/'''

import pandas as pd

"""get http://www.insecam.org/en/bytimezone/+01:00/

find all links on page with 'view' in them and save them to the dataframe under column 'url' and with '+01:00' in the 'timezone' column
save one row per link.

then do the same for all timezones.
http://www.insecam.org/en/bytimezone/{timezonecode}/

the element you want to target within each camera page is the id='stream'
the url for the stream is in the src attribute of the element.




```html
<video>
 <source src="http://80.56.142.202:83/mjpg/video.mjpg">
</video>
<a rel="nofollow" href="http://80.56.142.202:83/" target="new"><img id="image0" src="http://80.56.142.202:83/mjpg/video.mjpg" class="img-responsive img-rounded detailimage" alt="" title="Click here to enter the camera located in Netherlands, region Noord-Brabant, Эйндховен"></a>

<script>
var ch = 0;
var imageurls = new Array();
imageurls[0] = new String("http://80.56.142.202:83/mjpg/video.mjpg");
</script>
<script async="" src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js"></script>
<ins class="adsbygoogle" style="display:block; text-align:center;" data-ad-layout="in-article" data-ad-format="fluid" data-ad-client="ca-pub-9642036526375612" data-ad-slot="8439664353"></ins>
<script>
     (adsbygoogle = window.adsbygoogle || []).push({});
</script>
```

So, if you find the `nofollow` a rel and retrieve the href attribute, you will get the url for the stream.
The `title` attribute contains the location of the camera which is also useful to keep in the dataframe.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
from random import randint
import ratelimit
from ratelimit import limits, sleep_and_retry


# Step One - get all the links from the pages for each timezone by scraping the html from the pages.

cameras_df = pd.DataFrame(columns=['url', 'timezone', 'location'])

# what else could I get from the page? location, time, etc. Within the html of the `view` page there is a table in a the div with class'camera-details' which contains the location, time, etc. I could scrape that and add it to the dataframe. This includes Country, Country Code, Region, City, Latitude, Longitude, Zip, and Timezone.

# the css selector for this table is '.camera-details'

# Step One A. Request the html of each timezone page and scrape the links from the html.

# The timezones are: +01:00, +02:00, +03:00, +04:00, +05:00, +06:00, +07:00, +08:00, +09:00, +10:00, +11:00, +12:00, -01:00, -02:00, -03:00, -04:00, -05:00, -06:00, -07:00, -08:00, -09:00, -10:00, -11:00, -12:00, +13:00
timezones = ['+01:00', '+02:00', '+03:00', '+04:00', '+05:00', '+06:00', '+07:00', '+08:00', '+09:00', '+10:00', '+11:00', '+12:00', '-01:00', '-02:00', '-03:00', '-04:00', '-05:00', '-06:00', '-07:00', '-08:00', '-09:00', '-10:00', '-11:00', '-12:00', '+13:00']

# make a function to perform the scraping so we can use it for each timezone and use ratelimit to limit the number of requests per second.
@sleep_and_retry
@limits(calls=20, period=600)
def scrape_links(timezone, page):
    # get the html of the page
    url = f'http://www.insecam.org/en/bytimezone/{timezone}/?page={page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # find all the links with 'view' in them
    links = soup.find_all('a', href=re.compile('view'))
    # add the links to the dataframe
    for link in links:
        cameras_df.loc[len(cameras_df)] = [link['href'], timezone, link['title']]
    # sleep for a random amount of time between 1 and 5 seconds to avoid being blocked
    time.sleep(randint(1,5))
    return cameras_df

# Step One B. Loop through each timezone and scrape the links from the html of each page.
for timezone in timezones:
    # get the html of the first page
    page = 1
    url = f'http://www.insecam.org/en/bytimezone/{timezone}/?page={page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # find all the links with 'view' in them
    links = soup.find_all('a', href=re.compile('view'))
    # add the links to the dataframe
    for link in links:
        cameras_df.loc[len(cameras_df)] = [link['href'], timezone, link['title']]
    # get the number of pages in the timezone
    pages = soup.find('ul', class_='pagination')
    pages = pages.find_all('li')
    pages = pages[-2]
    pages = pages.find('a')
    pages = pages['href']
    pages = re.findall(r'\d+', pages)[0]
    pages = int(pages)
    # loop through each page and scrape the links
    for page in range(2, pages+1):
        scrape_links(timezone, page)
