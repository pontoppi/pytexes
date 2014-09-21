from texes2fits import *
import os

nights = ['/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug10/',
          '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug11/',
          '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug12/',
          '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug13/',
          '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug14/']
          
for night in nights:
    for file in os.listdir(night):
        if '.hd' in file:
            texes2fits(night+file[:-3])