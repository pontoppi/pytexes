from texes2fits import *
import os

nights = ['/Users/pontoppi/Box Sync/TEXES_NH3_rawdata/Aug10/',
          '/Users/pontoppi/Box Sync/TEXES_NH3_rawdata/Aug11/',
          '/Users/pontoppi/Box Sync/TEXES_NH3_rawdata/Aug12/',
          '/Users/pontoppi/Box Sync/TEXES_NH3_rawdata/Aug13/',
          '/Users/pontoppi/Box Sync/TEXES_NH3_rawdata/Aug14/']
          
for night in nights:
    for file in os.listdir(night):
        if '.hd' in file:
            texes2fits(night+file[:-3])