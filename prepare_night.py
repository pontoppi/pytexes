from texes2fits import *
import os

night = '/astro/pontoppi/DATA/TEXES/NH3_rawdata/Aug10/'

for file in os.listdir(night):
    if '.hd' in file:
        texes2fits(night+file[:-3])