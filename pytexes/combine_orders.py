import numpy as np
import numpy.ma as ma
import astropy.io.fits as pf
from scipy import constants
import matplotlib.pylab as plt

class CombSpec():
    def __init__(self, cal_files, write_path=None):
        self.cal_files = cal_files

        norders = len(cal_files)
        self.orders = []
        for file in cal_files:
            order = fits.getdata(file,1)
            self.orders.append(order)
            
        
        import pdb;pdb.set_trace()
        
    def correct_slope(self):
        pass