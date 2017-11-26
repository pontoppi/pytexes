import os
import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import Angle
from astropy import units as u
from PyAstronomy import pyasl

from scipy import constants,polyfit,poly1d
from scipy.interpolate import interp1d
import matplotlib.pylab as plt

import utils.helpers as helpers
from pytexes.observation import Environment

class CombSpec():
    def __init__(self, cal_files, write_path=None, micron=False, revise_wave=True):
        self.envi = Environment()
        self.cal_files = cal_files
        self.norders = len(cal_files)

        self.orders = []
        for file in cal_files:
            order = fits.getdata(file,1)
            self.header = fits.getheader(file)
            self.orders.append(order)

        self.nsamp = self.orders[0]['wave'].shape[0]
            
        self.correct_slope()
        self.wave, self.flux, self.uflux, self.sky_sci, self.sky_std, self.sky_model = self.combine_orders()
        
        if micron:
            self.wave = 1e4/self.wave
            ssubs = np.argsort(self.wave)
            # We have to sort this in ascending order
            self.wave = self.wave[ssubs]
            self.flux = self.flux[ssubs]
            self.uflux = self.uflux[ssubs]
            self.sky_std = self.sky_std[ssubs]
            self.sky_sci = self.sky_sci[ssubs]
            self.sky_model = self.sky_model[ssubs]
        
        if revise_wave:
            self.revise_wavelength()
            
        if write_path:
            self.file = self.write_spec(path=write_path)
        
    def correct_slope(self,rank=4):
        
        index = np.linspace(0,self.nsamp-1,self.nsamp)
        trends = np.zeros((self.nsamp,self.norders))
        i = 0
        '''
        for order in self.orders:
            fsubs = np.isfinite(order['flux'])
            pars = polyfit(index[fsubs][20:-20],order['flux'][fsubs][20:-20],deg=rank,w=order['uflux'][fsubs][20:-20])
            trend = poly1d(pars)
            trend_samp = trend(index)
            
            trends[:,i] = trend_samp/np.median(trend_samp)    
            i += 1

        trend_mean = np.median(trends,axis=1)

        '''
        
        fluxes = np.zeros((self.nsamp,self.norders))
        for i in np.arange(self.norders):
            flux = self.orders[i]['flux']
            flux = flux/np.median(flux)    
            fluxes[:,i] = flux
            i += 1
            
        trend_mean = np.nanmedian(fluxes,axis=1)
        fsubs = np.isfinite(trend_mean)
        pars = polyfit(index[fsubs],trend_mean[fsubs],deg=rank)
        trend_smooth = poly1d(pars)(index)
        
        for order in self.orders:
            order['flux'] /= trend_smooth

    def combine_orders(self):
        waves = np.zeros((self.nsamp,self.norders))
        minwaves = np.zeros(self.norders)
        maxwaves = np.zeros(self.norders)

        i = 0        
        for order in self.orders:
            waves[:,i] = order['wave']
            minwaves[i] = np.min(waves[:,i])
            maxwaves[i] = np.max(waves[:,i])
            i = i+1
        
        minwave_master = np.min(minwaves)
        maxwave_master = np.max(minwaves)

        delta   = np.median(waves-np.roll(waves,1,axis=0))
        wave_master = np.arange(minwave_master,maxwave_master,delta)
        flux_master = np.zeros(wave_master.shape[0])
        uflux_master = np.zeros(wave_master.shape[0])
        sky_sci_master = np.zeros(wave_master.shape[0])
        sky_std_master = np.zeros(wave_master.shape[0])
        nspec_master = np.zeros(wave_master.shape[0])
        
        for i in np.arange(self.norders):
            order = self.orders[i]
            minwave = minwaves[i]
            maxwave = maxwaves[i]
            gsubs = np.where((wave_master>minwave) & (wave_master<=maxwave))
            flux_master[gsubs] += interp1d(order['wave'],order['flux'])(wave_master[gsubs])
            uflux_master[gsubs] += interp1d(order['wave'],order['uflux'])(wave_master[gsubs])
            sky_sci_master[gsubs] += interp1d(order['wave'],order['sky_sci'])(wave_master[gsubs])
            sky_std_master[gsubs] += interp1d(order['wave'],order['sky_std'])(wave_master[gsubs])
            nspec_master[gsubs] += 1
            
        flux_master /= nspec_master
        uflux_master /= nspec_master
        sky_sci_master /= nspec_master
        sky_std_master /= nspec_master
        
        modelsky = self.get_modelsky(wave_master)
        
        return wave_master, flux_master, uflux_master, sky_sci_master, sky_std_master, modelsky['irr']

    def get_modelsky(self,wave_master):
        sys_dir = self.envi._getSysPath()
        
        modelsky_file = self.envi.getRefFiles()['skyspec']
        modelsky_data = fits.getdata(os.path.join(sys_dir,modelsky_file),1)
        wave = 1e4/modelsky_data['wave'].flatten()
        irr = modelsky_data['irr'].flatten()
        #ssubs = np.argsort(wave)
        #wave = wave[ssubs]
        #irr = irr[ssubs]
        irr_int = np.interp(wave_master,wave,irr)
        
        #import pdb;pdb.set_trace()
        modelsky = {'wave':wave_master,'irr':irr_int}
        
        return modelsky

    def revise_wavelength(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(self.sky_sci)
        ylim = ax.get_ylim()
        
        xpos = []
        def onclick(event):
            
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                 ('double' if event.dblclick else 'single', event.button,
                 event.x, event.y, event.xdata, event.ydata))
            xpos.append(event.xdata)
            ax.plot([event.xdata,event.xdata],ylim)
            plt.draw()
            
        cid = fig.canvas.mpl_connect('button_press_event', onclick)        
        plt.show(1)
        cid = fig.canvas.mpl_disconnect(cid)
        pixelpos = np.array(xpos)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.plot(self.wave, self.sky_model)
        ylim = ax.get_ylim()
        
        for pixel in pixelpos:
            pass
                
        xpos = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)        
        plt.show(1)
        cid = fig.canvas.mpl_disconnect(cid)
        
        wavepos = np.array(xpos)
        
        coeffs = np.polyfit(pixelpos,wavepos,1)
        
        new_wave_func = np.poly1d(coeffs)
        
        self.new_wave = new_wave_func(np.arange(self.wave.size))
        
        # Remember to sample the sky model on the new wavelength grid
        self.sky_model = np.interp(self.new_wave,self.wave,self.sky_model)
        self.wave = self.new_wave
        
        plt.plot(self.wave,self.sky_std/np.nanmedian(self.sky_std))
        plt.plot(self.wave,self.sky_model/np.nanmedian(self.sky_model))
        plt.show()
        
        # Calculate the barycentric correction 
        lat = 19.8238
        lon = 155.4689
        alt = 4213.0
        jd = Time(self.header['DATE-OBS']).jd
        ra = Angle(self.header['TARGRA'], unit='hourangle').degree
        dec = Angle(self.header['TARGDEC'], unit=u.deg).degree
        
        corr, hjd = pyasl.helcorr(lon, lat, alt, \
                    ra, dec, jd)
                
        cc = 299792.458 #km/s       
        self.wave *= (1.0 + corr/cc)    
                
    def write_spec(self, filename=None, path='.'):
        c1  = fits.Column(name='wave', format='E', array=self.wave)
        c2  = fits.Column(name='flux', format='E', array=self.flux)
        c3  = fits.Column(name='uflux', format='E', array=self.uflux)
        c4  = fits.Column(name='sky_sci', format='E', array=self.sky_sci)
        c5  = fits.Column(name='sky_std', format='E', array=self.sky_std)
        c6  = fits.Column(name='sky_model', format='E', array=self.sky_model)

        coldefs = fits.ColDefs([c1,c2,c3,c4,c5,c6])

        tbhdu = fits.BinTableHDU.from_columns(coldefs)
        hdu = fits.PrimaryHDU(header=self.header)
        thdulist = fits.HDUList([hdu,tbhdu])

        if filename is None:
            basename = helpers.getBaseName(self.header)
            filename = path+'/'+basename+'_combspec.fits'

        thdulist.writeto(filename,overwrite=True)

        return filename  