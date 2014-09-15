import scipy.io as io
import numpy as np
import astropy.io.fits as pf
import asciitable as at
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

def texes2fits(file):

    detector_shape = (256,256)
    data = np.fromfile(file,dtype='>i4')
    header_raw = at.read(file+'.hd',delimiter='=')
        
    header_dict = {}
    for pair in header_raw:
        key = pair[0]
        try:
            value=float(pair[1])
        except:
            value=pair[1]
            
        header_dict[key] = value    
    
    nnods = header_dict['nnod']
    try:
        cube = np.reshape(-data,(nnods*2,detector_shape[0],detector_shape[1]))
    except ValueError:
        print "Failed reading ", file
        return

    hdu = pf.PrimaryHDU(cube)
    hdu.header.extend(header_dict.items())
#    for key in header_dict.keys():
#        hdu.header[key] = header_dict[key]
    
    hdu.writeto(file+'.fits',clobber=True)
    
    print "successfully read ", file
