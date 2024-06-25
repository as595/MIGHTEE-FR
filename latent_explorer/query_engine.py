from cata2data import CataData

from astropy.io import fits
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from astropy import units as u

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T

image_path = '/Users/ascaife/SRC/GITHUB/MIGHTEE-FR/data/MIGHTEE_Continuum_Early_Science_COSMOS_r-1p2.app.restored.circ.fits'
    
def get_image(ra: float, dec: float, low: float = 0, tensor=False):

    # make temporary catalogue:
    dict = {'#ra':[ra], 'dec':[dec]}
    df = pd.DataFrame(dict)
    df.to_csv('temp.txt', sep=' ', index=False, header=True)
    
    # input parameters
    imagesize = 70
    field = 'COSMOS'

    # create dataset:
    mightee_data = CataData(
        catalogue_paths=['temp.txt'],
        image_paths=[image_path],
        field_names=[field],
        cutout_width=imagesize
    )

    img = mightee_data[0][0]
    
    return img


def array_to_png(img):
    im = Image.fromarray(img)
    im = im.convert("L")

    return im

