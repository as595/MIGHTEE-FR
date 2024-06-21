from typing import Any, Optional

import numpy as np
import pandas as pd
from astropy.wcs import WCS


def image_preprocessing(images):
    """Example preprocessing function for basic images.

    Args:
        images (list): list of images

    Returns:
        list: list of processed images
    """

    processed = []
    for img in images:

        # set all pixels outside a radial distance of 0.5 x image width to zero:
        width = np.rint(img.shape[0]/2)
        centre = (width, width)
        maj = width # pixels
    
        Y, X = np.ogrid[:img.shape[1], :img.shape[1]]
        dist_from_centre = np.sqrt((X - centre[0])**2 + (Y-centre[1])**2)
        mask = dist_from_centre <= maj

        img *= mask.astype(int)
        
        # nan-valued pixels set to zero:
        img[np.where(np.isnan(img))]=0.0

        # threhold at 3 x sigma_rms
        rms = 4e-6 # 4uJy/beam (due to confusion)
        img[np.where(img<=3.*rms)] = 0.0

        # rescale image
        img_max = np.max(img)
        img -= 3.*rms
        img /= (img_max - 3*rms)
        img *= 255.

        processed.append(img)

    return processed


def wcs_preprocessing(wcs, field: str):
    """Example preprocessing function for wcs (world coordinate system).

    Args:
        wcs: Input wcs.
        field (str): field name matching the respective wcs.

    Returns:
        Altered wcs.
    """
    if field in ["COSMOS"]:
        return (wcs.dropaxis(3).dropaxis(2),)
    elif field in ["XMMLSS"]:
        raise UserWarning(
            f"This may cause issues in the future. It is unclear where header would have been defined."
        )
    else:
        return wcs


def catalogue_preprocessing(
    df: pd.DataFrame, random_state: Optional[int] = None
) -> pd.DataFrame:
    """Example Function to make preselections on the catalog to specific
    sources meeting given criteria.

    Args:
        df (pd.DataFrame): Data frame containing catalogue information.
        random_state (Optional[int], optional): Random state seed. Defaults to None.

    Returns:
        pd.DataFrame: Subset catalogue.
    """
    # Only consider resolved sources
    df = df.loc[df["RESOLVED"] == 1]

    # Sort by S_INT (integrated flux)
    df = df.sort_values("S_INT", ascending=False)

    # Only consider unique islands of sources
    # df = df.groupby("ISL_ID").first()
    df = df.drop_duplicates(subset=["ISL_ID"], keep="first")

    # Sort by field
    # df = df.sort_values("field")

    return df.reset_index(drop=True)