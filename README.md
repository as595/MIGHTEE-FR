# MIGHTEE-FR
Fanaroff Riley Classification for the MeerKAT MIGHTEE survey

---
### MIGHTEE Data

This project uses the public release of the MeerKAT MIGHTEE Survey [COSMOS Early Science image](https://archive-gw-1.kat.ac.za/public/repository/10.48479/emmd-kf31/index.html) in combination with the [Cata2Data](https://github.com/mb010/Cata2Data) library to build a test dataset based on the [source catalogue](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/mnras/516/1/10.1093_mnras_stac2140/1/stac2140_supplemental_file.txt?Expires=1721326470&Signature=0-GDWoKHy-L7oEezOjr1i4sh5VE8VbG3ougu2acVEpldMVnh8witzUa65jXKPLUQmbrPNF3xC-siUkQ5TJoFs4EV7UTrzTwYO0i13lH3RQGMBIwTYGprssidQt~azEy1yad5CV7RKQAUI-osy743YkbjWAo~VHOwcX6BvQg5QCHFFL1E0vAajqCY~v~c7oeHK0UwWmlOtEU2JIVF6VQbjibGyIeCrGvB00yu7Pp9aUFYLBRjqOLuzJsevSFJZ8fb6-yACQd~Kx0dELi7s5aKQUF9G7zFY5G6~dKiOAOmm3Ri2pcsmvctfFplTNjneCydY1~~OYVccrY9Q1wV0ZLdDw__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA) from [Whittam+ (2022)](https://arxiv.org/abs/2207.12379).

```python
from cata2data import CataData

# define paths:
image_path = '/Users/ascaife/SRC/GITHUB/MIGHTEE-FR/images/MIGHTEE_Continuum_Early_Science_COSMOS_r-1p2.app.restored.circ.fits'
catlog_path = '/Users/ascaife/SRC/GITHUB/MIGHTEE-FR/catalogue/imogen_cat.txt'

# create dataset:
mightee_data = CataData(
    catalogue_paths=[catlog_path],
    image_paths=[image_path],
    field_names=['COSMOS'],
    cutout_width=70,
)

# rename columns to define image centers:
mightee_data.df.rename(mapper={"RA_host":"ra", "DEC_host":"dec"}, axis="columns", inplace=True)

# eyeball an example:
idx = 60
mightee_data.plot(idx)
mightee_data.df.iloc[idx : idx + 1]

```
![](https://github.com/as595/MIGHTEE-FR/blob/main/images/src60.png | width=100)
<img src="[https://github.com/favicon.ico](https://github.com/as595/MIGHTEE-FR/blob/main/images/src60.png)" width="100">

---
### Classification Model

The classification model is based on the [RGZ foundation model](https://github.com/inigoval/byol) from [Slijepcevic+ (2023)](https://arxiv.org/abs/2305.16127), fine-tuned on a small number of manually classified sources from the MIGHTEE survey (see Section 3.4 of the RGZ paper).
