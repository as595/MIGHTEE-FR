# -----------------------------------------------------------------------------------------
# [240620 - AMS] created
# -----------------------------------------------------------------------------------------

# general libraries
import pylab as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
import os, sys

# pytorch libraries
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

# AI4Astro libraries
from cata2data import CataData
from byol.utilities import embed_dataset
from byol.datasets import RGZ108k
from byol.datasets import MBFRFull, MBHybrid, MBFRConfident, MBFRUncertain
from byol.models import BYOL

# local libraries
from reducer import Reducer


# USER INPUTS
# -----------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------

# define paths:
image_path = '/Users/ascaife/SRC/GITHUB/MIGHTEE-FR/data/MIGHTEE_Continuum_Early_Science_COSMOS_r-1p2.app.restored.circ.fits'
catlog_path = '/Users/ascaife/SRC/GITHUB/MIGHTEE-FR/catalogue/imogen_cat.txt'

# input parameters 
imagesize = 70
field = 'COSMOS'
mu = 0.008008896
sig = 0.05303395

# other datasets:
paths={}
paths["rgz"] = "/Users/ascaife/SRC/GITHUB/_data/rgz"
paths["mb"] = "/Users/ascaife/SRC/GITHUB/_data/mb"

# model checkpoint
ckpt = '/Users/ascaife/SRC/GITHUB/byol/byol/byol.ckpt'

# umap parameters
embedding = 'rgz_embedding_15.parquet' 
train_data = None
PCA_COMPONENTS = 200
UMAP_N_NEIGHBOURS = 75
UMAP_MIN_DIST = 0.01
METRIC = "cosine"

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# MAIN CODE
# --------------------------------------------------------------------------------
# test dataset (MIGHTEE)

# create dataset:
mightee_data = CataData(
    catalogue_paths=[catlog_path],
    image_paths=[image_path],
    field_names=[field],
    cutout_width=imagesize,
)

# rename columns to define image centres:
mightee_data.df.rename(mapper={"RA_host":"ra", "DEC_host":"dec"}, axis="columns", inplace=True)

# create (test) dataloader:
dataloader = DataLoader(data, batch_size=64, shuffle=False)

# --------------------------------------------------------------------------------
# comparison datasets

transform = T.Compose(
    [
        T.CenterCrop(imagesize),
        T.ToTensor(),
        T.Normalize((mu,), (sig,)),
    ]
)

rgz = RGZ108k(
    paths["rgz"],
    train=True,
    transform=transform,
    download=False,
    remove_duplicates=False,
    cut_threshold=20,
    mb_cut=True,
    )

mb = MBFRFull(paths["mb"], 
              train=True, 
              transform=transform, 
              download=False, 
              aug_type="torchvision"
             )

# --------------------------------------------------------------------------------
# import model

# load model from checkpoint
byol = BYOL.load_from_checkpoint(ckpt)
byol.eval()

# separate encoder
encoder = byol.encoder
encoder.eval()

config = byol.config

# -----------------------------------------------------------------------------------------
# visualise embeddings

reducer = Reducer(encoder, PCA_COMPONENTS, UMAP_N_NEIGHBOURS, UMAP_MIN_DIST, METRIC, embedding=embedding)
reducer.fit(train_data)

rgz_umap = reducer.transform(rgz)
mb_umap = reducer.transform(mb)
mightee_umap = reducer.transform(mightee_data)

fig, ax = pl.subplots()
#fig.set_size_inches(fig_size)

ax.scatter(rgz_umap[:, 0], rgz_umap[:, 1], label="RGZ DR1", marker=marker, s=marker_size, alpha=alpha)
ax.scatter(mb_umap[:, 0], mb_umap[:, 1], label="MiraBest", marker=marker, s=marker_size, alpha=alpha)
ax.scatter(mightee_umap[:, 0], mightee_umap[:, 1], label="MIGHTEE", marker=marker, s=marker_size, alpha=alpha)

pl.gca().set_aspect("equal", "datalim")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_xlabel("umap x", fontsize=fontsize)
ax.set_ylabel("umap y", fontsize=fontsize)
ax.legend(fontsize=fontsize, markerscale=10)
fig.tight_layout()
fig.savefig("byol_umap_mbrgz.png", bbox_inches="tight", pad_inches=0.05, dpi=600)
