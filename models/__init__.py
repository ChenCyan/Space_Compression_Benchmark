from models import _jpeg2000

from models import cae1d, cae1dm
from models import sscnet
from models import cae3d
from models import hycot

from models import hycass

models = {
    # --- JPEG2000 ---
    "jpeg2000": _jpeg2000.JPEG2000,
    
    # --- PCA ---
    # implemented inside respective notebook

    # --- 1D-CAE ---
    "cae1d_cr004": cae1d.ConvolutionalAutoencoder1D,
    "cae1d_cr008": cae1dm.cae1d_cr008,
    "cae1d_cr016": cae1dm.cae1d_cr016,
    "cae1d_cr032": cae1dm.cae1d_cr032,

    # --- SSCNet ---
    "sscnet_cr004": sscnet.sscnet_cr004,
    "sscnet_cr008": sscnet.sscnet_cr008,
    "sscnet_cr016": sscnet.sscnet_cr016,
    "sscnet_cr032": sscnet.sscnet_cr032,
    "sscnet_cr051": sscnet.sscnet_cr051,
    "sscnet_cr101": sscnet.sscnet_cr101,
    "sscnet_cr128": sscnet.sscnet_cr128,
    "sscnet_cr185": sscnet.sscnet_cr185,
    "sscnet_cr202": sscnet.sscnet_cr202,
    "sscnet_cr269": sscnet.sscnet_cr269,
    "sscnet_cr512": sscnet.sscnet_cr512,
    "sscnet_cr1024": sscnet.sscnet_cr1024,

    # --- 3D-CAE ---
    "cae3d_cr004": cae3d.cae3d_cr004,
    "cae3d_cr008": cae3d.cae3d_cr008,
    "cae3d_cr016": cae3d.cae3d_cr016,
    "cae3d_cr032": cae3d.cae3d_cr032,
    "cae3d_cr051": cae3d.cae3d_cr051,
    "cae3d_cr064": cae3d.cae3d_cr064,
    "cae3d_cr127": cae3d.cae3d_cr127,
    "cae3d_cr253": cae3d.cae3d_cr253,

    # --- HYCOT ---
    "hycot_cr004": hycot.hycot_cr004,
    "hycot_cr008": hycot.hycot_cr008,
    "hycot_cr016": hycot.hycot_cr016,
    "hycot_cr032": hycot.hycot_cr032,
    "hycot_cr064": hycot.hycot_cr064,
    "hycot_cr128": hycot.hycot_cr128,
    "hycot_cr256": hycot.hycot_cr256,
    "hycot_cr512": hycot.hycot_cr512,

    # --- HYCASS ---
    "hycass_cr004_spatial0x_n1024": hycass.hycass_cr004_spatial0x_n1024,
    "hycass_cr004_spatial1x_n128": hycass.hycass_cr004_spatial1x_n128,
    "hycass_cr004_spatial2x_n128": hycass.hycass_cr004_spatial2x_n128,
    "hycass_cr004_spatial3x_n128": hycass.hycass_cr004_spatial3x_n128,

    "hycass_cr008_spatial0x_n1024": hycass.hycass_cr008_spatial0x_n1024,
    "hycass_cr008_spatial1x_n128": hycass.hycass_cr008_spatial1x_n128,
    "hycass_cr008_spatial2x_n128": hycass.hycass_cr008_spatial2x_n128,
    "hycass_cr008_spatial3x_n128": hycass.hycass_cr008_spatial3x_n128,

    "hycass_cr016_spatial0x_n1024": hycass.hycass_cr016_spatial0x_n1024,
    "hycass_cr016_spatial1x_n128": hycass.hycass_cr016_spatial1x_n128,
    "hycass_cr016_spatial2x_n128": hycass.hycass_cr016_spatial2x_n128,
    "hycass_cr016_spatial3x_n128": hycass.hycass_cr016_spatial3x_n128,

    "hycass_cr032_spatial0x_n1024": hycass.hycass_cr032_spatial0x_n1024,
    "hycass_cr032_spatial1x_n128": hycass.hycass_cr032_spatial1x_n128,
    "hycass_cr032_spatial2x_n128": hycass.hycass_cr032_spatial2x_n128,
    "hycass_cr032_spatial3x_n128": hycass.hycass_cr032_spatial3x_n128,

    "hycass_cr037_spatial1x_n128": hycass.hycass_cr037_spatial1x_n128,
    "hycass_cr037_spatial2x_n128": hycass.hycass_cr037_spatial2x_n128,
    "hycass_cr037_spatial3x_n128": hycass.hycass_cr037_spatial3x_n128,

    "hycass_cr050_spatial0x_n1024": hycass.hycass_cr050_spatial0x_n1024,
    "hycass_cr050_spatial1x_n128": hycass.hycass_cr050_spatial1x_n128,
    "hycass_cr050_spatial2x_n128": hycass.hycass_cr050_spatial2x_n128,
    "hycass_cr050_spatial3x_n128": hycass.hycass_cr050_spatial3x_n128,

    "hycass_cr064_spatial0x_n1024": hycass.hycass_cr064_spatial0x_n1024,
    "hycass_cr064_spatial1x_n128": hycass.hycass_cr064_spatial1x_n128,
    "hycass_cr064_spatial2x_n128": hycass.hycass_cr064_spatial2x_n128,
    "hycass_cr064_spatial3x_n128": hycass.hycass_cr064_spatial3x_n128,

    "hycass_cr101_spatial0x_n1024": hycass.hycass_cr101_spatial0x_n1024,
    "hycass_cr101_spatial1x_n128": hycass.hycass_cr101_spatial1x_n128,
    "hycass_cr101_spatial2x_n128": hycass.hycass_cr101_spatial2x_n128,
    "hycass_cr101_spatial3x_n128": hycass.hycass_cr101_spatial3x_n128,

    "hycass_cr123_spatial0x_n1024": hycass.hycass_cr123_spatial0x_n1024,
    "hycass_cr123_spatial1x_n128": hycass.hycass_cr123_spatial1x_n128,
    "hycass_cr123_spatial2x_n128": hycass.hycass_cr123_spatial2x_n128,
    "hycass_cr123_spatial3x_n128": hycass.hycass_cr123_spatial3x_n128,

    "hycass_cr185_spatial0x_n1024": hycass.hycass_cr185_spatial0x_n1024,
    "hycass_cr185_spatial1x_n128": hycass.hycass_cr185_spatial1x_n128,
    "hycass_cr185_spatial2x_n128": hycass.hycass_cr185_spatial2x_n128,
    "hycass_cr185_spatial3x_n128": hycass.hycass_cr185_spatial3x_n128,

    "hycass_cr202_spatial0x_n1024": hycass.hycass_cr202_spatial0x_n1024,
    "hycass_cr202_spatial1x_n128": hycass.hycass_cr202_spatial1x_n128,
    "hycass_cr202_spatial2x_n128": hycass.hycass_cr202_spatial2x_n128,
    "hycass_cr202_spatial3x_n128": hycass.hycass_cr202_spatial3x_n128,

    "hycass_cr222_spatial1x_n128": hycass.hycass_cr222_spatial1x_n128,
    "hycass_cr222_spatial2x_n128": hycass.hycass_cr222_spatial2x_n128,
    "hycass_cr222_spatial3x_n128": hycass.hycass_cr222_spatial3x_n128,

    "hycass_cr369_spatial0x_n1024": hycass.hycass_cr369_spatial0x_n1024,
    "hycass_cr369_spatial1x_n128": hycass.hycass_cr369_spatial1x_n128,
    "hycass_cr369_spatial2x_n128": hycass.hycass_cr369_spatial2x_n128,
    "hycass_cr369_spatial3x_n128": hycass.hycass_cr369_spatial3x_n128,

    "hycass_cr404_spatial1x_n128": hycass.hycass_cr404_spatial1x_n128,
    "hycass_cr404_spatial2x_n128": hycass.hycass_cr404_spatial2x_n128,
    "hycass_cr404_spatial3x_n128": hycass.hycass_cr404_spatial3x_n128,

    "hycass_cr738_spatial1x_n128": hycass.hycass_cr738_spatial1x_n128,
    "hycass_cr738_spatial2x_n128": hycass.hycass_cr738_spatial2x_n128,
    "hycass_cr738_spatial3x_n128": hycass.hycass_cr738_spatial3x_n128,

    "hycass_cr808_spatial1x_n128": hycass.hycass_cr808_spatial1x_n128,
    "hycass_cr808_spatial2x_n128": hycass.hycass_cr808_spatial2x_n128,
    "hycass_cr808_spatial3x_n128": hycass.hycass_cr808_spatial3x_n128,
    "hycass_cr808_spatial4x_n128": hycass.hycass_cr808_spatial4x_n128,

    "hycass_cr888_spatial2x_n128": hycass.hycass_cr888_spatial2x_n128,
    "hycass_cr888_spatial3x_n128": hycass.hycass_cr888_spatial3x_n128,

    "hycass_cr1476_spatial1x_n128": hycass.hycass_cr1476_spatial1x_n128,
    "hycass_cr1476_spatial2x_n128": hycass.hycass_cr1476_spatial2x_n128,
    "hycass_cr1476_spatial3x_n128": hycass.hycass_cr1476_spatial3x_n128,

    "hycass_cr1776_spatial2x_n128": hycass.hycass_cr1776_spatial2x_n128,
    "hycass_cr1776_spatial3x_n128": hycass.hycass_cr1776_spatial3x_n128,

}