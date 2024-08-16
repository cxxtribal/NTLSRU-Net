import os
import numpy as np
import datetime
from datetime import timedelta
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib as mlp
from osgeo import gdal
from matplotlib.colors import Normalize
import matplotlib.colors as colors

para = 'WJN'
filePath = 'E:/'+para+'cv1_0110_fill_smooth/'
startYear = 2013
endYear = 2019
OutputDirectory = filePath + 'output/'
rgb = ([0, 208, 246], [0, 165, 215], [147, 241, 8], [96, 210, 34], [255, 255, 0], [255, 197, 0], [255, 160, 0],
       [255, 125, 0], [255, 0, 0], [159, 0, 0])
rgb = np.array(rgb) / 255.0
cmap = colors.ListedColormap(rgb, name='my_color')
# cbticks = np.arange(0, 1.01, 0.1)
# cbticks = np.arange(0, 0.0601, 0.006)
vmin = 0
if para == 'WJN':
    vmax = 1
    cbticks = np.arange(0, 1.01, 0.1)
else:
    vmax = 0.06
    cbticks = np.arange(0, 0.0601, 0.006)
norm = Normalize(vmin=vmin, vmax=vmax)
extent = (0, 1, 0, 1)

for i in range(startYear, endYear):
    curPath = filePath + str(i) + '/'
    for lists in os.listdir(curPath):
        filename = lists
        outputFullPath = OutputDirectory+ str(i)
        if not os.path.exists(outputFullPath):
            os.makedirs(outputFullPath)
        outputFullPath = outputFullPath+'/'+filename[0:8]+'.png'
        if os.path.exists(outputFullPath):
            continue
        tifpath = os.path.join(curPath, filename)
        tifdata01 = gdal.Open(tifpath)
        # tif01 = r"E:\WJNcv1_0110\2015\20150513.tif"
        # tifdata01 = gdal.Open(tif01)
        # rows = tifdata01.RasterXSize
        # columns = tifdata01.RasterYSize
        # bands = tifdata01.RasterCount
        # img_geotrans = tifdata01.GetGeoTransform()
        # img_proj = tifdata01.GetProjection()
        tif_data01 = tifdata01.ReadAsArray()
        tif_data01[tif_data01 == -9999] = np.NAN
        # vmin = np.nanmin(tif_data01)
        # vmax = np.nanmax(tif_data01)
        fig, ax = plt.subplots(1, 1, figsize=(4.2, 4.2), sharey=True)
        im1 = ax.imshow(tif_data01, extent=extent, norm=norm, cmap=cmap)
        ax.set_axis_off()
        position = fig.add_axes([0.98, 0.15, 0.015, 0.7])
        cb = fig.colorbar(im1, cax=position)
        # cbticks=np.array(cbticks)
        cb.set_ticks(cbticks)
        colorbarfontdict = {"size": 4.5, "color": cmap, 'family': 'Times New Roman'}
        # cb.ax.set_title('Values',fontdict=colorbarfontdict,pad=8)
        # cb.ax.set_ylabel('EvapotTranspiration(ET)',fontdict=colorbarfontdict)
        # cb.ax.tick_params(labelsize=11, direction='in')
        # cb.ax.set_yticklabels(['0','10','20','30','40','50','>60'],family='Times New Roman')
        # fig.suptitle('One Colorbar for Multiple Map Plot ',size=22,family='Times New Roman', x=.55,y=.9)
        plt.savefig(outputFullPath, dpi=300, bbox_inches='tight', width=4, height=4)
        plt.close("all")
        # plt.show()



