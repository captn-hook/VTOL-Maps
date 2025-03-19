import numpy as np
from math import cos, floor, log, pi, sin, tan, tau
from PIL import Image
import os
import requests
import scipy.ndimage
import matplotlib.pyplot as plt

def getTile(x,y,z):
    data_path = f"data/{x}_{y}_{z}.png"
    data_url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

    if not os.path.isdir("data"): os.mkdir("data")

    if not os.path.isfile(data_path):
        print(f"Downloading tile... (x:{x} y:{y} z:{z})")
        r = requests.get(data_url)
        with open(data_path,'wb') as f: f.write(r.content)

    # Load data from file
    img = Image.open(data_path)
    img = np.array(img.getdata())
    
    tile = (img[:,0] * 256 + img[:,1] + img[:,2] / 256) - 32768
    tile = tile.reshape((256,256))
    return tile

# get all tiles and merge them into one tile
def getTiles(x1, y1, x2, y2, z):
    cols = []
    for x in range(x1, x2+1):
        tiles = []
        for y in range(y1, y2+1):
            tiles.append(getTile(x,y,z))
        cols.append(np.concatenate(tiles, axis=0))
    return np.concatenate(cols, axis=1)

def getRegion(x1, y1, x2, y2, z):
    scale = 2**z
    
    tile_x1 = floor(x1*scale)
    tile_y1 = floor(y1*scale)
    tile_x2 = floor(x2*scale)
    tile_y2 = floor(y2*scale)

    data = getTiles(floor(x1*scale), floor(y1*scale), floor(x2*scale), floor(y2*scale), z)

    sub_x1 = floor((x1*scale - tile_x1)*256)
    sub_y1 = floor((y1*scale - tile_y1)*256)
    sub_x2 = floor((x2*scale - tile_x2)*256)
    sub_y2 = floor((y2*scale - tile_y2)*256)

    crop_left = sub_x1
    crop_top = sub_y1
    crop_right = 255 - sub_x2
    crop_bottom = 255 - sub_y2

    data = data[crop_top:-crop_bottom, crop_left:-crop_right]
    
    return data

def showData(data):
    import matplotlib.pyplot as plt
    plt.imshow(data)
    plt.show()

# convert lat,lon to x,y in espg:3857
def project(lat, lon):
    lat_r = lat * (pi/180)
    lon_r = lon * (pi/180)
    
    x = 1/tau * (lon_r + pi)
    y = 1/tau * (pi - log(tan(pi/4 + lat_r/2)))

    return x,y

def render(data, res, map_id):

    # crop data to square
    if data.shape[0] > data.shape[1]:
        data = data[:data.shape[1],:]
    else:
        data = data[:,:data.shape[0]]

    # resize data to resolution
    data = scipy.ndimage.zoom(data, res/data.shape[0])

    # map (-80,6000) to (0,1023)
    data = ((data + 80) / 6080).clip(0,1) * 1023

    print(f"Size: {data.shape[0]}x{data.shape[1]}")
    # save heightmap for display
    fig, ax = plt.subplots(figsize=(data.shape[0]/100, data.shape[1]/100))
    ax.imshow(data, cmap='jet')
    ax.axis('off')  # Remove axes
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"{map_id}/height.png", bbox_inches='tight', pad_inches=0)
    # split data into 4 images
    img0 = (data -   0).clip(0,255).astype(np.uint8)
    img1 = (data - 256).clip(0,255).astype(np.uint8)
    img2 = (data - 512).clip(0,255).astype(np.uint8)
    img3 = (data - 768).clip(0,255).astype(np.uint8)
    
    # only use red channel
    img0 = np.stack((img0, np.zeros_like(img0), np.zeros_like(img0)), axis=2)
    img1 = np.stack((img1, np.zeros_like(img1), np.zeros_like(img1)), axis=2)
    img2 = np.stack((img2, np.zeros_like(img2), np.zeros_like(img2)), axis=2)
    img3 = np.stack((img3, np.zeros_like(img3), np.zeros_like(img3)), axis=2)
    
    # save images
    Image.fromarray(img0).save(f"{map_id}/height0.png")
    Image.fromarray(img1).save(f"{map_id}/height1.png")
    Image.fromarray(img2).save(f"{map_id}/height2.png")
    Image.fromarray(img3).save(f"{map_id}/height3.png")

    return data
    
    

def createVtm(map_id, map_edge, map_coast, map_biome, map_seed, size):
    if not os.path.isdir(map_id):
        os.mkdir(map_id)
    with open(f"{map_id}/{map_id}-s.vtm", "w", newline=None) as f:
        f.write("VTMapCustom\n{\n")
        f.write(f"\tmapID = {map_id}\n")
        f.write("\tmapName = \n\tmapDescription = \n\tmapType = HeightMap\n")
        f.write(f"\tedgeMode = {map_edge}\n")
        if map_edge == "Coast":
            f.write(f"\tcoastSide = {map_coast}\n")
        f.write(f"\tbiome = {map_biome}\n")
        f.write(f"\tseed = {map_seed}\n")
        f.write(f"\tmapSize = {size}\n")
        f.write("\tTerrainSettings\n\t{\n\t}\n}\n")

def generateMap(map_id, map_edge, map_coast, map_biome, map_seed, size, lat, lon, scale, detail):
    size_meters = size * 3072 / scale
    earth_radius = 6378137
    x,y = project(lat, lon)
    step = size_meters / (2 * pi * earth_radius * cos(lat * pi/180))

    data = getRegion(x-step/2,y-step/2,x+step/2,y+step/2,detail)
    data = data * scale
    createVtm(map_id, map_edge, map_coast, map_biome, map_seed, size)
    return render(data, size*20+1, map_id)