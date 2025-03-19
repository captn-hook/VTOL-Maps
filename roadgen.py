import requests
import json
import math
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

tagTable = {
    "highway":['secondary', 'trunk'],
} 
# 'secondary_link', 'primary_link', 'trunk_link', 'motorway_link', 'motorway', 'tertiary_link', 'secondary', 'tertiary', 'trunk', 'primary'
colorTable = [(255, 0), (0, 255)]


def fetch_roads(bbox, map_id):
    """
    Fetch road data from the Overpass API within the specified bounding box.

    Parameters:
    bbox (tuple): A tuple containing the bounding box coordinates (min_lat, min_lon, max_lat, max_lon).

    Returns:
    dict: The JSON response from the Overpass API containing the road data.
    """
    if not os.path.exists(map_id):
        os.makedirs(map_id)
    elif os.path.exists(f"{map_id}/roads.json"):
        with open(f"{map_id}/roads.json", "r") as f:
            return json.load(f)

    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [bbox:{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}]
    [out:json]
    [timeout:90];
    (
        way
        (
            {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}
        );
    );
    out geom;
    """
    response = requests.post(overpass_url, data={"data": overpass_query})
    response.raise_for_status()  # Raise an exception for HTTP errors

    # save to /map_id/roads.json
    with open(f"{map_id}/roads.json", "w") as f:
        f.write(response.text)
    return response.json()

def km_to_deg_lat(km):
    return km / 111.0

def km_to_deg_lon(km, latitude):
    return km / (111.0 * abs(math.cos(math.radians(latitude))))

def deg_to_km_lat(deg):
    return deg * 111.0

def deg_to_km_lon(deg, latitude):
    return deg * 111.0 * abs(math.cos(math.radians(latitude)))

def check_tags(tags):
    # if key matches and value is in tagTable[key]
    for key in tagTable:
        if key in tags and tags[key] in tagTable[key]:
            return True
    return False

def lat_to_px(lat, bbox, height):
    return int((bbox[2] - lat) / (bbox[2] - bbox[0]) * height)

def lon_to_px(lon, bbox, width):
    return int((lon - bbox[1]) / (bbox[3] - bbox[1]) * width)

def getHeights(x, y, heights):
    # clamp to the bounds of the heightmap
    x, y = int(x), int(y)
    x = max(0, min(x, heights.shape[1] - 1))
    y = max(0, min(y, heights.shape[0] - 1))
    return heights[y, x] * 3.072 + 4

def generateRoads(map_id, lat, lon, size, chunks, heights):

    """
    Generate road data for the specified map.

    Parameters:
    map_id (str): The ID of the map.
    lat (float): The latitude of the center of the map.
    lon (float): The longitude of the center of the map.
    size (int): The size of the map.
    scale (int): The scaling factor.
    detail (int): The detail level.

    Returns:
    None
    """
    # (min_lat, min_lon, max_lat, max_lon)
    map_bbox = (
        lat - km_to_deg_lat(size / 2),
        lon - km_to_deg_lon(size / 2, lat),
        lat + km_to_deg_lat(size / 2),
        lon + km_to_deg_lon(size / 2, lat)
    )

    road_data = fetch_roads(map_bbox, map_id)

    

    segments = []

    j = 0
    for way in road_data["elements"]:
        j += 1
        if "tags" not in way:
            continue
        elif check_tags(way["tags"]):
            points = []
            last = None
            rcolor = None
            direction = []
            for coords in way["geometry"]:
                x = lon_to_px(coords["lon"], map_bbox, heights.shape[1])
                y = lat_to_px(coords["lat"], map_bbox, heights.shape[0])
                yorg = y
                # y is actually flipped top to bottom
                y = heights.shape[0] - y
                points.append((x, y))

                if last is None:
                    last = (x, y, yorg)
                else:
                    if len(direction) == 0:
                        direction.append((x - last[0], y - last[1]))
                        rcolor = random.randint(0, 255)
                    else:
                        # if the direction changes, refresh last
                        avgDir = (sum([d[0] for d in direction]) / len(direction), sum([d[1] for d in direction]) / len(direction))
                        curDir = (x - last[0], y - last[1])

                        # if we are going straight, add to the direction list and go to next node
                        eps = 8
                        if abs(avgDir[0] - curDir[0]) < eps and abs(avgDir[1] - curDir[1]) < eps:
                            # go to next node
                            direction.append((x - last[0], y - last[1]))
                        else:
                            # finish the segment
                            h1 = getHeights(last[0], last[1], heights)
                            h2 = getHeights(x, y, heights)                 
                            c = tagTable["highway"].index(way["tags"]["highway"])           
                            segments.append({
                                "s": (last[0], h1, last[1]),
                                "e": (x, h2, y),
                                "sd": (last[0], last[2]),
                                "ed": (x, yorg),
                                "type": c,
                                "color": (colorTable[c][0], colorTable[c][1], rcolor),
                                "ide": j
                            })
                            direction = []
                            last = (x, y, yorg)
            
    # Draw over the map
    fig, ax = plt.subplots(figsize=(heights.shape[0]/100, heights.shape[1]/100))
    ax.imshow(heights, cmap='jet')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    Path = mpath.Path

    i = 0
    chunksf = {}
    last_segment = None
    for segment in segments:
        i += 1

        # Draw the road on the map
        m = ((segment["sd"][0] + segment["ed"][0]) / 2, (segment["sd"][1] + segment["ed"][1]) / 2)
        p = mpatches.PathPatch(
            Path([(segment["sd"][0], segment["sd"][1]), m, (segment["ed"][0], segment["ed"][1])], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
            fc='none', transform=ax.transData, edgecolor=(segment["color"][0] / 255, segment["color"][1] / 255, segment["color"][2] / 255), lw=1
        )
        ax.add_patch(p)


        # Convert to bezier in game units (1m = 3072 game units)
        # so total size is 3072*chunks units square, origin is the bottom left corner
        total_size = 3072 * chunks
        
        # scale from pixel coordinates to game units, max pixel size is height.shape[0]
        s = (segment["s"][0] / heights.shape[0] * total_size, segment["s"][1], segment["s"][2] / heights.shape[0] * total_size)
        e = (segment["e"][0] / heights.shape[0] * total_size, segment["e"][1], segment["e"][2] / heights.shape[0] * total_size)

        # middle point is the average of the start and end points
        m = ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2, (s[2] + e[2]) / 2)        

        length = math.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2 + (s[2] - e[2]) ** 2)

        # figure out which chunks this segment is in
        cx = int(m[0] // 3072)
        cy = int(m[2] // 3072)


        segmentg = {
            "id": i,
            "ide": segment["ide"],
            "type": segment["type"],
            "bridge": False,
            "length": length,
            "s": s,
            "m": m,
            "e": e,
        }

        # check for connections if ides are the same
        if last_segment is not None and last_segment[1]["ide"] == segmentg["ide"]:
            # determine if they already start/end near the same point
            dist = math.sqrt((last_segment[1]["e"][0] - segmentg["s"][0]) ** 2 + (last_segment[1]["e"][2] - segmentg["s"][2]) ** 2)
                
            if dist < 200:
                segmentg["s"] = last_segment[1]["e"]
                last_segment[1]["ns"] = segmentg["id"]
                segmentg["ps"] = last_segment[1]["id"]

                
                # if there is a connection, draw a circle
                p = mpatches.Circle((segment["sd"][0], segment["sd"][1]), 3, fc=(segment["color"][0] / 255, segment["color"][1] / 255, segment["color"][2] / 255), ec='none', transform=ax.transData)
                ax.add_patch(p)
        

        if (cx, cy) not in chunksf:
            chunksf[(cx, cy)] = []
            
        if last_segment is not None:
            chunksf[last_segment[0]].append(last_segment[1])

        last_segment = ((cx, cy), segmentg)
    chunksf[last_segment[0]].append(last_segment[1])

    print(f"Generated {i} segments")
    print(f"Generated {len(chunksf)} chunks")
    plt.savefig(f"{map_id}/roads.png", bbox_inches='tight', pad_inches=0)

    

    with open(f"{map_id}/segments.txt", "w") as f:
        # header:
        f.write('\tBezierRoads\n\t{\n')
        
        for chunk in chunksf:
            f.write('\t\tChunk\n\t\t{\n\t\t\t')
            f.write(f'grid = ({chunk[0]}, {chunk[1]})\n')

            for segment in chunksf[chunk]:
                f.write('\t\t\tSegment\n\t\t\t{\n')
                f.write(f'\t\t\t\tid = {segment["id"]}\n')
                f.write(f'\t\t\t\ttype = {segment["type"]}\n')
                f.write(f'\t\t\t\tbridge = {segment["bridge"]}\n')
                f.write(f'\t\t\t\tlength = {segment["length"]}\n')
                f.write(f'\t\t\t\ts = ({segment["s"][0]}, {segment["s"][1]}, {segment["s"][2]})\n')
                f.write(f'\t\t\t\tm = ({segment["m"][0]}, {segment["m"][1]}, {segment["m"][2]})\n')
                f.write(f'\t\t\t\te = ({segment["e"][0]}, {segment["e"][1]}, {segment["e"][2]})\n')

                if "ps" in segment:
                    f.write(f'\t\t\t\tps = {segment["ps"]}\n')
                if "ns" in segment:
                    f.write(f'\t\t\t\tns = {segment["ns"]}\n')

                f.write('\t\t\t}\n')
                
            f.write('\t\t}\n')
        f.write('\t}')

    print("Saved road data to segments.txt")
        
    with open(f"{map_id}/{map_id}.vtm", "r+") as f:
        content = f.read()[:-2]  # Read the content and remove the last two characters
        f.seek(0)  # Move the cursor to the beginning of the file
        f.truncate()  # Clear the file content
        f.write(content)  # Write the modified content back to the file
        with open(f"{map_id}/segments.txt", "r") as r:  
            f.write(r.read())  # Append the road data
            f.write('\n}')  # Add the closing bracket

    print("Appended road data to map file")

