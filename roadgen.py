import requests
import json
import math
import os
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import itertools

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

def avg(direction):
    return ((sum([n[0] for n in direction]) / len(direction), sum([n[1] for n in direction]) / len(direction)))

class Node:
    def __init__(self, x, h, y, px, py, id, next, prev):
        self.x = x
        self.h = h + 2
        self.y = y
        
        # visualization coordinates
        self.px = px
        self.py = py

        # openstreetmap node ids
        self.id = id

        self.next = next
        self.prev = prev

        self.connected = []

        self.type = 0

class RoadSegment:
    
    id_iter = itertools.count()

    def __init__(self, s, e, roadType = 0, ps = None, ns = None):
        # vtol parameters
        self.id = next(self.id_iter)
        self.type = roadType
        self.bridge = False
        self.length = math.sqrt((s.x - e.x) ** 2 + (s.y - e.y) ** 2)
        self.s = s
        self.m = ((s.x + e.x) / 2, (s.h + e.h) / 2 + 2, (s.y + e.y) / 2)
        self.e = e
        self.ps = ps
        self.ns = ns
        
    def __str__(self):
        string = "\t\t\tSegment\n\t\t\t{\n"
        string += f"\t\t\t\tid = {self.id}\n"
        string += f"\t\t\t\ttype = {self.type}\n"
        string += f"\t\t\t\tbridge = {self.bridge}\n"
        string += f"\t\t\t\tlength = {self.length}\n"
        string += f"\t\t\t\ts = ({self.s.x}, {self.s.h}, {self.s.y})\n"
        string += f"\t\t\t\tm = ({self.m[0]}, {self.m[1]}, {self.m[2]})\n"
        string += f"\t\t\t\te = ({self.e.x}, {self.e.h}, {self.e.y})\n"
        if self.ps is not None:
            string += f"\t\t\t\tps = {self.ps.id}\n"
        if self.ns is not None:
            string += f"\t\t\t\tns = {self.ns.id}\n"
        string += "\t\t\t}\n"
        return string
        
    def chunk(self):
        return (int(self.m[0] // 3072), int(self.m[2] // 3072))
    

    
def generateRoads(map_id, lat, lon, size, chunks, heights):
    # (min_lat, min_lon, max_lat, max_lon)
    map_bbox = (
        lat - km_to_deg_lat(size / 2),
        lon - km_to_deg_lon(size / 2, lat),
        lat + km_to_deg_lat(size / 2),
        lon + km_to_deg_lon(size / 2, lat)
    )

    road_data = fetch_roads(map_bbox, map_id)

    nodes = []
    head_nodes = []

    for way in road_data["elements"]:
        if "tags" not in way:
            continue
        elif check_tags(way["tags"]):
            i = 0
            for coords in way["geometry"]:
                x = lon_to_px(coords["lon"], map_bbox, heights.shape[1])
                y = lat_to_px(coords["lat"], map_bbox, heights.shape[0])
                yorg = y
                xorg = x
                # y is actually flipped top to bottom
                y = heights.shape[0] - y

                # get the height of the terrain at this point
                height = getHeights(x, y, heights)

                # Convert to bezier in game units (1m = 3072 game units)
                # so total size is 3072*chunks units square, origin is the bottom left corner
                total_size = 3072 * chunks
                
                # scale from pixel coordinates to game units, max pixel size is height.shape[0]
                x = x / heights.shape[1] * total_size
                y = y / heights.shape[0] * total_size

                # add the node to the list
                nodes.append(Node(x, height, y, xorg, yorg, way["nodes"][i], None, None))
                
                if i > 0:
                    nodes[-2].next = nodes[-1]
                    nodes[-1].prev = nodes[-2]
                else:
                    # get the index of the tag in the tagTable
                    nodes[-1].type = tagTable["highway"].index(way["tags"]["highway"])
                    head_nodes.append(nodes[-1])
                i += 1

    segments = []
    orphans = []
    def make_segment(s, ps=None):
        # make a segment from s to e
        e = s.next
        
        direction = [(s.x, s.y), (e.x, e.y)]

        while e.next is not None:
            # if the next node is in a straight line, skip to the next node
            a = avg(direction)
            eps = 300
            if abs(a[0] - e.next.x) < eps and abs(a[1] - e.next.y) < eps:
                e = e.next
                direction.append((e.x, e.y))
            else:
                # we have reached a corner, make a segment
                segments.append(RoadSegment(s, e, s.type, ps))

                # if the end has a next, make another segment
                if e.next is not None:
                    make_segment(e, segments[-1])
                else:
                    orphans.append(e)
                return
    print(f"There are {len(orphans)} orphan nodes")

    # for node in head_nodes, try and make a segment, skipping to next if the point is in a straight line
    for node in head_nodes:
        s = node
        if node.next is None:
            print("Orphan head node")
            continue
        make_segment(s)

    for segment in segments:
        # if the segment has a ps, find and connect its ns  
        if segment.ps is not None:
            for other in segments:
                if segment.ps is other:
                    segment.ps = other
                    other.ns = segment
                    break

    # ensure all segments are connected if their start/end are close enough
    eps = 200
    for segment in segments:
        for other in segments:
            if segment is other:
                continue
            # if segment has no next segment and other has no previous segment
            if segment.ns is None and other.ps is None:
                if abs(segment.e.x - other.s.x) < eps and abs(segment.e.y - other.s.y) < eps:
                    print("Connecting segment")
                    segment.ns = other
                    other.ps = segment

                    segment.e = other.s
                    
                    # update the length of the segment
                    segment.length = math.sqrt((segment.s.x - segment.e.x) ** 2 + (segment.s.y - segment.e.y) ** 2)
                    break

            # if segment has no previous segment and other has no next segment
            if segment.ps is None and other.ns is None:
                if abs(segment.s.x - other.e.x) < eps and abs(segment.s.y - other.e.y) < eps:
                    print("Connecting segment")
                    segment.ps = other
                    other.ns = segment

                    segment.s = other.e

                    # update the length of the segment
                    segment.length = math.sqrt((segment.s.x - segment.e.x) ** 2 + (segment.s.y - segment.e.y) ** 2)
                    
                
    print(f"Generated {len(segments)} segments")
    
    # Draw over the map
    fig, ax = plt.subplots(figsize=(heights.shape[0]/100, heights.shape[1]/100))
    ax.imshow(heights, cmap='jet')
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    Path = mpath.Path

    for segment in segments:
        # draw the segment as a bezier curve
        mid = ((segment.s.px + segment.e.px) / 2, (segment.s.py + segment.e.py) / 2)
        p = mpath.Path([(segment.s.px, segment.s.py), mid, (segment.e.px, segment.e.py)], [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3])
        patch = mpatches.PathPatch(p, facecolor='none', edgecolor=(random.random(), random.random(), random.random()), lw=1)
        ax.add_patch(patch)

    plt.savefig(f"{map_id}/roads.png", bbox_inches='tight', pad_inches=0)

    # divide into chunks
    chunked_segments = {}
    for segment in segments:
        chunk = segment.chunk()
        if chunk not in chunked_segments:
            chunked_segments[chunk] = []
        chunked_segments[chunk].append(segment)
        
    # save the segments to a file
    with open(f"{map_id}/segments.txt", "w") as f:
        f.write("\tBezierRoads\n\t{\n")
        for chunk in chunked_segments:
            f.write(f"\t\tChunk\n\t\t")
            f.write("{\n\t\t\tgrid = (")
            f.write(f"{chunk[0]}, {chunk[1]}")
            f.write(")\n")
            for segment in chunked_segments[chunk]:
                f.write(str(segment))
            f.write("\t\t}\n")
        f.write("\t}\n")
    print("Saved road data to file")

    with open(f"{map_id}/{map_id}-s.vtm", "r") as f:
        # delete map_id.vtm if it exists
        if os.path.exists(f"{map_id}/{map_id}.vtm"):
            os.remove(f"{map_id}/{map_id}.vtm")
        with open(f"{map_id}/{map_id}.vtm", "w") as w:
            content = f.read()[:-2]  # Read the content and remove the last two characters
            w.write(content)  # Write the modified content back to the file
            with open(f"{map_id}/segments.txt", "r") as r:  
                w.write(r.read())  # Append the road data
                w.write('\n}')  # Add the closing bracket

    print("Appended road data to map file")

