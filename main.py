import mapgen
import roadgen
import os
import numpy as np
# This program generates a VTOL VR map from real world elevation data
# data provided by https://www.mapzen.com/blog/elevation/
# Edit these settings to customize the map

# ID of the map
map_id = "bend"

# What to generate at the edge of the map
# "Water", "Hills", "Coast"
map_edge = "Hills"

# Side of the map the coast is on
# only used if edge mode is "Coast"
# "North", "South", "East", "West"
map_coast = "West"

# Biome of the map
# "Boreal", "Desert", "Arctic"
map_biome = "Boreal"

# Seed for the map
map_seed = "seed"

# Coordinates for the center of the map
lat, lon = 44.0582, -121.3153

# VTOL VR map size (8-64)
# each unit is ~3km
size = 32

# Scaling factor
scale = 1

# Detail level (0-14)
# for low quality previews use 8
# for very high quality use 11
detail = 14

def main():
    if not os.path.exists(map_id):
        os.makedirs(map_id)
    if os.path.exists(f"{map_id}/heights.npy"):
        heights = np.load(f"{map_id}/heights.npy")
    else:
        heights = mapgen.generateMap(map_id, map_edge, map_coast, map_biome, map_seed, size, lat, lon, scale, detail)
        np.save(f"{map_id}/heights.npy", heights)

    roadgen.generateRoads(map_id, lat, lon, size * 3.072, size, heights)

if __name__ == "__main__":
    main()