The road data in this file is structured in a hierarchical format, resembling JSON but using a custom syntax. Here's a breakdown of the structure:

1. **BezierRoads**: The top-level container for all road data.
2. **Chunk**: Represents a section of the road grid. Each chunk has a `grid` attribute indicating its position.
3. **Segment**: Represents a segment of the road within a chunk. Each segment has several attributes:
   - `id`: Unique identifier for the segment.
   - `type`: Type of the segment (e.g., road type).
   - `bridge`: Boolean indicating if the segment is a bridge.
   - `length`: Length of the segment.
   - `s`, `m`, `e`: Coordinates for the start, middle, and end points of the segment.
   - `ps`, `ns`: Previous and next segment identifiers (optional).
   - `si`, `ei`: Start and end intersection identifiers (optional).

4. **Intersection**: Represents intersections where multiple segments meet. Each intersection has:
   - `id`: Unique identifier for the intersection.
   - `segments`: List of segment IDs that connect at this intersection.

### Example Breakdown

```plaintext
Segment
{
	id = 0
	type = 0
	bridge = True
	length = 361.0468
	s = (101483.60388183594, 1476.9443359375, 90746.13037109375)
	m = (101703.23972553594, 1473.2765632278285, 90810.491886982214)
	e = (101830.01022571904, 1471.2211917049251, 90847.74090279032)
	ps = 3
	ns = 1
}
```

- **id**: 0
- **type**: 0 (road type)
- **bridge**: True (this segment is a bridge)
- **length**: 361.0468 units
- **s**: Start coordinates (x, y, z)
- **m**: Middle coordinates (x, y, z)
- **e**: End coordinates (x, y, z)
- **ps**: Previous segment ID (3)
- **ns**: Next segment ID (1)

This hierarchical structure allows for detailed representation of road networks, including segments, their connections, and intersections.

# 
```
var result = await fetch(
    "https://overpass-api.de/api/interpreter",
    {
        method: "POST",
        // The body contains the query
        // to understand the query language see "The Programmatic Query Language" on
        // https://wiki.openstreetmap.org/wiki/Overpass_API#The_Programmatic_Query_Language_(OverpassQL)
        body: "data="+ encodeURIComponent(`
            [bbox:30.618338,-96.323712,30.591028,-96.330826]
            [out:json]
            [timeout:90]
            ;
            (
                way
                    (
                         30.626917110746,
                         -96.348809105664,
                         30.634468750236,
                         -96.339893442898
                     );
            );
            out geom;
        `)
    },
).then(
    (data)=>data.json()
)

console.log(JSON.stringify(result , null, 2))
```

# Nodes, Ways, Relations
OpenStreetMap has three types of objects. Every object can carry an arbitrary number of tags. Also, every object has an id. The combination of type and id is unique, but the id alone is not.

Nodes are defined as a coordinate in addition to the id and tags. A node can represent a point of interest, or an object of minuscule extent. Because nodes are the only type of object that has a coordinate, most of the nodes serve only as a coordinate for an intermediate point within a way and carry no tags.

Ways consist of a sequence of references to nodes in addition to the id and tags. In this manner ways get a geometry by using the coordinates of the referenced nodes. But they also have a topology: two ways are connected if both point at a position to the same node.

Ways can refer to the same node multiple times. The common case for this is a closed way where the first and last entry point to the same node. All other cases are syntactically correct but semantically deprecated.

Relations have a sequence of members in addition to the id and tags. Each member is a pair of a reference to a node, a way or a relation and a so-called role. The role is a text string. Relations were invented to represent turn restrictions and these have few required members. They now also serve as boundaries of countries, counties, multipolygons, and routes. Therefore, their formal structure varies wildly, and, for example, boundary and route relations can extend over hundreds or thousands of kilometers.

Relations only have geometries if a data user interprets them to have geometries. A relation is not required to represent a geometry. Multipolygons as a type of relations are now understood almost everywhere: For example, if the ways in a relation form closed rings, such relations are understood as an area. Interpretations start at the question whether the presence of the tag area=yes is required for this. Other relations, such as routes or turn restrictions, obtain their geometry as the sum of the geometries of their members of type node and way.

Relations on top of relations are technically possible, but have little practical relevance. Relations on relations also create a risk that if the members of the members are also resolved until the ultimately referenced nodes, then one gets insane amounts of data. For that reason there are so many different approaches depending on context to resolve references of relations partially that a whole section is dedicated to that.# VTOL-Maps
#   V T O L - M a p s  
 