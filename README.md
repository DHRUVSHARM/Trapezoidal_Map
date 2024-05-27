
# Trapezoidal Map for Line Segment Intersection Queries

## Overview

This repository contains the implementation of a trapezoidal map algorithm for efficient planar point loacation queries. The algorithm constructs a search structure to manage and query line segments within a bounded region, supporting various geometric operations. The main idea is to organize a plane into various trapezoids , and then use a DAG structure to effeciently answer queries about point location. The leaf nodes represent the trapezoids, and decisions are taken based on X and Y coordinates of points, The search structure is dynamically updated in such a way new points added are able to be located correctly. 

## Technologies Used

- **Python**
- **Matplotlib** (for visualization)
- **Pandas** (for data manipulation)

## Features

- **Line Segment Management**: Efficiently manage and query line segments within a bounded region.
- **Efficient Point Location Queries**: Quickly locate the trapezoid containing a given query point.
- **Visualization**: Visualize line segments and their intersections for debugging and analysis.

## Algorithm Details

### Overview

The main algorithm implemented in this repository involves the construction and management of a trapezoidal map for line segment intersection queries. The algorithm supports efficient point location queries within the segments using a search structure consisting of different types of nodes.

### Implementation

- **Node Classes**: The algorithm uses different node classes (`XNode`, `YNode`, `LeafNode`) to represent decision nodes based on x-coordinates, y-coordinates, and trapezoidal regions.
- **Data Structures**: Four global dictionaries (`P`, `Q`, `S`, `T`) are used to build the adjacency matrix and manage relationships between nodes.
- **Process Function**: Reads input data and segments, and visualizes them for debugging purposes.
- **RIC (Randomized Incremental Construction)**: This function constructs the trapezoidal map by iteratively adding segments and updating the map.
- **Query Function**: Locates the trapezoid that contains a given query point, using the search structure.

### Classes and Functions

#### Node Classes

- **Node**: Base class for nodes in the graph.
- **XNode**: Represents decision nodes based on x-coordinates.
- **YNode**: Represents decision nodes based on y-coordinates or segments.
- **LeafNode**: Represents trapezoidal regions in the map.

#### Key Functions

- `process(filename)`: Reads the input file and stores segment data.
- `ric(segments, bounding_box, trapezoids)`: Implements the randomized incremental construction of the trapezoidal map.
- `query(node, query_point, segment_of_interest=None)`: Finds the leaf node corresponding to the trapezoid containing the query point.
- `find_intersected_trapezoids(segment, trapezoids)`: Identifies trapezoids intersected by a given segment.
- `update(trapezoids_intersected, trapezoids, left_point, right_point, segment)`: Updates the trapezoidal map and search structure when adding new segments.
- `create_adjacency_matrix_and_output(filename, trapezoids)`: Generates an adjacency matrix and outputs it to a file.

## Use Cases

- **Geographic Information Systems (GIS)**: Efficiently manage and query geographic data.
- **Computer Graphics**: Handle geometric operations and visualizations in computer graphics applications.
- **Robotics and Path Planning**: Plan and navigate paths in environments with obstacles.

## Prerequisites

- Python 3.x
- Matplotlib
- Pandas

## How to Run

1. Clone the repository:

   ```sh
   git clone https://github.com/your-username/trapezoidal-map.git
   cd trapezoidal-map
   ```

2. Install the required packages:

   ```sh
   pip install -r requirements.txt
   ```

3. Prepare the input file with line segments:

   - The input file should be named `input.txt` and contain the number of segments, bounding box coordinates, and segment coordinates.

4. Run the main script:

   ```sh
   python main.py
   ```
