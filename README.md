Dhruv Sharma, ds7042@rit.edu

# Trapezoidal Map and Planar Point Location Codebase README

## Overview
This codebase is designed to implement a trapezoidal map and planar point location algorithm, essential for computational geometry tasks such as map searching and spatial data structures.

## File Descriptions

### `trapezoidal_map_and_planar_point_location.py`
- **Description**: Main driver script. Initializes the trapezoidal map, processes input, builds the map, displays the adjacency matrix, and queries points.
- **Usage**: Run directly. Requires `input.txt` for setup and outputs `output.txt` with the adjacency matrix.

### `trapezoid.py`
- **Description**: Contains `Trapezoid` class and related node classes.
- **Usage**: Imported by the main script.

### `points_and_segments.py`
- **Description**: Contains `Point` and `Segment` class definitions.
- **Usage**: Imported by the main script.

### `input.txt`
- **Description**: Input data file with a list of segments for the trapezoidal map.
- **Usage**: Should be formatted with endpoint coordinates for segments.

### `output.txt`
- **Description**: Output file with the adjacency matrix from the main script.
- **Usage**: Automatically generated.

## Running the Codebase

Ensure all `.py` files and `input.txt` are in the same directory. Run:

```bash
python trapezoidal_map_and_planar_point_location.py
```

Follow prompts for point input.

## Dependencies

The code may depend on libraries such as `collections` and `pandas`. Install with:

```bash
pip install pandas
```

## Notes

- Modify `input.txt` for different segments.
- Python version compatibility is essential.
- Output is logged to both the command line and `output.txt`.
