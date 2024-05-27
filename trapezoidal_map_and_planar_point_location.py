"""
author : Dhruv Sharma 
input file used : ds7042
main file with the driver code and necessary functionality
"""

import collections
from points_and_segments import *
from trapezoid import *
import matplotlib.pyplot as plt
import pandas as pd

# this will be the root of our search structure
root_node = None

# 4 global dictionaries , which will help build the adjacency matrix at the end
# left node identifier -> Xnodes ( all of which has the pointer to the single Point object)
P = collections.defaultdict(list)
# right node identifier -> Xnodes ( all of which has a pointer to the single Point object)
Q = collections.defaultdict(list)
# segment identifier -> Ynodes ( all of which has a pointer to the single Segment object)
S = collections.defaultdict(list)
# trapezoid identifier -> LeafNodes ( all of which has a pointer to the Trapezoid object via the map)
T = collections.defaultdict(list)


# these dictionaries are displayed at the end for debugging and to get a clear idea
# of the data structures to manage the relationships between nodes

def display_dictionaries():
    # used for debugging to display dictionaries
    print("Dictionary P (Left Node Identifiers):")
    for key, value in P.items():
        print(f"  {key}: {value}")

    print("\nDictionary Q (Right Node Identifiers):")
    for key, value in Q.items():
        print(f"  {key}: {value}")

    print("\nDictionary S (Segment Identifiers):")
    for key, value in S.items():
        print(f"  {key}: {value}")

    print("\nDictionary T (Trapezoid Identifiers):")
    for key, value in T.items():
        print(f"  {key}: {value}")


class Node:
    """ Base class for nodes in the graph. """

    def __init__(self):
        # storing the parents
        self.parents = []


class XNode(Node):
    """ Node to represent decision based on x-coordinate. """

    def __init__(self, x_value, point_represented, left=None, right=None):
        super().__init__()
        self.point_represented = point_represented
        # this is the point object ( mostly for printing purposes )
        self.x_value = x_value
        self.left = left
        self.right = right

    def __repr__(self):
        return f"XNode({self.x_value})"


class YNode(Node):
    """ Node to represent decision based on y-coordinate or segment. """

    def __init__(self, segment, above=None, below=None):
        super().__init__()
        # this is the segment object
        self.segment = segment
        self.above = above
        self.below = below

    def __repr__(self):
        return f"YNode({self.segment})"


class LeafNode(Node):
    """ Node to represent a trapezoid or a specific region in the map. """

    def __init__(self, trapezoid_id):
        super().__init__()
        self.trapezoid_id = trapezoid_id

    def print_parents(self):
        for parent in self.parents:
            print(parent)

    def reconnect_parents_to_new_subtree(self, new_subtree_root):
        for parent in self.parents:
            if isinstance(parent, XNode):
                if parent.left == self:
                    parent.left = new_subtree_root
                elif parent.right == self:
                    parent.right = new_subtree_root

            if isinstance(parent, YNode):
                if parent.above == self:
                    parent.above = new_subtree_root
                elif parent.below == self:
                    parent.below = new_subtree_root

            new_subtree_root.parents.append(parent)
        self.parents = []
        # disconnection , removal of prev

    def __repr__(self):
        return f"LeafNode({self.trapezoid_id})"


def process(filename):
    # Read file and store data
    with open(filename, 'r') as file:
        num_segments = int(file.readline())
        bounding_box = list(map(int, file.readline().split()))
        segments = []
        for _ in range(num_segments):
            segment = list(map(int, file.readline().split()))
            segments.append(segment)

    # Visualization i used for understanding and debugging
    """
    fig, ax = plt.subplots()
    ax.set_xlim(bounding_box[0], bounding_box[2])
    ax.set_ylim(bounding_box[1], bounding_box[3])

    for segment in segments:
        x_values = [segment[0], segment[2]]
        y_values = [segment[1], segment[3]]

        # Plotting line segments
        ax.plot(x_values, y_values, color='blue')

        # Marking endpoints
        ax.plot(segment[0], segment[1], 'ro')  # Start point
        ax.plot(segment[2], segment[3], 'ro')  # End point

        # Dotted lines for endpoints
        ax.plot([segment[0], segment[0]], [bounding_box[1], bounding_box[3]], 'r--', alpha=0.5)
        ax.plot([segment[2], segment[2]], [bounding_box[1], bounding_box[3]], 'r--', alpha=0.5)

    plt.title('Line Segments with Endpoints')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.show()
    """
    return segments, bounding_box


def displaying_matrices():
    # Sample matrix data
    data = {
        'Column1': [1, 2, 3],
        'Column2': [4, 5, 6],
        'Column3': [7, 8, 9]
    }

    # Creating a DataFrame with row and column names
    df = pd.DataFrame(data, index=['Row1', 'Row2', 'Row3'])
    # Accessing and modifying a specific value
    # df.loc['Row1', 'Column2'] = 10000

    # Adding a last row for column sums
    df.loc['col_sum'] = df.sum()

    # Adding a last column for row sums
    df['row_sum'] = df.sum(axis=1)

    # Update row_sum and col_sum after changing a particular entry
    # For example, changing 'Row1' and 'Column2' to a new value
    df.loc['Row1', 'Column2'] = 500  # changing the value for demonstration purposes

    # Update col_sum and row_sum
    df.loc['col_sum'] = df.sum()
    df['row_sum'] = df.sum(axis=1)

    # Displaying the DataFrame
    print(df)


# Assuming you have already defined the LeafNode class as shown earlier


def is_point_above_segment(point, segment):
    """
    Check if a point is above a segment.
    :param point: A Point object representing the query point.
    :param segment: A Segment object representing the segment in a YNode.
    :return: True if the point is above the segment, False otherwise.
    """
    # Unpack points from the segment
    x1, y1 = segment.left.x, segment.left.y
    x2, y2 = segment.right.x, segment.right.y

    # Special case: vertical segment
    if x1 == x2:
        return point.y > max(y1, y2)

    # Calculate the y-coordinate on the segment at the x-coordinate of the point
    slope = (y2 - y1) / (x2 - x1)
    y_on_segment = y1 + slope * (point.x - x1)

    return point.y > y_on_segment


def query_point_lies_on_segment(point, segment):
    """
    Check if the point lies exactly on the segment.
    :param point: A Point object representing the query point.
    :param segment: A Segment object representing the segment in a YNode.
    :return: True if the point lies on the segment, False otherwise.
    """
    x1, y1 = segment.left.x, segment.left.y
    x2, y2 = segment.right.x, segment.right.y

    # Check for collinearity
    if (x2 - x1) * (point.y - y1) == (y2 - y1) * (point.x - x1):
        # Check if point is within the segment's bounding box
        return min(x1, x2) <= point.x <= max(x1, x2) and min(y1, y2) <= point.y <= max(y1, y2)

    return False


def compare_segment_slopes(segment1, segment2, point):
    """
    Compare the slopes of two segments at the point.
    :param segment1: First segment for slope comparison.
    :param segment2: Second segment for slope comparison.
    :param point: The point at which slopes are compared.
    :return: True if slope of segment1 is larger than slope of segment2.
    """

    def slope(segment):
        dx = segment.right.x - segment.left.x
        dy = segment.right.y - segment.left.y
        return dy / dx if dx != 0 else float('inf')

    return slope(segment1) > slope(segment2)


def query(node, query_point, segment_of_interest=None):
    """
    Find the leaf node corresponding to the trapezoid that contains the query point.
    :param segment_of_interest: This is None during the querying step , else it contains
    the segment added during an iteration of the ric
    here we also consider that queries that are on a segment are invalid since there could be
    multiple answers
    :param node: The current node in the search structure (initially the root).
    :param query_point: The point for which we are querying.
    :return: LeafNode that contains the query point.
    """
    current_node = node
    # print(current_node)

    # print("query point : " , query_point)

    while not isinstance(current_node, LeafNode):
        if isinstance(current_node, XNode):
            # If the query point is equal to or greater than the x-node's value, go right
            # print("the current node : ", current_node)
            # print("query point : ", query_point)
            if query_point.x <= current_node.x_value:
                current_node = current_node.right
            else:
                current_node = current_node.left
        elif isinstance(current_node, YNode):
            segment = current_node.segment
            if query_point_lies_on_segment(query_point, segment):
                # Compare slopes if the point lies on the segment
                if compare_segment_slopes(segment_of_interest, segment, query_point):
                    current_node = current_node.above
                else:
                    current_node = current_node.below
            elif is_point_above_segment(query_point, segment):
                current_node = current_node.above
            else:
                current_node = current_node.below

    return current_node


def find_intersected_trapezoids(segment, trapezoids):
    """
    Find the trapezoids intersected by the added segment.
    :param segment: The new line segment being added.
    :param trapezoids: The trapezoidal map.
    :return: List of indices of the trapezoids intersected by the segment.
    """
    global root_node
    intersected_trapezoids = []

    # Step 1 & 2: Identify endpoints and find the first trapezoid
    left_endpoint = segment.left
    right_endpoint = segment.right

    print("before query")
    display_dictionaries()

    current_trapezoid_node = query(root_node, left_endpoint, segment)
    print("the current queried leaf node is :  ", current_trapezoid_node)
    current_trapezoid = trapezoids[current_trapezoid_node.trapezoid_id][2]
    # Assuming the third element is the Trapezoid object

    # Step 3 & 4: Iteratively find neighboring trapezoids
    while current_trapezoid and right_endpoint.x > current_trapezoid.rightp.x:
        intersected_trapezoids.append(current_trapezoid.trapezoid_id)
        if is_point_above_segment(current_trapezoid.rightp, segment):
            # Move to lower right neighbor
            current_trapezoid = current_trapezoid.neighbors['bottom_right']
        else:
            # Move to upper right neighbor
            current_trapezoid = current_trapezoid.neighbors['top_right']
            # print("current trapezoid : " , current_trapezoid)

    # Add the last trapezoid
    if current_trapezoid:
        intersected_trapezoids.append(current_trapezoid.trapezoid_id)

    return intersected_trapezoids


def update_neighbors(intersected_trapezoid, new_trapezoid):
    # now we also need to find the neighbours for these new trapezoids
    for key, value in intersected_trapezoid.neighbors.items():
        if value is not None:
            if key in ["top_left", "bottom_left"]:
                # on the left side
                if new_trapezoid.top == value.top and new_trapezoid.neighbors["top_left"] is None:
                    new_trapezoid.neighbors["top_left"] = value
                if new_trapezoid.bottom == value.bottom and new_trapezoid.neighbors["bottom_left"] is None:
                    new_trapezoid.neighbors["bottom_left"] = value
            else:
                # on the right side
                if new_trapezoid.top == value.top and new_trapezoid.neighbors["top_right"] is None:
                    new_trapezoid.neighbors["top_right"] = value
                if new_trapezoid.bottom == value.bottom and new_trapezoid.neighbors["bottom_right"] is None:
                    new_trapezoid.neighbors["bottom_right"] = value


def update(trapezoids_intersected, trapezoids, left_point, right_point, segment):
    """
    changes to the trapezoidal map and the search structure are done here
    :param segment:
    :param right_point:
    :param left_point:
    :param trapezoids_intersected: trapezoids indexes to be removed and replaced
    :param trapezoids: trapezoidal map
    :return:
    """

    global root_node

    print("trapezoids  : ", T)
    print("trapezoids intersected  : ", trapezoids_intersected)

    for trapezoids_intersected_id in trapezoids_intersected:
        if trapezoids_intersected_id not in trapezoids:
            continue
        print("intersected id : ", trapezoids_intersected_id)
        intersected_trapezoid_identifier = trapezoids[trapezoids_intersected_id][0]
        intersected_trapezoid_leaf = trapezoids[trapezoids_intersected_id][1]
        intersected_trapezoid = trapezoids[trapezoids_intersected_id][2]

        # now we test for the cases :
        if left_point.x < intersected_trapezoid.leftp.x and intersected_trapezoid.rightp.x < right_point.x:
            # case 3 where the segment completely crosses the trapezoid
            """
            • A single trapezoid is replaced by two trapezoids, one above and one below
            the segment, denoted X and Y.
            • We replace the leaf node for the original trapezoid with a y-node whose
            children are leaf nodes associated with X and Y
            """
            # create 2 new trapezoids firstly
            X_trapezoid = Trapezoid(intersected_trapezoid.top, None,
                                    intersected_trapezoid.leftp, intersected_trapezoid.rightp)
            Y_trapezoid = Trapezoid(None, intersected_trapezoid.bottom,
                                    intersected_trapezoid.leftp, intersected_trapezoid.rightp)

            X_trapezoid.bottom = Y_trapezoid
            Y_trapezoid.top = X_trapezoid

            # now we also need to find the neighbours for these new trapezoids
            update_neighbors(intersected_trapezoid, X_trapezoid)
            update_neighbors(intersected_trapezoid, Y_trapezoid)

            # now we remove the intersected trapezoid from the map , and from the identifier dictionary
            trapezoids.pop(intersected_trapezoid.trapezoid_id)
            T.pop(intersected_trapezoid_identifier)

            # create LeafNodes for the trapezoids , and make the subtree , also we can add to identifier
            # dictionary
            Xleaf = LeafNode(X_trapezoid.trapezoid_id)
            T[X_trapezoid.identifier].append(Xleaf)
            Yleaf = LeafNode(Y_trapezoid.trapezoid_id)
            T[Y_trapezoid.identifier].append(Yleaf)
            segment_y_node = YNode(segment, Xleaf, Yleaf)
            S[segment.identifier].append(segment_y_node)
            # keep track of the parent as well
            Xleaf.parents.append(segment_y_node)
            Yleaf.parents.append(segment_y_node)

            # now we put in the new trapezoids
            trapezoids[X_trapezoid.trapezoid_id] = [X_trapezoid.identifier, Xleaf, X_trapezoid]
            trapezoids[Y_trapezoid.trapezoid_id] = [Y_trapezoid.identifier, Yleaf, Y_trapezoid]

            # and finally we delete and replace from the search structure
            intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(segment_y_node)

        elif left_point.x >= intersected_trapezoid.leftp.x and right_point.x <= intersected_trapezoid.rightp.x:
            # case 2 where the segment completely lies inside the trapezoid
            print("in here !!!")

            U_trapezoid = Trapezoid(intersected_trapezoid.top,
                                    intersected_trapezoid.bottom,
                                    intersected_trapezoid.leftp,
                                    left_point)
            Y_trapezoid = Trapezoid(intersected_trapezoid.top, None,
                                    left_point,
                                    right_point)
            Z_trapezoid = Trapezoid(None, intersected_trapezoid.bottom,
                                    left_point,
                                    right_point)
            X_trapezoid = Trapezoid(intersected_trapezoid.top,
                                    intersected_trapezoid.bottom,
                                    right_point,
                                    intersected_trapezoid.rightp)

            Y_trapezoid.bottom = X_trapezoid
            Z_trapezoid.top = Y_trapezoid

            U_trapezoid.neighbors["top_right"] = Y_trapezoid
            U_trapezoid.neighbors["bottom_right"] = Z_trapezoid

            Y_trapezoid.neighbors["top_left"] = U_trapezoid
            Y_trapezoid.neighbors["top_right"] = X_trapezoid

            Z_trapezoid.neighbors["bottom_left"] = U_trapezoid
            Z_trapezoid.neighbors["bottom_right"] = X_trapezoid

            X_trapezoid.neighbors["top_left"] = Y_trapezoid
            X_trapezoid.neighbors["bottom_left"] = Z_trapezoid

            update_neighbors(intersected_trapezoid, U_trapezoid)
            update_neighbors(intersected_trapezoid, X_trapezoid)

            # now we remove the intersected trapezoid from the map , and from the identifier dictionary
            trapezoids.pop(intersected_trapezoid.trapezoid_id)
            T.pop(intersected_trapezoid_identifier)

            # create LeafNodes for the trapezoids , and make the subtree , also we can add to identifier
            # dictionary
            Uleaf = LeafNode(U_trapezoid.trapezoid_id)
            T[U_trapezoid.identifier].append(Uleaf)
            Yleaf = LeafNode(Y_trapezoid.trapezoid_id)
            T[Y_trapezoid.identifier].append(Yleaf)
            Xleaf = LeafNode(X_trapezoid.trapezoid_id)
            T[X_trapezoid.identifier].append(Xleaf)
            Zleaf = LeafNode(Z_trapezoid.trapezoid_id)
            T[Z_trapezoid.identifier].append(Zleaf)

            segment_y_node = YNode(segment, Yleaf, Zleaf)
            S[segment.identifier].append(segment_y_node)

            point_q_node = XNode(right_point.x, right_point, segment_y_node, Xleaf)
            Q[right_point.identifier].append(point_q_node)
            point_p_node = XNode(left_point.x, left_point, Uleaf, point_q_node)
            P[left_point.identifier].append(point_p_node)

            # keep track of the parent as well
            Yleaf.parents.append(segment_y_node)
            Zleaf.parents.append(segment_y_node)
            Xleaf.parents.append(point_q_node)
            Uleaf.parents.append(point_p_node)

            # now we put in the new trapezoids
            trapezoids[U_trapezoid.trapezoid_id] = [U_trapezoid.identifier, Uleaf, U_trapezoid]
            trapezoids[Y_trapezoid.trapezoid_id] = [Y_trapezoid.identifier, Yleaf, Y_trapezoid]
            trapezoids[Z_trapezoid.trapezoid_id] = [Z_trapezoid.identifier, Zleaf, Z_trapezoid]
            trapezoids[X_trapezoid.trapezoid_id] = [X_trapezoid.identifier, Xleaf, X_trapezoid]

            # and finally we delete and replace from the search structure
            intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(point_p_node)

            if isinstance(root_node, LeafNode):
                # we have to change the global root
                root_node = point_p_node

        else:
            # case 3 where one point lies inside and the other out
            X_trapezoid = Trapezoid(intersected_trapezoid.top,
                                    intersected_trapezoid.bottom,
                                    intersected_trapezoid.leftp,
                                    left_point)
            Y_trapezoid = Trapezoid(intersected_trapezoid.top, None,
                                    left_point,
                                    right_point)
            Z_trapezoid = Trapezoid(None, intersected_trapezoid.bottom,
                                    left_point,
                                    intersected_trapezoid.rightp)

            Y_trapezoid.bottom = Z_trapezoid
            Z_trapezoid.top = Y_trapezoid

            X_trapezoid.neighbors["top_right"] = Y_trapezoid
            X_trapezoid.neighbors["bottom_right"] = Z_trapezoid

            Y_trapezoid.neighbors["top_left"] = X_trapezoid

            Z_trapezoid.neighbors["bottom_left"] = X_trapezoid

            update_neighbors(intersected_trapezoid, X_trapezoid)
            update_neighbors(intersected_trapezoid, Y_trapezoid)
            update_neighbors(intersected_trapezoid, Z_trapezoid)

            # now we remove the intersected trapezoid from the map , and from the identifier dictionary
            trapezoids.pop(intersected_trapezoid.trapezoid_id)
            T.pop(intersected_trapezoid_identifier)

            # create LeafNodes for the trapezoids , and make the subtree , also we can add to identifier
            # dictionary
            Xleaf = LeafNode(X_trapezoid.trapezoid_id)
            T[X_trapezoid.identifier].append(Xleaf)
            Yleaf = LeafNode(Y_trapezoid.trapezoid_id)
            T[Y_trapezoid.identifier].append(Yleaf)
            Zleaf = LeafNode(Z_trapezoid.trapezoid_id)
            T[Z_trapezoid.identifier].append(Zleaf)
            segment_y_node = YNode(segment, Yleaf, Zleaf)
            S[segment.identifier].append(segment_y_node)

            subtree_root = None
            # we need to know which is inside
            if left_point.x >= intersected_trapezoid.leftp.x:
                # left in , right out
                point_p_node = XNode(left_point.x, left_point, Xleaf, segment_y_node)
                P[left_point.identifier].append(point_p_node)
                Xleaf.parents.append(point_p_node)
                subtree_root = point_p_node
            else:
                # right in left out
                point_q_node = XNode(right_point.x, right_point, Xleaf, segment_y_node)
                Q[right_point.identifier].append(point_q_node)
                Xleaf.parents.append(point_q_node)
                subtree_root = point_q_node

            # keep track of the parent as well
            Yleaf.parents.append(segment_y_node)
            Zleaf.parents.append(segment_y_node)

            # now we put in the new trapezoids
            trapezoids[Y_trapezoid.trapezoid_id] = [Y_trapezoid.identifier, Yleaf, Y_trapezoid]
            trapezoids[Z_trapezoid.trapezoid_id] = [Z_trapezoid.identifier, Zleaf, Z_trapezoid]
            trapezoids[X_trapezoid.trapezoid_id] = [X_trapezoid.identifier, Xleaf, X_trapezoid]

            # and finally we delete and replace from the search structure
            intersected_trapezoid_leaf.reconnect_parents_to_new_subtree(subtree_root)


def ric(segments, bounding_box, trapezoids):
    """
    the random incremental construction algorithm
    :return:
    """
    global root_node
    # trapezoidal map stores the leaf node as well as a display identifier
    left_p = Point(bounding_box[0], bounding_box[1], True)
    right_p = Point(bounding_box[2], bounding_box[3], False)
    # we will not put these in the dictionary, as we never create any xnodes for them

    first_trapezoid = Trapezoid(None, None, left_p, right_p)
    root_node = LeafNode(first_trapezoid.trapezoid_id)
    # leaf node has a pointer to the trapezoid
    print("Initial trapezoidal map root:", root_node)
    # map storing the leaf node as well
    trapezoids[first_trapezoid.trapezoid_id] = [f"T{root_node.trapezoid_id}", root_node,
                                                first_trapezoid]
    print("trapezoids : ", trapezoids)

    # also store in the dictionary
    T[first_trapezoid.identifier].append(root_node)

    display_dictionaries()

    for segment in segments:
        left_point = Point(segment[0], segment[1], True)
        right_point = Point(segment[2], segment[3], False)
        segment = Segment(left_point, right_point)
        print(segment)
        # creating a segment
        # finding the intersected trapezoids
        trapezoids_intersected = find_intersected_trapezoids(segment, trapezoids)
        # deleting the trapezoids intersected from the map and replacing them with the new trapezoids
        # to be added
        print("trapezoids intersected are : ", trapezoids_intersected)
        """
        Remove the leaves for the trapezoids from the search structure, and create leaves for
        the new trapezoids. Link the new leaves to the existing inner nodes
        by adding some new inner nodes, as explained below.
        this is the function where we will have the 3 cases 
        """
        # we are doing both the above operations in one function below
        update(trapezoids_intersected, trapezoids, left_point, right_point, segment)


def create_adjacency_matrix_and_output(filename, trapezoids):
    # Combine the keys from all dictionaries and sort them
    all_keys = sorted(set(P.keys()) | set(Q.keys()) | set(S.keys()) | set(T.keys()))

    # Initialize a DataFrame with zeros
    df = pd.DataFrame(0, index=all_keys, columns=all_keys)

    # Populate the DataFrame using the dictionaries
    for key in all_keys:
        if key in P or key in Q:
            # print("key  : ", key)
            for node in P[key]:
                left_child = node.left
                right_child = node.right

                if isinstance(left_child, XNode):
                    left_child = left_child.point_represented.identifier
                if isinstance(left_child, YNode):
                    left_child = left_child.segment.identifier
                if isinstance(left_child, LeafNode):
                    left_child = trapezoids[left_child.trapezoid_id][0]
                if isinstance(right_child, XNode):
                    right_child = right_child.point_represented.identifier
                if isinstance(right_child, YNode):
                    right_child = right_child.segment.identifier
                if isinstance(right_child, LeafNode):
                    right_child = trapezoids[right_child.trapezoid_id][0]

                df.loc[left_child, key] += 1
                df.loc[right_child, key] += 1
        if key in S:
            for node in S[key]:
                left_child = node.above
                right_child = node.below

                if isinstance(left_child, XNode):
                    left_child = left_child.point_represented.identifier
                if isinstance(left_child, YNode):
                    left_child = left_child.segment.identifier
                if isinstance(left_child, LeafNode):
                    left_child = trapezoids[left_child.trapezoid_id][0]
                if isinstance(right_child, XNode):
                    right_child = right_child.point_represented.identifier
                if isinstance(right_child, YNode):
                    right_child = right_child.segment.identifier
                if isinstance(right_child, LeafNode):
                    right_child = trapezoids[right_child.trapezoid_id][0]

                df.loc[left_child, key] += 1
                df.loc[right_child, key] += 1

    # Add a last row and column for sums
    df.loc['Sum'] = df.sum()
    df['Sum'] = df.sum(axis=1)

    # Write the DataFrame to a text file
    with open(filename, 'w') as f:
        f.write(df.to_string())


def query_v1(root_node, query_point):
    current_node = root_node
    # print(current_node)
    # print("query point : " , query_point)

    path = []

    while not isinstance(current_node, LeafNode):
        if isinstance(current_node, XNode):
            # If the query point is equal to or greater than the x-node's value, go right
            # print("the current node : ", current_node)
            # print("query point : ", query_point)
            path.append(current_node.point_represented.identifier)
            if query_point.x <= current_node.x_value:
                current_node = current_node.right
            else:
                current_node = current_node.left
        elif isinstance(current_node, YNode):
            segment = current_node.segment
            path.append(current_node.segment.identifier)
            if query_point_lies_on_segment(query_point, segment):
                # if the query point lies on the segment , we return the above trapezoid
                current_node = current_node.above
            elif is_point_above_segment(query_point, segment):
                current_node = current_node.above
            else:
                current_node = current_node.below

    path.append(f"T{current_node.trapezoid_id}")
    return path


def main():
    # Initialize trapezoidal map (you need to define this class)
    # trapezoidal_map = TrapezoidalMap()

    # Process input file and get segments and bounding box
    segments, bounding_box = process('input.txt')
    print("the segments are : ", segments)
    print("the bounding box : ", bounding_box)

    trapezoids = collections.defaultdict(list)

    ric(segments, bounding_box, trapezoids)

    display_dictionaries()

    print("***********************************************************************************************")
    # now we will use our dictionaries to display the output as an adjacency matrix
    print("writing to file the output adjacency matrix ... ")
    # Now call the function with your dictionaries and desired output filename
    create_adjacency_matrix_and_output('output.txt', trapezoids)
    print("***********************************************************************************************")
    input_values = input("Enter the point coordinates (x y): ").split()
    query_point = Point(float(input_values[0]), float(input_values[1]))
    # Find the traversal path , for the point location query
    path = query_v1(root_node, query_point)
    # Print the traversal path as a string
    print("Traversal path: " + ' '.join(path))


if __name__ == '__main__':
    main()
