"""
author : Dhruv Sharma 
file containing classes for Points and Segments
"""

class Point:
    left_counter = 0
    right_counter = 0

    def __init__(self, x, y, is_left=True):
        self.x = x
        self.y = y
        self.identifier = self.assign_identifier(is_left)

    def assign_identifier(self, is_left):
        if is_left:
            identifier = f"P{Point.left_counter}"
            Point.left_counter += 1
        else:
            identifier = f"Q{Point.right_counter}"
            Point.right_counter += 1
        return identifier

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, id={self.identifier})"


class Segment:
    counter = 1

    def __init__(self, left_point, right_point):
        self.left = left_point
        self.right = right_point
        self.identifier = f"S{Segment.counter}"
        Segment.counter += 1

    def __repr__(self):
        return f"Segment(left={self.left}, right={self.right}, id={self.identifier})"


if __name__ == '__main__':
    # Creating Points and Segments
    point1 = Point(0, 0, is_left=True)  # This will be P1
    point2 = Point(1, 1, is_left=False)  # This will be Q1
    point3 = Point(2, 2, is_left=True)  # This will be P2

    segment1 = Segment(point1, point2)  # This will be S1
    segment2 = Segment(point1, point3)  # This will be S2

    # Displaying Points and Segments
    print(point1)  # Output: Point(x=0, y=0, id=P1)
    print(point2)  # Output: Point(x=1, y=1, id=Q1)
    print(segment1)  # Output: Segment(left=Point(x=0, y=0
