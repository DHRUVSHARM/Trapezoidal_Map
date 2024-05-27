"""
author : Dhruv Sharma
file containing the definition for the Trapezoid class
"""
import collections


class Trapezoid:
    counter = 0  # Class-level attribute for tracking the next available ID

    def __init__(self, top, bottom, leftp, rightp):
        self.top = top
        self.bottom = bottom
        self.leftp = leftp
        self.rightp = rightp
        self.trapezoid_id = Trapezoid.counter
        self.identifier = f"T{Trapezoid.counter}"  # Assigning a unique identifier
        Trapezoid.counter += 1  # Incrementing the counter for the next trapezoid

        self.neighbors = collections.defaultdict()
        self.neighbors['top_left'] = None
        self.neighbors['bottom_left'] = None
        self.neighbors['top_right'] = None
        self.neighbors['bottom_right'] = None

    def __repr__(self):
        return f"Trapezoid(top={self.top}," \
               f" bottom={self.bottom}," \
               f" leftp={self.leftp}, " \
               f"rightp={self.rightp}, " \
               f"trapezoid_id={self.trapezoid_id}," \
               f" identifier={self.identifier}," \
               f" neighbors=[{self.neighbors}])"


# Example usage
if __name__ == '__main__':
    # Create a few trapezoids to see the identifiers
    trapezoid1 = Trapezoid(None, None, None, None)
    trapezoid2 = Trapezoid(None, None, None, None)

    print(trapezoid1)  # Should show identifier as T1
    print(trapezoid2)  # Should show identifier as T2
