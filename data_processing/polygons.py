import numpy as np


class Polygon_Opti:
    """
    Class to check if a point is inside a polygon

    Args:
        pol (np.array): (n_points,2) array of points defining the polygon
    """

    def __init__(self, pol: np.array):
        self.pol = pol
        # Precompute all of the line equations of all the sides
        self.coefs = [self._vec_to_line(pol[k], pol[k + 1]) for k in range(len(pol) - 1)] + [
            self._vec_to_line(pol[-1], pol[0])
        ]
        self.coefs = np.array(self.coefs)
        self.bb = self._get_bb_poly(pol)
        self.point_outside = np.array([self.bb[0] - 1, self.bb[2] - 1])
        self.n_pol = len(self.pol)

    def _vec_to_line(self, vec1, vec2):
        """Return the coefficients of the line defined by two points

        Args:
            vec1 (np.array): (2,) array of points
            vec2 (np.array): (2,) array of points
        """
        # assert vec1.shape == vec2.shape == (2,)
        a = vec2[1] - vec1[1]
        b = vec1[0] - vec2[0]
        c = vec2[0] * vec1[1] - vec1[0] * vec2[1]
        return a, b, c

    def _get_bb_poly(self, pol):
        """Return the bounding box of the polygon

        Args:
            pol (np.array): (n_points,2) array of points defining the polygon
        """
        # assert pol.shape[1] == 2
        (x_min, y_min), (x_max, y_max) = pol.min(axis=0), pol.max(axis=0)
        return x_min, x_max, y_min, y_max

    def _is_inside_bb(self, points):
        """Check if points are inside the bounding box of the polygon

        Args:
            points (np.array): (n_points,2) array of points
        """
        x_min, x_max, y_min, y_max = self.bb
        return (
            (x_min <= points[:, 0])
            & (points[:, 0] <= x_max)
            & (y_min <= points[:, 1])
            & (points[:, 1] <= y_max)
        )

    def _vec_to_line_array(self, vec1: np.array):
        """Return the coefficients of the line defined by two points

        Args:
            vec1 (np.array): (n_points,2) points to compute the line equation
        """
        a = self.point_outside[1] - vec1[:, 1]
        b = vec1[:, 0] - self.point_outside[0]
        c = self.point_outside[0] * vec1[:, 1] - vec1[:, 0] * self.point_outside[1]
        return a, b, c

    def _dist_hyperplane_array(self, points, a, b, c):
        """Distance between points and the hyperplanes defined by a,b,c

        Args:
            points (np.array): (n_points,2) array of points
            a (np.array): (n_sides,) array of coefficients
            b (np.array): (n_sides,) array of coefficients
            c (np.array): (n_sides,) array of coefficients

        Returns:
            np.array: (n_points,n_sides) array of distances

        """

        # return points[:, 0, None] * a + points[:, 1, None] * b + c
        return np.dot(points, np.vstack((a, b))) + c # -> this is faster

    def _colinear_array(self, a1, b1, a2, b2, eps=1e-6):
        """Check if lines are colinear by checking the determinant of the matrix

        Args:
            a1 (np.array): (n_sides,) array of coefficients
            b1 (np.array): (n_sides,) array of coefficients
            a2 (np.array): (n_sides,) array of coefficients
            b2 (np.array): (n_sides,) array of coefficients
            eps (float): threshold for the colinearity

        Returns:
            np.array: (n_sides,n_sides) array of boolean indicating if the lines are colinear
        """
        col = np.abs((a1[:, None] * b2) - (b1[:, None] * a2)) < eps
        return col.T

    def are_inside(self, points):
        """Check if points are inside the polygon.
        A point is considered inside if a ray starting from the point to a point outside the polygon intersects the polygon an odd number of times.

        Args:
            points (np.array): (n_points,2) array of points

        Returns:
            np.array: (n_points,) array of boolean indicating if the points are inside the polygon
        """
        a1, b1, c1 = self._vec_to_line_array(points)
        a2, b2, c2 = self.coefs[:, 0], self.coefs[:, 1], self.coefs[:, 2]

        side_0_array = self.pol
        side_1_array = np.concatenate([self.pol[1:], self.pol[:1]], axis=0)
        point_0_array = points
        point_1_array = np.array([self.point_outside] * points.shape[0])

        d1 = self._dist_hyperplane_array(side_0_array, a1, b1, c1)
        d2 = self._dist_hyperplane_array(side_1_array, a1, b1, c1)

        d3 = self._dist_hyperplane_array(point_0_array, a2, b2, c2).T
        d4 = self._dist_hyperplane_array(point_1_array, a2, b2, c2).T

        col = self._colinear_array(a1, b1, a2, b2)

        pts_inter_side = np.logical_not((d1 * d2 > 0) | (d3 * d4 > 0) | col)
        nb_inter_pts = pts_inter_side.sum(axis=0)
        return np.mod(nb_inter_pts, 2) == 1
