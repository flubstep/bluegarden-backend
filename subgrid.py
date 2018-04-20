import numpy as np
from scipy import stats

from geometry_utils import reduce_line_segment
from debug_log import debug_log


class Triangle(object):
    """
    Object to match the API of ThreeJS
    https://threejs.org/docs/#api/math/Triangle
    """
    pass


class Line3(object):
    """
    Object to match the API of ThreeJS
    https://threejs.org/docs/#api/math/Line3
    """
    pass


class LidarSubgrid(object):
    """
    A subgrid inside of a grid square is shaped like this:

    x0,y0,z00 ------ x1,y0,z10
          |           |
          |           |
          |           |
          |           |
    x0,y1,z01 ------ x1,y1,z11
    """

    def __init__(self, points, rect):
        self._points = points
        self._lines = []
        self._triangles = []
        self._rect = rect
        self.classification = self._get_classification()

    def _get_classification(self):
        classifications = self._points[:,6]
        classifications = classifications[classifications != 1]
        if len(classifications) > 0:
            mode = stats.mode(classifications).mode[0]
            return int(mode)
        else:
            return 1 # Unclassified

    def compress_stage1(self, stddev_threshhold=50, nearness_threshhold=25):
        # Stage 1: If all z coords are all within some tolerance, define
        # the plane as a flat square
        zs = self._points[:,2]
        if np.std(zs) < stddev_threshhold:
            old_length = len(self._points)
            x0, y0, x1, y1 = self._rect
            z = float(np.mean(zs))
            self._triangles = [
                x1, y0, z,
                x0, y1, z,
                x0, y0, z,

                x1, y0, z,
                x1, y1, z,
                x0, y1, z
            ] # TODO: change so that you don't convert to buffer yet
            zmask = np.abs(zs - z) > nearness_threshhold
            self._points = self._points[zmask]
            new_length = len(self._points)
            debug_log(f'Reduced subgrid size from {old_length} -> {new_length}')

    def compress_stage1_floor(self, nearness_threshhold=25):
        # Stage 1: If all z coords are all within some distance of the floor
        # of the z positions, then make a guess at the floor.
        zs = self._points[:,2]
        zfloor = np.min(zs) + nearness_threshhold
        floor = zs[zs <= zfloor]
        if len(floor) > 35: # TODO: arbitrary?
            old_length = len(self._points)
            x0, y0, x1, y1 = self._rect
            z = float(zfloor) - nearness_threshhold
            self._triangles = [
                x1, y0, z,
                x0, y1, z,
                x0, y0, z,

                x1, y0, z,
                x1, y1, z,
                x0, y1, z
            ] # TODO: change so that you don't convert to buffer yet
            self._points = self._points[zs > zfloor]
            new_length = len(self._points)
            debug_log(f'Reduced subgrid size from {old_length} -> {new_length}')

    def compress_stage2_detect_discontinuities(self, nearness_threshhold=25):
        # Stage 1: If all z coords are all within some distance of the floor
        # of the z positions, then make a guess at the floor.
        zs = self._points[:,2]
        zfloor = np.min(zs) + nearness_threshhold
        floor = zs[zs <= zfloor]
        if len(floor) > 35: # TODO: arbitrary?
            old_length = len(self._points)
            x0, y0, x1, y1 = self._rect
            z = float(zfloor) - nearness_threshhold
            self._triangles = [
                x1, y0, z,
                x0, y1, z,
                x0, y0, z,

                x1, y0, z,
                x1, y1, z,
                x0, y1, z
            ] # TODO: change so that you don't convert to buffer yet
            self._points = self._points[zs > zfloor]
            new_length = len(self._points)
            debug_log(f'Reduced subgrid size from {old_length} -> {new_length}')

    def compress_stage2(self):
        failures = 0
        old_length = len(self._points)
        P = self._points
        while failures < 20:
            results = reduce_line_segment(P)
            if results:
                P, L1, L2 = results
                self._lines += [float(x) for x in L1[:3]]
                self._lines += [float(x) for x in L2[:3]]
            else:
                failures += 1
        new_length = len(P)
        debug_log(f'Reduced subgrid size from {old_length} -> {new_length} + {len(self._lines)//3} lines')
        self._points = P
