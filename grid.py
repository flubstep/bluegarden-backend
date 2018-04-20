from laspy.file import File
import sys
import os
import json
import time
import numpy as np

from debug_log import debug_log
from subgrid import LidarSubgrid
from colors import rgb_for_classification


BASE_PATH = '/Users/albert/Development/dclidar/data/'


class LidarGrid(object):

    def __init__(self, grid_id):
        self.grid_id = grid_id
        filepath = os.path.join(BASE_PATH, f'{grid_id}.las')
        self._infile = File(filepath, mode='r')
        self._points = None
        self._xrange = (None, None)
        self._yrange = (None, None)

    def load_from_lasfile(self):
        points_list = []
        point0 = self._infile.points[0][0]
        N = len(self._infile.points)
        N_segment = int(N / 50)
        self._xrange = [point0[0], point0[0]]
        self._yrange = [point0[1], point0[1]]

        debug_log(f'Loading {N} points from datafile...')
        start_time = time.time()

        # Process points sequentially in order to extract min/max
        # TODO: can this be vectorized?
        for i, p in enumerate(self._infile.points):
            if i % N_segment == 0 and i > 0:
                debug_log(f'{int(i/N*100)}% complete...')
            point = np.array(p[0].tolist())
            points_list.append(point)
            x, y = point[0], point[1]
            self._xrange = (min(self._xrange[0], x), max(self._xrange[1], x))
            self._yrange = (min(self._yrange[0], y), max(self._yrange[1], y))

        self._points = np.array(points_list)
        elapsed_time = time.time() - start_time
        debug_log(f'Load completed in {elapsed_time}s.')

    def load_from_json_file(self, filename):
        points = json.load(open(filename, 'r'))
        point0 = points[0]
        N = len(points)
        self._xrange = [point0[0], point0[0]]
        self._yrange = [point0[1], point0[1]]

        debug_log(f'Loading {N} points from jsonfile...')
        start_time = time.time()
        points_list = []
        for p in points:
            parr = np.array([p[0], p[1], p[2], p[3], 0, 0, p[4]])
            points_list.append(parr)
            x, y = p[0], p[1]
            self._xrange = (min(self._xrange[0], x), max(self._xrange[1], x))
            self._yrange = (min(self._yrange[0], y), max(self._yrange[1], y))

        self._points = np.array(points_list)
        elapsed_time = time.time() - start_time
        debug_log(f'Load completed in {elapsed_time}s.')

    def normalize_to_grid_center(self):
        grid_center_x = sum(self._xrange) / 2
        grid_center_y = sum(self._yrange) / 2

        debug_log(f'Normalizing points around {grid_center_x},{grid_center_y}')

        center = np.zeros(self._points[0].shape)
        center[:2] = [grid_center_x, grid_center_y]
        self._points -= center
        self._xrange = [xloc - grid_center_x for xloc in self._xrange]
        self._yrange = [yloc - grid_center_y for yloc in self._yrange]

    def clip(self, sample=1):
        debug_log(f'Clipping points to {sample}x size of original')
        old_length = len(self._points)
        max_x = (self._xrange[1] - self._xrange[0]) / 2 * (sample**0.5)
        max_y = (self._yrange[1] - self._yrange[0]) / 2 * (sample**0.5)
        x_mask = np.abs(self._points[:,0]) < max_x
        y_mask = np.abs(self._points[:,1]) < max_y
        self._points = self._points[x_mask & y_mask]
        new_length = len(self._points)
        debug_log(f'Points reduced from {old_length} -> {new_length}')

    def dump_to_json_file(self, filename, sample=1):
        debug_log(f'Writing json to file {filename}')
        points = self._points
        if sample < 1:
            max_x = (self._xrange[1] - self._xrange[0]) / 2 * (sample**0.5)
            max_y = (self._yrange[1] - self._yrange[0]) / 2 * (sample**0.5)
            points = [
                p for p in points
                if abs(p[0]) < max_x and abs(p[1]) < max_y
            ]
        points_for_export = [
            [
                float(p[0]),  # Xloc
                float(p[1]),  # Yloc
                float(p[2]),  # Zloc
                float(p[3]),  # Intensity
                float(p[6])   # Classification byte
            ] for p in points
        ]
        return json.dump(points_for_export, open(filename, 'w'))

    def group_by_subgrids(self, S=1000):
        grids = {}
        for p in self._points:
            x, y = p[0], p[1]
            sx, sy = (x // S) * S, (y // S) * S
            grids.setdefault((sx, sy), []).append(p)
        return {g: np.array(l) for g,l in grids.items()}

    def compress_and_dump_to_float32_buffer(self, filename, S=200, use_compression=True):
        N = len(self._points)
        subgrids = self.group_by_subgrids(S)
        points = {
            'position': [],
            'size': [],
            'color': []
        }
        lines = {
            'position': []
        }
        triangles = {
            'position': [],
            'color': []
        }
        for (x0, y0), point_array in subgrids.items():
            x1, y1 = x0 + S, y0 + S
            rect = (x0, y0, x1, y1)
            sg = LidarSubgrid(point_array, rect)

            # Reduce as much of the pointcloud to triangles as possible
            if use_compression:
                #sg.compress_stage2()
                sg.compress_stage1_floor(nearness_threshhold=100)

            # Add points to the float32 buffer
            # TODO: could be accomplished through np.reshape?
            for p in sg._points:
                # Float32Buffer, size=3
                for pc in p[:3]:
                    points['position'].append(float(pc))

                # Float32Buffer, size=3, color values normalized to 0 -> 1
                rgb = rgb_for_classification(p[6])
                for v in rgb:
                    points['color'].append(v / 256)

                # Float32Buffer, size=1, millimeters
                points['size'].append(10)

            for x in sg._lines:
                lines['position'].append(x)

            # TODO: right now subgrid triangles are assumed to
            # be in float32 buffer form, so change this when that
            # is no longer the case
            for x in sg._triangles:
                triangles['position'].append(x)

            T = int(len(sg._triangles)) // 3
            T_rgb = rgb_for_classification(sg.classification)
            for _ in range(T):
                for v in T_rgb:
                    triangles['color'].append(v / 256)

        # Calculate values after compression
        N1 = len(points['size'])
        debug_log(f'Point cloud size reduced from {N} -> {N1}.')

        output_obj = {
            'points': points,
            'lines': lines,
            'triangles': triangles
        }
        json.dump(output_obj, open(filename, 'w'))


def run_small_1815_original():
    lidar = LidarGrid('1815')
    lidar.load_from_json_file(
        '/Users/albert/Development/dclidar/data/1815.sample2.json'
        )
    lidar.compress_and_dump_to_float32_buffer(
        '/Users/albert/Development/dclidar/data/1815.small_uncompressed.json',
        use_compression=False
        )


def run_small_1815():
    lidar = LidarGrid('1815')
    lidar.load_from_json_file(
        '/Users/albert/Development/dclidar/data/1815.sample2.json'
        )
    lidar.compress_and_dump_to_float32_buffer(
        '/Users/albert/Development/dclidar/data/1815.linecompressed.json',
        S=200
        )


def run_medium_1815():
    lidar = LidarGrid('1815')
    lidar.load_from_json_file(
        '/Users/albert/Development/dclidar/data/1815.sample1.json'
        )
    lidar.compress_and_dump_to_float32_buffer(
        '/Users/albert/Development/dclidar/data/1815.medium_buffered.json',
        S=200
        )


def run_1815():
    lidar = LidarGrid('1815')
    lidar.load_from_lasfile()
    lidar.normalize_to_grid_center()
    lidar.compress_and_dump_to_float32_buffer('/Users/albert/Development/dclidar/data/1815.buffered.json', S=200)


def run_2518():
    lidar = LidarGrid('2518')
    lidar.load_from_lasfile()
    lidar.normalize_to_grid_center()
    #lidar.clip(0.5)
    lidar.compress_and_dump_to_float32_buffer('/Users/albert/Development/dclidar/data/2518.buffered.json', S=200)


if __name__ == '__main__':
    run_2518()