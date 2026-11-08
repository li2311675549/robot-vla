"""Footprint-inflated occupancy grid, 8-connected A*, and collision-safe smoothing."""
import heapq
import math
import numpy as np


class NoPath(RuntimeError):
    pass


class OccupancyGrid:
    def __init__(self, bounds=(-6, 3, -3, 3), resolution=.08, radius=.64):
        self.bounds = tuple(bounds)
        self.resolution = resolution
        self.radius = radius
        self.shape = (math.ceil((bounds[1]-bounds[0])/resolution)+1,
                      math.ceil((bounds[3]-bounds[2])/resolution)+1)
        self.occupied = np.zeros(self.shape, dtype=bool)

    def cell(self, point):
        return tuple(np.round((np.asarray(point[:2])-[self.bounds[0], self.bounds[2]])/self.resolution).astype(int))

    def world(self, cell):
        return np.asarray(cell)*self.resolution + [self.bounds[0], self.bounds[2]]

    def free(self, cell):
        return 0 <= cell[0] < self.shape[0] and 0 <= cell[1] < self.shape[1] and not self.occupied[cell]

    def rectangle(self, xmin, xmax, ymin, ymax):
        # Euclidean dilation of an axis-aligned rectangle by the robot bounding
        # circle, including stowed arm and a safety margin. No point-robot model.
        x = self.bounds[0] + np.arange(self.shape[0])*self.resolution
        y = self.bounds[2] + np.arange(self.shape[1])*self.resolution
        dx = np.maximum(np.maximum(xmin-x, x-xmax), 0)
        dy = np.maximum(np.maximum(ymin-y, y-ymax), 0)
        self.occupied |= dx[:, None]**2 + dy[None, :]**2 <= self.radius**2

    def visible(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        count = max(2, int(np.linalg.norm(b-a)*3)+1)
        return all(self.free(tuple(np.round(p).astype(int))) for p in np.linspace(a, b, count))

    def plan(self, start, goal):
        first, last = self.cell(start), self.cell(goal)
        if not self.free(first) or not self.free(last):
            raise NoPath(f"Start or goal inside inflated obstacle: {first}, {last}")
        frontier = [(0., first)]
        cost, parent = {first: 0.}, {}
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == last:
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nxt = (current[0]+dx, current[1]+dy)
                if not self.free(nxt):
                    continue
                if dx and dy and (not self.free((current[0]+dx, current[1])) or not self.free((current[0], current[1]+dy))):
                    continue
                value = cost[current]+math.hypot(dx, dy)
                if value < cost.get(nxt, float('inf')):
                    cost[nxt] = value
                    parent[nxt] = current
                    heapq.heappush(frontier, (value + math.dist(nxt, last), nxt))
        else:
            raise NoPath("A* exhausted reachable grid")
        path = [last]
        while path[-1] != first:
            path.append(parent[path[-1]])
        path.reverse()
        smooth = [path[0]]
        i = 0
        while i < len(path)-1:
            j = len(path)-1
            while j > i+1 and not self.visible(path[i], path[j]):
                j -= 1
            smooth.append(path[j])
            i = j
        result = [self.world(p).tolist() for p in smooth]
        result[0], result[-1] = list(start[:2]), list(goal[:2])
        return result
