# -*- coding: utf-8 -*-
import numpy as np
import math
from scipy.ndimage import gaussian_filter


class Universe:
    def __init__(self, n_elements):
        self.parent = np.arange(n_elements, dtype=np.int32)
        self.rank = np.zeros(n_elements, dtype=np.int32)
        self.size = np.ones(n_elements, dtype=np.int32)
        self.max_weight = np.zeros(n_elements, dtype=np.float32)

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def join(self, i, j, weight):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
                self.size[root_j] += self.size[root_i]
                self.max_weight[root_j] = weight
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
                self.size[root_i] += self.size[root_j]
                self.max_weight[root_i] = weight
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1
                self.size[root_i] += self.size[root_j]
                self.max_weight[root_i] = weight


def build_edges(img):
    height, width, _ = img.shape
    edges = []

    def color_distance(p1, p2):
        return math.sqrt(np.sum((p1 - p2) ** 2))

    for y in range(height):
        for x in range(width):
            idx = y * width + x
            if x < width - 1:
                edges.append((idx, y * width + x + 1, color_distance(img[y, x], img[y, x + 1])))
            if y < height - 1:
                edges.append((idx, (y + 1) * width + x, color_distance(img[y, x], img[y + 1, x])))
            if x < width - 1 and y < height - 1:
                edges.append((idx, (y + 1) * width + x + 1, color_distance(img[y, x], img[y + 1, x + 1])))
            if x > 0 and y < height - 1:
                edges.append((idx, (y + 1) * width + x - 1, color_distance(img[y, x], img[y + 1, x - 1])))
    return edges


def segment_image_felzenszwalb(image, scale=300.0, sigma=0.8, min_size=20):
    image = image.astype(np.float32)
    smoothed = np.zeros_like(image)
    for c in range(image.shape[2]):
        smoothed[:, :, c] = gaussian_filter(image[:, :, c], sigma)

    height, width, _ = smoothed.shape
    num_pixels = height * width
    edges = build_edges(smoothed)
    edges.sort(key=lambda e: e[2])

    universe = Universe(num_pixels)
    threshold = np.full(num_pixels, scale / 1.0, dtype=np.float32)

    for u, v, weight in edges:
        ru, rv = universe.find(u), universe.find(v)
        if ru != rv:
            if weight <= threshold[ru] and weight <= threshold[rv]:
                universe.join(ru, rv, weight)
                nr = universe.find(ru)
                universe.max_weight[nr] = weight
                threshold[nr] = weight + (scale / universe.size[nr])

    for u, v, weight in edges:
        ru, rv = universe.find(u), universe.find(v)
        if ru != rv:
            if universe.size[ru] < min_size or universe.size[rv] < min_size:
                universe.join(ru, rv, weight)

    labels = np.array([universe.find(y * width + x) for y in range(height) for x in range(width)], dtype=np.int32)
    return labels.reshape(height, width)
