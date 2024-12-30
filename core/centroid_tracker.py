import numpy as np
from scipy.spatial import distance

class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=50):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Update the tracker with new bounding box rectangles.
        `rects` is a list of bounding boxes in the form [(startX, startY, endX, endY), ...].
        """

        # Mark all existing objects as disappeared
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        # Register each new centroid as a new object for the first time
        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            # Grab existing object IDs and their centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute the distance matrix between old and new centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Find the smallest distance for each pair and match them
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            # Track which rows and columns have been examined
            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                # If the distance is too large, skip the pair
                if D[row, col] > self.max_distance:
                    continue

                # Update the centroid for the matched object
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Find unmatched rows and columns
            unused_rows = set(range(0, D.shape[0])) - used_rows
            unused_cols = set(range(0, D.shape[1])) - used_cols

            # Mark unmatched existing objects as disappeared
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Register new objects for unmatched input centroids
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects
