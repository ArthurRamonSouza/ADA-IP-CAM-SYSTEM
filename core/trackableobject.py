class TrackableObject:
    def __init__(self, objectID, centroid):
        """
        Initialize a trackable object with a unique ID and its first centroid.
        """
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False
        self.direction = None

    def update(self, new_centroid):
        """
        Update the object's centroid list and calculate direction.
        """
        self.centroids.append(new_centroid)

        # Optional: Calculate direction based on centroids
        if len(self.centroids) > 1:
            dx = new_centroid[0] - self.centroids[-2][0]
            dy = new_centroid[1] - self.centroids[-2][1]
            if abs(dy) > abs(dx):  # Vertical movement
                self.direction = "up" if dy < 0 else "down"
            else:  # Horizontal movement
                self.direction = "left" if dx < 0 else "right"

    def is_counted(self):
        """
        Check if the object has been counted.
        """
        return self.counted

    def mark_counted(self):
        """
        Mark the object as counted.
        """
        self.counted = True
