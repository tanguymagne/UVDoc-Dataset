import cv2
import numpy as np
import scipy
from sklearn import decomposition
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

GRID_SIZE = [89, 61]
# GRID_SIZE = [45, 31]
HIDE_TQDM = True


def getPCA(points):
    """
    Get PCA components and singular values of a set of points.
    Args:
        points is an array of shape n_points x 2
    """
    pca = decomposition.PCA()
    points_normalized = points / np.max(points)
    pca = pca.fit(points_normalized)
    comp = pca.components_
    sv = pca.singular_values_
    if np.abs(comp[0, 1]) < np.abs(comp[0, 0]):
        comp[0], comp[1] = comp[1], comp[0]
    if comp[0, 1] < 0:
        comp[0] = -comp[0]
    if comp[1, 0] < 0:
        comp[1] = -comp[1]
    return comp, sv


def get_offsets(normal, point):
    """
    Find the offset that defines the lines defined by normal and passing
    through point.
    Args:
        point is an array of shape n_point x 2
        normal is an array of shape 2
    """
    return -point @ normal


def eval_lines(normal, offsets, point):
    """
    Find the lines defines by normal and offsets at points.
    Args:
        normal is an array of shape 2
        offset is an array of shape n_lines
        point is an array of shape n_points, 2
    """
    return np.expand_dims(point @ normal, 0) + np.expand_dims(offsets, 1)


class KeypointDetector:
    """
    Keypoints detector in the UV image.
    """

    def __init__(self, channel="G") -> None:
        """
        Channel is a string, either 'R', 'G' or 'B'.
        It corresponds to the channel to use for keypoint detection.
        """

        if channel == "R":
            self.channel = 0
        elif channel == "B":
            self.channel = 2
        else:
            self.channel = 1

        self.detectionParams = cv2.SimpleBlobDetector_Params()
        self.setDetectionParams()

    def setDetectionParams(
        self,
        minDistBetweenBlobs=10,
        minThreshold=100,
        maxThreshold=175,
        thresholdStep=5,
        filterByArea=True,
        minArea=75,
        maxArea=250,
        filterByCircularity=True,
        minCircularity=0.6,
        maxCircularity=1.0,
        filterByConvexity=True,
        minConvexity=0.9,
        maxConvexity=1.0,
        filterByInertia=True,
        minInertiaRatio=0.15,
        maxInertiaRatio=1.0,
    ):
        """
        Set detection parameters for the detector.
        """
        # Set Minimum distance between blobs
        self.detectionParams.minDistBetweenBlobs = minDistBetweenBlobs
        # Set threshold for binarizing images
        self.detectionParams.minThreshold = minThreshold
        self.detectionParams.maxThreshold = maxThreshold
        self.detectionParams.thresholdStep = thresholdStep
        # Filter by Area
        self.detectionParams.filterByArea = filterByArea
        self.detectionParams.minArea = minArea
        self.detectionParams.maxArea = maxArea
        # Filter by Circularity
        self.detectionParams.filterByCircularity = filterByCircularity
        self.detectionParams.minCircularity = minCircularity
        self.detectionParams.maxCircularity = maxCircularity
        # Filter by Convexity
        self.detectionParams.filterByConvexity = filterByConvexity
        self.detectionParams.minConvexity = minConvexity
        self.detectionParams.maxConvexity = maxConvexity
        # Filter by Inertia
        self.detectionParams.filterByInertia = filterByInertia
        self.detectionParams.minInertiaRatio = minInertiaRatio
        self.detectionParams.maxInertiaRatio = maxInertiaRatio

    def detectPoints(self, imageUV):
        """
        Detect the dots in a UV-lit image.
        """
        # Create a detector with the parameter
        detector = cv2.SimpleBlobDetector_create(self.detectionParams)

        # Apply a percentile filter on the selected channel of the image
        percentil_filter = scipy.ndimage.percentile_filter(
            imageUV[:, :, self.channel], 95, size=3
        )
        process = 255 - np.repeat(
            np.expand_dims(percentil_filter, -1), 3, axis=-1
        )
        self.keypoints = list(detector.detect(process))
        return self.keypoints


class KeypointSorter:
    MASK_THRESHOLD = 30

    def __init__(self, keypoints) -> None:
        self.coord = keypoints

    def setKeypoints(self, keypoints):
        """
        Set the list of keypoints.
        keypoints is an array of shape Nx2
        """
        self.coord = keypoints
        if len(self.coord) == np.product(GRID_SIZE):
            return True
        else:
            return False

    def findTopLeftCorner(self):
        """
        Find the top left corner out of the list of points.
        """
        # Compute the diagonal direction of the set of points
        comp, sv = getPCA(self.coord)
        d = -(sv[1] * comp[0] + sv[0] * comp[1])

        offsets = get_offsets(d, self.coord)
        level_set = eval_lines(d, offsets, self.coord) >= 0

        # Calculate the number of points on each side of the diagonal line
        number_halfspace_point = np.sum(level_set, axis=-1)
        min_halfspace_points = np.min(number_halfspace_point)
        top_lefts = np.where(min_halfspace_points == number_halfspace_point)[0]
        if len(top_lefts) > 1:
            raise ValueError("Found several top left corners ...")
        else:
            self.top_left_index = top_lefts[0]

    def findBorders(self, mask=None, imageUV=None):
        """
        Find border points in the set of keypoints and order them.
        Assume the top left corner has already been found.
        """
        if mask is None and imageUV is None:
            raise AttributeError(
                "Either 'mask' or 'imageUV' should be provided."
            )
        # Create the mask if not provided
        if mask is None:
            mask_ = (imageUV[:, :, 1] > self.MASK_THRESHOLD).astype(int)
            mask_filter = scipy.ndimage.percentile_filter(
                mask_, 5, size=3
            ).astype(np.uint8)
            mask = np.repeat(np.expand_dims(mask_filter, -1), 3, axis=-1) * 255

        # Convert to grey scale
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Find contour of the paper sheet
        contours, _ = cv2.findContours(
            image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
        )
        contour = [c for c in contours if len(c) > 100]
        if len(contour) > 1:
            raise ValueError("More than one contour found")
        self.contour = contour[0][:, 0, :]

        # Find all border points
        NN = NearestNeighbors(n_neighbors=1)
        NN = NN.fit(self.coord)
        nearests = NN.kneighbors(self.contour, return_distance=False).flatten()

        border_indice = []
        border_appearance_hist = []
        for nearest in nearests[::-1]:
            if border_indice == []:
                border_indice.append(nearest)
                border_appearance_hist.append(1)
            elif nearest == border_indice[-1]:
                border_appearance_hist[-1] += 1
            elif nearest == border_indice[0]:
                border_appearance_hist[0] += 1
            else:
                border_indice.append(nearest)
                border_appearance_hist.append(1)

        to_remove = len(border_indice) - 2 * (np.sum(GRID_SIZE) - 2)
        less_appearance = np.argpartition(
            np.array(border_appearance_hist), to_remove
        )[:to_remove]
        sorted_index_less_appearance = np.sort(less_appearance)
        for ind in sorted_index_less_appearance[::-1]:
            border_indice.pop(ind)

        # Order the border points
        begin = np.where(border_indice == self.top_left_index)[0][0]
        ordered_border = border_indice[begin:] + border_indice[:begin]
        self.already_ordered = ordered_border

        self.grid_index = -np.ones(GRID_SIZE, dtype=int)
        i = 0
        j = 0
        i_up = True
        j_up = True
        for k in range(len(ordered_border)):
            self.grid_index[i, j] = ordered_border[k]
            if j < GRID_SIZE[1] - 1 and j_up:
                j += 1
                if j == GRID_SIZE[1] - 1:
                    j_up = False
            elif j == GRID_SIZE[1] - 1 and i_up:
                i += 1
                if i == GRID_SIZE[0] - 1:
                    i_up = False
            elif j > 0:
                j -= 1
            elif i > 0:
                i -= 1

    def initilize_filling_inside_grid(self):
        """
        Initialize the filling of the inside of the grid.
        """
        self.coord_normalized = self.coord / np.max(self.coord)

        self.i = 1
        self.j = 1

        self.is_filled = False

    def fillCoord_ij(self):
        """
        Fill the coordinate at the current position of the grid.
        """
        # Create a list of all keypoints that are not already ordered
        self.all_not_ordered = set(list(range(len(self.coord)))) - set(
            self.already_ordered
        )
        self.all_not_ordered = list(self.all_not_ordered)
        all_not_ordered_coord_normalized = self.coord_normalized[
            self.all_not_ordered
        ]

        # Nearest Neighbor search over keypoint not already ordered
        NN = NearestNeighbors(n_neighbors=min(3, len(self.all_not_ordered)))
        NN = NN.fit(all_not_ordered_coord_normalized)
        distance, nearest = NN.kneighbors(
            self.coord_normalized[
                [
                    self.grid_index[self.i - 1, self.j],
                    self.grid_index[self.i - 1, self.j - 1],
                    self.grid_index[self.i, self.j - 1],
                ]
            ],
            return_distance=True,
        )
        # Found the new keypoints as the one present in all nearest
        # neighborhood
        found = list(set.intersection(*[set(n) for n in nearest]))
        if len(found) == 1:
            # One have been found, this is the one to put here
            ind = self.all_not_ordered[found[0]]
            self.grid_index[self.i, self.j] = ind
            self.already_ordered.append(ind)
        elif len(found) > 1:
            # Several have been found, the selected one, will be the
            # one with the lowest distance
            dist = [0] * len(found)
            for k, potential in enumerate(found):
                for li, neighbor in enumerate(nearest):
                    dist[k] += distance[li][
                        np.where(potential == neighbor)[0][0]
                    ]
            found = found[np.argmin(dist)]
            self.grid_index[self.i, self.j] = self.all_not_ordered[found]
            self.already_ordered.append(self.all_not_ordered[found])
        else:
            # None have been found, this is an error
            raise ValueError(
                "No Overlap in neighbors, found no obvious keypoints"
            )

        self.increase_ij()

    def remove_ordered_to(self, index):
        """
        Remove all ordered keypoints from the grid, until the index
        given.
        """
        self.decrease_ij()
        while self.grid_index[self.i, self.j] != index:
            self.grid_index[self.i, self.j] = -1
            self.already_ordered.pop(-1)
            self.decrease_ij()

        self.increase_ij()

    def decrease_ij(self):
        self.j -= 1
        if self.j == 0 and self.i != 1:
            self.i -= 1
            self.j = GRID_SIZE[1] - 2

        if self.j == 0 and self.i == 1:
            raise ValueError("No more points to remove")

    def increase_ij(self):
        self.j += 1
        if self.j == GRID_SIZE[1] - 1 and self.i != GRID_SIZE[0] - 2:
            self.i += 1
            self.j = 1

        if self.j == GRID_SIZE[1] - 1 and self.i == GRID_SIZE[0] - 2:
            self.is_filled = True

    def set_ij_manually(self, index):
        """
        Set the current position of the grid to the index given.
        """
        self.grid_index[self.i, self.j] = index
        self.already_ordered.append(index)
        self.increase_ij()

    def fillInsideGrid(self):
        """
        Order the remaining points (inside of the grid).
        Assume the border points are already ordered.
        """
        while not self.is_filled:
            self.fillCoord_ij()

        self.grid = self.coord[self.grid_index]

    def finalizeGrid(self):
        self.grid = self.coord[self.grid_index]

    def pushBorderPoint(self, p1, p2, distance_from_border=2):
        """
        Find the nearest point on the border in direction p1 - p2.
        """
        direction = p1 - p2
        direction = direction / np.linalg.norm(direction)
        eps = 1e-5
        bottom = -100
        up = 100

        while up - bottom > eps:
            middle = (up + bottom) / 2
            p = p1 + middle * direction
            distance = cv2.pointPolygonTest(self.contour, p, True)
            if distance < distance_from_border:
                up = middle
            elif distance > distance_from_border:
                bottom = middle
            else:
                up = middle
                bottom = middle
        return p1 + bottom * direction

    def getFinalGrid(self, distance_from_border=2):
        """
        Pushes all borders points at distance_from_border pixels to the
        segmentation mask border.
        """
        final_grid = np.copy(self.grid)
        # Pushes top and bottom rows
        for i in range(1, GRID_SIZE[1] - 1):
            final_grid[0, i] = self.pushBorderPoint(
                final_grid[0, i], final_grid[1, i], distance_from_border
            )
            final_grid[-1, i] = self.pushBorderPoint(
                final_grid[-1, i], final_grid[-2, i], distance_from_border
            )
        # Pushes left and right columns
        for j in range(1, GRID_SIZE[0] - 1):
            final_grid[j, 0] = self.pushBorderPoint(
                final_grid[j, 0], final_grid[j, 1], distance_from_border
            )
            final_grid[j, -1] = self.pushBorderPoint(
                final_grid[j, -1], final_grid[j, -2], distance_from_border
            )

        # Pushes corner points
        final_grid[0, 0] = self.pushBorderPoint(
            final_grid[0, 0], final_grid[1, 1], distance_from_border
        )
        final_grid[0, -1] = self.pushBorderPoint(
            final_grid[0, -1], final_grid[1, -2], distance_from_border
        )
        final_grid[-1, 0] = self.pushBorderPoint(
            final_grid[-1, 0], final_grid[-2, 1], distance_from_border
        )
        final_grid[-1, -1] = self.pushBorderPoint(
            final_grid[-1, -1], final_grid[-2, -2], distance_from_border
        )
        self.final_grid = final_grid
