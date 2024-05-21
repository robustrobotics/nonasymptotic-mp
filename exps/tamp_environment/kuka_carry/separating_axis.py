import numpy as np
import pybullet_tools.utils as pbu

def separating_axis_theorem(boxes1, boxes2):
    # Extract center, half-extents, and orientation from the vectorized representations
    centers1, half_extents1, orientations1 = boxes1[:, :3], boxes1[:, 3:6], boxes1[:, 6:].reshape(-1, 3, 3)
    centers2, half_extents2, orientations2 = boxes2[:, :3], boxes2[:, 3:6], boxes2[:, 6:].reshape(-1, 3, 3)

    # Compute the axes to test for separation
    axes = np.zeros((boxes1.shape[0], 15, 3))
    for i in range(3):
        axes[:, i] = orientations1[:, :, i]
        axes[:, i + 3] = orientations2[:, :, i]
        for j in range(3):
            axis = np.cross(orientations1[:, :, i], orientations2[:, :, j])
            axis_norms = np.linalg.norm(axis, axis=1)
            valid_axes = axis_norms > 0
            axis[valid_axes] /= axis_norms[valid_axes, np.newaxis]
            axes[:, 6 + i * 3 + j] = axis

    # Compute the vertices of each box
    vertices1 = compute_vertices_vectorized(centers1, half_extents1, orientations1)
    vertices2 = compute_vertices_vectorized(centers2, half_extents2, orientations2)
    separated = np.ones(boxes1.shape[0], dtype=bool)

    # Test for separation along each axis
    for i in range(15):
        axis = axes[:, i]

        # Project the vertices of each box onto the axis
        projections1 = np.sum(vertices1 * axis[:, np.newaxis, :], axis=-1)
        projections2 = np.sum(vertices2 * axis[:, np.newaxis, :], axis=-1)

        min1 = np.min(projections1, axis=1)
        max1 = np.max(projections1, axis=1)
        min2 = np.min(projections2, axis=1)
        max2 = np.max(projections2, axis=1)

        # Check for separation and update the separation status
        separated = separated & ~(np.logical_or(max1 < min2, max2 < min1))

    return separated


def compute_vertices_vectorized(centers, half_extents, orientations):
    signs = np.array([[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                      [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]])
    vertices = centers[:, np.newaxis, :] + np.sum(
        half_extents[:, np.newaxis, :, np.newaxis] * signs[np.newaxis, :, np.newaxis, :] * orientations[:, np.newaxis, :, :],
        axis=-1
    )
    return vertices

class OBB:
    def __init__(self, center, half_extents, orientation):
        self.center = center
        self.half_extents = half_extents
        self.orientation = orientation

    def to_vectorized(self):
        vectorized = np.concatenate([self.center, self.half_extents, self.orientation.flatten()])
        return vectorized

    @staticmethod
    def from_oobb(oobb:pbu.OOBB):
        aabb, pose = oobb
        return OBB(center=pose[0], half_extents=pbu.get_aabb_extent(aabb)/2.0, orientation=pbu.matrix_from_quat(pose[1]))


if __name__=='__main__':
    boxes1 = []
    boxes2 = []
    # Test case 1: No collision (boxes separated along the x-axis)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    boxes2.append(OBB(center=np.array([2.5, 0, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=np.eye(3)).to_vectorized())
    # Expected output: False

    # Test case 2: No collision (boxes separated along the y-axis)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    boxes2.append(OBB(center=np.array([0, 2.5, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=np.eye(3)).to_vectorized())

    # Expected output: False

    # Test case 3: Collision (boxes intersecting)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    boxes2.append(OBB(center=np.array([0.5, 0.5, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=np.eye(3)).to_vectorized())

    # Expected output: True

    # # Test case 5: No collision (boxes with different orientations)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    rotation_matrix = np.array([[0.707, -0.707, 0],
                                [0.707, 0.707, 0],
                                [0, 0, 1]])
    boxes2.append(OBB(center=np.array([2.5, 0, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=rotation_matrix).to_vectorized())

    # Expected output: False

    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    rotation_matrix = np.array([[0.707, -0.707, 0],
                                [0.707, 0.707, 0],
                                [0, 0, 1]])
    boxes2.append(OBB(center=np.array([1.5, 0, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=rotation_matrix).to_vectorized())

    # Expected output: True

    # Test case 7: No collision (boxes with different orientations and positions)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    rotation_matrix = np.array([[0.866, -0.5, 0],
                                [0.5, 0.866, 0],
                                [0, 0, 1]])
    boxes2.append(OBB(center=np.array([2.5, 1.5, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=rotation_matrix).to_vectorized())

     # Expected output: False

    # Test case 8: Collision (boxes with different orientations and positions)
    boxes1.append(OBB(center=np.array([0, 0, 0]),
            half_extents=np.array([1, 1, 1]),
            orientation=np.eye(3)).to_vectorized())

    rotation_matrix = np.array([[0.866, -0.5, 0],
                                [0.5, 0.866, 0],
                                [0, 0, 1]])
    boxes2.append(OBB(center=np.array([1.5, 1, 0]),
            half_extents=np.array([0.5, 0.5, 0.5]),
            orientation=rotation_matrix).to_vectorized())

    # Expected output: True

    collision = separating_axis_theorem(np.stack(boxes1, axis=0), np.stack(boxes2, axis=0))
    print("Collision:", collision)  # Expected output: False