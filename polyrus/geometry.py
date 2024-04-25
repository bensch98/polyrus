import numpy as np
import open3d as o3d

class TriangleMesh():
    """
    Wrapper class around open3d triangle mesh with capabilities for deep learning purposes.

    ```python
    from polyrus import TriangleMesh
    ```
    """
    def __init__(self, mesh: o3d.cuda.pybind.geometry.TriangleMesh | o3d.t.geometry.TriangleMesh = None, mask: np.ndarray = None):
        self.mesh = mesh
        self.mask = mask
        self.vertices, self.faces, self.edges = self.numpy()

    def _edges(self, faces: np.ndarray):
        """
        Parameters:
            faces (np.array): Array of faces, each face defined by indices into the vertices array, shape (n_faces, 3)

        Returns:
            np.array: Edges of the triangle mesh, shape (n_edges, 2)
        """
        edges = np.hstack([
            faces[:, [0, 1]],
            faces[:, [1, 2]],
            faces[:, [2, 0]],
        ]).reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        return edges

    def numpy(self):
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        edges = np.unique(self._edges(faces), axis=0)
        return vertices, faces, edges

    def euler_characteristic(self):
        return self.vertices.shape[0] - self.edges.shape[0] + self.faces.shape[0]
    
    def boundary_edges(self, return_counts: bool = False) -> np.ndarray:
        """
        Finds the boundary edges from the given faces of a mesh.

        Parameters:
            return_counts (bool): If True returns n_boundary_edges

        Returns:
            np.array: Boundary edges, shape(n_boundary_edges, 2)
            int (optional): Number of boundary edges, n_boundary_edges
        """
        unique_edges, counts = np.unique(self._edges(self.faces), axis=0, return_counts=True)
        boundary_edges = unique_edges[counts == 1]
        if return_counts:
            return boundary_edges, len(boundary_edges)
        return boundary_edges

    def boundary_vertices(self, return_counts: bool = False) -> np.ndarray:
        """
        Finds the boundary vertices from the given boundary edges of a mesh.

        Parameters:
            return_counts (bool): If True returns n_boundary_vertices

        Returns:
            np.array: Boundary vertices, shape(n_boundary_vertices, 2)
            int (optional): Number of boundary vertices, n_boundary_vertices
        """
        boundary_vertices = np.unique(self.boundary_edges().flatten())
        if return_counts:
            return boundary_vertices, len(boundary_vertices)
        return np.unique(self.boundary_edges().flatten())

    def boundary_faces(self, return_counts: bool = False) -> np.ndarray:
        """
        Finds the boundary faces from the given boundary edges of a mesh.

        Parameters:
            return_counts (bool): If True returns n_boundary_faces

        Returns:
            np.array: Boundary faces, shape(n_boundary_faces, 2)
            int (optional): Number of boundary faces, n_boundary_faces
        """
        edges = np.vstack([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]],
        ])
        edges = np.sort(edges, axis=1)
        unique_edges, indices, counts = np.unique(edges, return_inverse=True, return_counts=True, axis=0)
        boundary_edge_mask = counts[indices] == 1
        is_boundary_face = boundary_edge_mask.reshape(-1, 3).any(axis=1)
        boundary_face_indices = np.nonzero(is_boundary_face)[0]
        if return_counts:
            return boundary_face_indices, len(boundary_face_indices)
        return boundary_face_indices

    def segmentation_mask(self, fname): pass 
