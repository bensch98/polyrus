import copy
from enum import Enum, auto
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import distinctipy

import polyrus
from polyrus.utils import unpack_dict_list


class Geometry(Enum):
    TRIANGLEMESH = auto()
    LINESET = auto()
    POINTCLOUD = auto()


class TriangleMesh:
    """
    Wrapper class around open3d triangle mesh with capabilities for deep learning purposes.

    ```python
    from polyrus import TriangleMesh
    ```
    """

    def __init__(
        self,
        mesh: o3d.cuda.pybind.geometry.TriangleMesh
        | o3d.t.geometry.TriangleMesh = None,
        mask: np.ndarray = None,
    ):
        self.mesh = mesh
        self.mask = mask
        self.vertices, self.faces, self.edges = self.numpy()
        self.ls = self.to(self.mesh, Geometry.LINESET)

    def _edges(self, faces: np.ndarray):
        """
        Parameters:
            faces (np.array): Array of faces, each face defined by indices into the vertices array, shape (n_faces, 3)

        Returns:
            np.array: Edges of the triangle mesh, shape (n_edges, 2)
        """
        edges = np.hstack(
            [
                faces[:, [0, 1]],
                faces[:, [1, 2]],
                faces[:, [2, 0]],
            ]
        ).reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        return edges

    def numpy(self):
        vertices = np.asarray(self.mesh.vertices)
        faces = np.asarray(self.mesh.triangles)
        edges = np.unique(self._edges(faces), axis=0)
        return vertices, faces, edges

    def n_vfe(self):
        return len(self.vertices), len(self.faces), len(self.edges)

    @staticmethod
    def to(mesh, geometry: Geometry) -> o3d.cuda.pybind.geometry:
        """
        Converts triangle mesh to another geometry.

        Parameters:
            mesh (o3d.cuda.pybind.geometry.TriangleMesh): Triangle mesh to convert to another type.
            geometry (Geometry): Output type (Geometry.TRIANGLEMESH, Geometry.LINESET, Geometry.POINTCLOUD)

        Return:
            o3d.cuda.pybind.geometry: Either a triangle mesh, lineset or point cloud
        """
        if geometry == geometry.TRIANGLEMESH:
            return mesh
        elif geometry == geometry.LINESET:
            return o3d.cuda.pybind.geometry.LineSet().create_from_triangle_mesh(mesh)
        elif geometry == geometry.POINTCLOUD:
            pc = o3d.geometry.PointCloud()
            pc.points = mesh.vertices
            return pc
        else:
            raise ValueError(
                f"Geometry conversion not implemented for type: {geometry.name}"
            )

    def euler_characteristic(self) -> int:
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
        unique_edges, counts = np.unique(
            self._edges(self.faces), axis=0, return_counts=True
        )
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
            np.ndarray: Boundary faces, shape(n_boundary_faces, 2)
            int (optional): Number of boundary faces, n_boundary_faces
        """
        edges = np.vstack(
            [
                self.faces[:, [0, 1]],
                self.faces[:, [1, 2]],
                self.faces[:, [2, 0]],
            ]
        )
        edges = np.sort(edges, axis=1)
        unique_edges, indices, counts = np.unique(
            edges, return_inverse=True, return_counts=True, axis=0
        )
        boundary_edge_mask = counts[indices] == 1
        is_boundary_face = boundary_edge_mask.reshape(-1, 3).any(axis=1)
        boundary_face_indices = np.nonzero(is_boundary_face)[0]
        if return_counts:
            return boundary_face_indices, len(boundary_face_indices)
        return boundary_face_indices

    def segmentation_mask(self, fname: str | Path) -> NDArray[np.uint8]:
        """
        Parameters:
            fname (str | pathlib.Path): Path to the .txt file that contains the vertex labels row-wise.
        Returns:
            np.ndarray: Row-wise array containing the vertex-wise labels.
        """
        self.mask = np.loadtxt(fname, dtype=np.uint8)
        if self.mask.shape[0] != len(self.vertices):
            raise ValueError(
                f"The loaded mask dimensions ({self.mask.shape}) do not match the number of vertices ({len(self.vertices)})"
            )
        return self.mask

    @staticmethod
    def _segmentation_boundary(faces: NDArray, y):
        faces = np.array(faces)
        first_labels = y[faces[:, 0]]
        consistent_labels = y[faces] == first_labels[:, None]
        inconsistent_faces = np.any(~consistent_labels, axis=1)
        inconsistent_face_indices = faces[inconsistent_faces]
        unique_inconsistent_indices = np.unique(inconsistent_face_indices)
        return unique_inconsistent_indices

    def crop(
        self, mask: NDArray[np.uint8], filter_threshold: int = -float("inf")
    ) -> tuple[dict[int, list[o3d.cuda.pybind.geometry]], np.ndarray]:
        # TODO: efficient rewrite required
        if mask.ndim != 1:
            raise ValueError(
                f"Expected mask to be a 1-dimensional array, got {mask.shape}"
            )
        n_classes = len(np.unique(mask))
        compfeatures = {k: [] for k in range(n_classes + 1)}
        # TODO: validate mask before
        segmentation_boundary_indices = TriangleMesh._segmentation_boundary(
            self.faces, mask
        )
        y2 = np.copy(mask)

        for i in range(n_classes, -1, -1):
            if i == 0:
                y2[segmentation_boundary_indices] = 0
            tmpmesh = copy.deepcopy(self.mesh)
            class_indices = np.nonzero(y2 != i)[0]
            tmpmesh.remove_vertices_by_index(class_indices.tolist())
            segmesh = self.mesh.select_by_index(np.where(y2 == i)[0])
            segmesh.compute_vertex_normals()
            clidxi, nfacesi, _ = segmesh.cluster_connected_triangles()
            compmask = np.asarray(clidxi)

            for j, _ in enumerate(nfacesi):
                featmesh = copy.deepcopy(segmesh)
                featmesh.compute_vertex_normals()
                featmesh.remove_triangles_by_index(np.where(compmask != j)[0])
                featmesh = featmesh.remove_unreferenced_vertices()
                compfeatures[i].append(featmesh)
        # filter based on vertices
        compfeatures = {
            key: [mesh for mesh in meshes if len(mesh.vertices) > filter_threshold]
            for key, meshes in compfeatures.items()
        }
        return compfeatures, segmentation_boundary_indices

    def show(self):
        o3d.visualization.draw_geometries([self.mesh])

    def show_segmentation(
        self,
        compfeatures,
        mask: NDArray[np.uint8] = None,
        show_as: Geometry = Geometry.TRIANGLEMESH,
        rand: float = 0,
        colors=None,
    ) -> None:
        def randomize(color, b):
            color = np.array(color)
            noise = np.random.uniform(-b, b, size=color.shape)
            return np.clip(color + noise, 0, 1)

        if not mask:
            mask = self.mask
        if not colors:
            colors = distinctipy.get_colors(np.max(mask) + 1)

        compfeatures = {
            k: [
                TriangleMesh.to(g, show_as).paint_uniform_color(
                    randomize(colors[k], rand)
                )
                for g in geometries
            ]
            for k, geometries in compfeatures.items()
        }
        o3d.visualization.draw_geometries(unpack_dict_list(compfeatures))


if __name__ == "__main__":
    tmesh = TriangleMesh(o3d.io.read_triangle_mesh("test/2900326.off"))
    tmesh.mask = tmesh.segmentation_mask("test/2900326_label.txt")
    compfeatures, _ = tmesh.crop(tmesh.mask)
    tmesh.show_segmentation(compfeatures, colors=polyrus.COLORS_NORMAL, show_as=Geometry.LINESET)
    # tmesh.show_segmentation(compfeatures, rand=0.4)
    # o3d.visualization.draw_geometries([compfeatures[1][0]])
