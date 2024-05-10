import copy
from enum import Enum, auto
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import open3d as o3d
import distinctipy

import polyrus
from polyrus.utils import unpack_dict_list, fit_plane, fit_line, ray_triangle_intersection, point_to_pcd_distance, plane_line_intersection


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
        self.mesh.compute_vertex_normals()
        self.feature_meshes = None
        self.mask = mask
        self.vertices, self.faces, self.edges = self.numpy()
        self.ls = self.to(self.mesh, Geometry.LINESET)
    
    def __getitem__(self, idx):
        return self.feature_meshes[idx]
    
    def __str__(self):
        v, f, e = self.n_vfe()
        return f"TriangleMesh: {v} vertices, {f} faces, {e} edges"
    
    def __repr__(self):
        return self.__str__()
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index < len(self.feature_meshes):
            result = self.feature_meshes[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration

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
        if mask.ndim != 1:
            raise ValueError(
                f"Expected mask to be a 1-dimensional array, got {mask.shape}"
            )
        elif len(mask) != len(self.vertices):
            raise ValueError(
                f"Mask length ({len(mask)}) does not match the number of vertices ({len(self.vertices)})" 
            )
        n_classes = np.max(mask)
        compfeatures = {k: [] for k in range(n_classes + 1)}
        segmentation_boundary_indices = TriangleMesh._segmentation_boundary(
            self.faces, mask
        )

        # TODO: probably possible in an easier way
        for i in range(n_classes, -1, -1):
            if i == 0:
                mask[segmentation_boundary_indices] = 0
            classmesh = self.mesh.select_by_index(np.where(mask == i)[0])
            classmesh.compute_vertex_normals()
            cluster_idxi, n_facesi, _ = classmesh.cluster_connected_triangles()
            cluster_mask = np.asarray(cluster_idxi)

            for j, n_faces in enumerate(n_facesi):
                if not n_faces > filter_threshold:
                    continue
                featmesh = copy.deepcopy(classmesh)
                featmesh.remove_triangles_by_index(np.where(cluster_mask != j)[0])
                featmesh = featmesh.remove_unreferenced_vertices()
                compfeatures[i].append(TriangleMesh(featmesh))
        return compfeatures, segmentation_boundary_indices

    def show(self, idx=None, misc: list = []) -> None:
        if not idx:
            o3d.visualization.draw_geometries([self.mesh] + misc)
        else:
            o3d.visualization.draw_geometries([tm.mesh for tm in self[idx]] + misc)

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
    
    def intersect_plane(self, n: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Args:
            n: normal
            p: point
        """
        distances = np.dot(self.vertices - p, n)
        intersections = []
        for e0, e1 in self.edges:
            d0, d1 = distances[[e0, e1]]
            if d0 * d1 < 0:
                v0, v1 = self.vertices[[e0, e1]]
                t = d0 / (d0 - d1)
                point = v0 + t * (v1 - v0)
                intersections.append(point)
        intersections = np.array(intersections)
        return intersections
    
    def intersect_line(self, v: np.ndarray, t: np.ndarray) -> np.ndarray:
        intersections = []
        for j in [1, -1]:
            for i in range(len(self.faces)):
                triangle = self.vertices[self.faces[i]]
                intersection = ray_triangle_intersection(v*j, t, triangle)
                if len(intersection) != 0:
                    intersections.append(intersection)
        return np.array(intersections)
    
    def farthest_vertex_from_plane(self, n:np.ndarray, d:int) -> np.ndarray:
        distances = np.dot(self.vertices, n) + d
        return self.vertices[np.argmax(np.abs(distances))]
    
    def shifting_plane(self, steps:int=10) -> np.ndarray:
        def filter(x: np.ndarray):
            l2 = np.linalg.norm(x - x.mean(axis=0), axis=1)
            return x[l2 <= np.median(l2)]

        boundary_vertices = self.vertices[self.boundary_vertices()]
        n, d, p0 = fit_plane(boundary_vertices)
        p1 = self.farthest_vertex_from_plane(n, d)
        t_values = np.linspace(0, 1, steps)
        points = np.array([(1-t)*p0 + t*p1 for t in t_values])
        geometric_mediani = []
        for point in points:
            intersections = self.intersect_plane(n, point)
            if len(intersections) != 0:
                geometric_mediani.append(intersections.mean(axis=0))
        geometric_mediani = np.array(geometric_mediani)
        v, t = fit_line(filter(geometric_mediani))
        cip = plane_line_intersection(n, p0, v, t)
        intersections = self.intersect_line(v, t)
        distances = point_to_pcd_distance(cip, intersections)
        cep = intersections[np.argmax(np.abs(distances))]
        return cip, cep
    
    def spherical_boundary_score(self, n:np.ndarray, p:np.ndarray, cip:np.ndarray, cep:np.ndarray, steps:int=10) -> float:
        radii = 0
        t_values = np.linspace(0, 1, steps)
        line = np.array([(1-t)*cip + t*cep for t in t_values])
        for p in line:
            intersections = self.intersect(n, p)
            if np.array_equal(intersections, np.array([])):
                continue
            distances = np.linalg.norm(intersections - p, axis=1)
            radii += distances[np.argmin(distances)]
        return radii / steps

# TODO:
# - [ ] pcd sampling
# - [ ] save point cloud 