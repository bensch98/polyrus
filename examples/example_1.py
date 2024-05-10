import numpy as np
import open3d as o3d
from polyrus import TriangleMesh

if __name__ == "__main__":
    tmesh = TriangleMesh(o3d.io.read_triangle_mesh("test/2900326.off"))
    tmesh.mask = tmesh.segmentation_mask("test/2900326_label.txt")
    compfeatures, _ = tmesh.crop(tmesh.mask, filter_threshold=100)
    tmesh.feature_meshes = compfeatures
    
    for mesh in tmesh[1]:
        cip, cep = mesh.shifting_plane()
        pcd = o3d.cuda.pybind.geometry.PointCloud()
        pcd.points = o3d.cuda.pybind.utility.Vector3dVector(np.vstack((cip, cep)))
        mesh.show(misc=[pcd])