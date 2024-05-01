import open3d as o3d

def unpack_dict_list(d):
    return [i for s in d.values() for i in s]

def show(geometries):
    o3d.visualization.draw_geometries([geometries])