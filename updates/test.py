# import omfvista
# import pyvista as pv
# proj = omfvista.load_project("../assets/test_file.omf")
# assay = proj["wolfpass_WP_assay"]
# topo = proj["Topography"]
# dacite = proj["Dacite"]
# vol = proj["Block Model"]
# # proj.plot(multi_colors=True)
# # Create a plotting window
# p = pv.Plotter(notebook=False)
# # Add the bounds axis
# # p.show_bounds()
# # p.add_bounding_box()
#
# # Add our datasets
# p.add_mesh(topo, opacity=0.5)
# p.add_mesh(
#     dacite,
#     color="orange",
#     opacity=0.6,
# )
#
# thresh_vol = vol.threshold([0.5, 5])
#
# p.add_mesh(thresh_vol, cmap="coolwarm", clim=vol.get_data_range())
#
# # Add the assay logs: use a tube filter that varius the radius by an attribute
# p.add_mesh(assay.tube(radius=3), cmap="viridis")
#
# p.show()
import os
import shutil
import tempfile
import unittest

import numpy as np
import omf
import pyvista

import omfvista

PROJECT = omf.Project(name="Test project", description="Just some assorted elements")

POINTSET = omf.PointSet()
POINTSET.name = "Random Points"
POINTSET.description = "Just random points"
POINTSET.vertices = np.random.rand(100, 3)
POINTSET.color = "green"
POINTSET.attributes.append(omf.NumericAttribute(name="More rand data", array=np.random.rand(100), location="vertices"))
POINTSET.attributes.append(omf.NumericAttribute(name="rand data", array=np.random.rand(100), location="vertices"))
#                           textures=[
#                                   omf.ImageTexture(
#                                           name='test image',
#                                           image='test_image.png',
#                                           origin=[0, 0, 0],
#                                           axis_u=[1, 0, 0],
#                                           axis_v=[0, 1, 0]
#                                           ),
#                                   omf.ImageTexture(
#                                           name='test image',
#                                           image='test_image.png',
#                                           origin=[0, 0, 0],
#                                           axis_u=[1, 0, 0],
#                                           axis_v=[0, 0, 1]
#                                           )
#                                   ],

LINESET = omf.LineSet()
LINESET.name = "Random Line"
LINESET.vertices = np.random.rand(100, 3)
LINESET.segments = np.floor(np.random.rand(50, 2) * 100).astype(int)
LINESET.attributes.append(omf.NumericAttribute(name="rand vert data", array=np.random.rand(100), location="vertices"))
LINESET.attributes.append(omf.NumericAttribute(name="rand segment data", array=np.random.rand(50), location="segments"))
LINESET.color = "#0000FF"


SURFACE = omf.Surface()
SURFACE.name = "trisurf"
SURFACE.vertices = np.random.rand(100, 3)
SURFACE.triangles = np.floor(np.random.rand(50, 3) * 100).astype(int)
SURFACE.attributes.append(omf.NumericAttribute(name="rand vert data", array=np.random.rand(100), location="vertices"))
SURFACE.attributes.append(omf.NumericAttribute(name="rand face data", array=np.random.rand(50), location="faces"))
SURFACE.color = [100, 200, 200]

GRID = omf.TensorGridSurface(
    tensor_u=np.ones(10).astype(float),
    tensor_v=np.ones(15).astype(float),
    origin=[50.0, 50.0, 50.0],
    axis_u=[1.0, 0, 0],
    axis_v=[0, 0, 1.0],
    offset_w=np.random.rand(11, 16).flatten(),
)
GRID.attributes.append(omf.NumericAttribute(
    name="rand vert data",
    array=np.random.rand(11, 16).flatten(),
    location="vertices"))

GRID.attributes.append(omf.NumericAttribute(
    name="rand face data",
    array=np.random.rand(10, 15).flatten(order="F"),
    location="faces"))

# GRID.textures=[
#     omf.ImageTexture(
#         name='test image',
#         image='test_image.png',
#         origin=[2., 2., 2.],
#         axis_u=[5., 0, 0],
#         axis_v=[0, 2., 5.]
#     )
# ]
# TODO: test more BM types
VOLUME = omf.TensorGridBlockModel(
    name="vol_ir",
    axis_u=[1, 1, 0],
    axis_v=[0, 0, 1],
    axis_w=[1, -1, 0],
    tensor_u=np.ones(10).astype(float),
    tensor_v=np.ones(15).astype(float),
    tensor_w=np.ones(20).astype(float),
    origin=[10.0, 10.0, -10])

VOLUME.attributes.append(omf.NumericAttribute(
            name="Random Data",
            location="cells",
            array=np.random.rand(10, 15, 20).flatten()))


VOLUME_IR = omf.TensorGridBlockModel(
    name="vol_ir",
    axis_u=[1, 1, 0],
    axis_v=[0, 0, 1],
    axis_w=[1, -1, 0],
    tensor_u=np.ones(10).astype(float),
    tensor_v=np.ones(15).astype(float),
    tensor_w=np.ones(20).astype(float),
    origin=[10.0, 10.0, -10])

VOLUME_IR.attributes.append(omf.NumericAttribute(
            name="Random Data",
            location="cells",
            array=np.random.rand(10, 15, 20).flatten()))


PROJECT.elements = [POINTSET, LINESET, SURFACE, GRID, VOLUME, VOLUME_IR]
if not PROJECT.validate():
    raise AssertionError("Testing data is not valid.")

omf.save(PROJECT, "./test2.omf")
omf.load("./test2.omf")