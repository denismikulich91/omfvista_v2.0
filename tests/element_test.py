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
    name="gridsurf"
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
    name="vol",
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


class TestElements(unittest.TestCase):
    """
    This creates a dummy OMF project of random data
    """

    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.project_filename = os.path.join(self.test_dir, "project.omf")
        omf.save(PROJECT, self.project_filename)
        self.vtm_filename = os.path.join(self.test_dir, "project.vtm")

    def tearDown(self):
        # Remove the test data directory after the test
        shutil.rmtree(self.test_dir)

    def _check_multi_block(self, proj):
        self.assertEqual(proj.n_blocks, len(PROJECT.elements))
        self.assertEqual(proj.get_block_name(0), "Random Points")
        self.assertEqual(proj.get_block_name(1), "Random Line")
        self.assertEqual(proj.get_block_name(2), "trisurf")
        self.assertEqual(proj.get_block_name(3), "gridsurf")
        self.assertEqual(proj.get_block_name(4), "vol")
        self.assertEqual(proj.get_block_name(5), "vol_ir")

    def test_file_io(self):
        # Write out the project using omf
        omf.load(self.project_filename)
        # Read it back in using omfvista
        proj = omfvista.load_project(self.project_filename)
        print("name check", proj.get_block_name(5))
        self._check_multi_block(proj)

    def test_wrap_project(self):
        proj = omfvista.wrap(PROJECT)
        self._check_multi_block(proj)

    def test_wrap_list_of_elements(self):
        proj = omfvista.wrap(PROJECT.elements)
        self._check_multi_block(proj)

    def test_wrap_lineset(self):
        line = omfvista.wrap(LINESET)
        self.assertTrue(isinstance(line, pyvista.PolyData))
        # Note that omfvista adds a `Line Index` array
        self.assertEqual(line.n_arrays, len(LINESET.attributes) + 1)
        self.assertEqual(line.n_cells, LINESET.num_cells)
        self.assertEqual(line.n_points, LINESET.num_nodes)

    def test_wrap_pointset(self):
        pts = omfvista.wrap(POINTSET)
        self.assertTrue(isinstance(pts, pyvista.PolyData))
        self.assertEqual(pts.n_arrays, len(POINTSET.attributes))
        self.assertEqual(pts.n_cells, POINTSET.num_cells)
        self.assertEqual(pts.n_points, POINTSET.num_nodes)

    def test_wrap_surface(self):
        surf = omfvista.wrap(SURFACE)
        self.assertTrue(isinstance(surf, pyvista.PolyData))
        self.assertEqual(surf.n_arrays, len(SURFACE.attributes))
        self.assertEqual(surf.n_cells, SURFACE.num_cells)
        self.assertEqual(surf.n_points, SURFACE.num_nodes)
        grid = omfvista.wrap(GRID)
        self.assertTrue(isinstance(grid, pyvista.StructuredGrid))
        self.assertEqual(grid.n_arrays, len(GRID.attributes))
        self.assertEqual(grid.n_cells, GRID.num_cells)
        self.assertEqual(grid.n_points, GRID.num_nodes)

    def test_wrap_volume(self):
        # TODO: test self.assertTrue(isinstance(vol, pyvista.RectilinearGrid)) what is it?
        vol = omfvista.wrap(VOLUME)
        self.assertEqual(vol.n_arrays, 1)
        self.assertTrue(isinstance(vol, pyvista.StructuredGrid))
        self.assertEqual(vol.n_arrays, len(VOLUME.attributes))
        self.assertEqual(vol.n_cells, VOLUME.num_cells)
        self.assertEqual(vol.n_points, VOLUME.num_nodes)
        vol_ir = omfvista.wrap(VOLUME_IR)
        self.assertEqual(vol_ir.n_arrays, 1)
        self.assertTrue(isinstance(vol_ir, pyvista.StructuredGrid))
        self.assertEqual(vol_ir.n_arrays, len(VOLUME_IR.attributes))
        self.assertEqual(vol_ir.n_cells, VOLUME_IR.num_cells)
        self.assertEqual(vol_ir.n_points, VOLUME_IR.num_nodes)


if __name__ == "__main__":
    import unittest

    unittest.main()
