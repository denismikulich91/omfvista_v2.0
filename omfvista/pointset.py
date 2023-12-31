"""Methods to convert point set objects to VTK data objects"""


__all__ = [
    "point_set_to_vtk",
]

__displayname__ = "Point Set"

import numpy as np
import pyvista

from omfvista.utilities import add_data, add_texture_coordinates


def point_set_to_vtk(pse, origin=(0.0, 0.0, 0.0)):
    """Convert the point set to a :class:`pyvista.PolyData` data object.

    Args:
        pse (:class:`omf.pointset.PointSet`): The point set to convert

    Return:
        :class:`pyvista.PolyData`
    """
    points = np.array(pse.vertices)
    output = pyvista.PolyData(points)
    add_texture_coordinates(output, pse.textures, pse.name)
    add_data(output, pse.attributes)
    output.points += np.array(origin)
    return output


point_set_to_vtk.__displayname__ = "Point Set to VTK"
