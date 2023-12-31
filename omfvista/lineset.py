"""Methods to convert line set objects to VTK data objects"""


__all__ = [
    "line_set_to_vtk",
]

__displayname__ = "Line Set"

import numpy as np
import pyvista

from omfvista.utilities import add_data


def line_set_to_vtk(lse, origin=(0.0, 0.0, 0.0)):
    """Convert the line set to a :class:`pyvista.PolyData` data object.

    Args:
        lse (:class:`omf.lineset.LineSet`): The line set to convert

    Return:
        :class:`pyvista.PolyData`
    """

    ids = np.array(lse.segments.array).reshape(-1, 2).astype(np.int_)
    lines = np.c_[np.full(len(ids), 2, dtype=np.int_), ids]

    output = pyvista.PolyData()
    output.points = np.array(lse.vertices.array)
    output.lines = lines

    indices = output.connectivity().cell_data["RegionId"]
    output["Line Index"] = indices
    # Now add data to lines:
    # output[lse.name] = indices
    # output = pyvista.PolyData(points)
    print(len(lse.segments.array))
    # add_data(output, lse)

    # TODO: if subtype is borehole make a tube

    output.points += np.array(origin)
    print(output)
    return output


line_set_to_vtk.__displayname__ = "Line Set to VTK"
