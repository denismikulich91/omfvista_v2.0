import omfvista
import pyvista as pv
project = omfvista.load_project("../assets/test_file.omf")

vol = project['Block Model']
assay = project['wolfpass_WP_assay']
topo = project['Topography']
dacite = project['Dacite']

p = pv.Plotter(notebook=False)
# Add our datasets
p.add_mesh(topo, cmap='gist_earth', opacity=0.5)
p.add_mesh(assay, color='blue', line_width=3)
p.add_mesh(dacite, color='yellow', opacity=0.6)
# Add the volumetric dataset with a thresholding tool
p.add_mesh_threshold(vol)
# Add the bounds axis
p.show_bounds()
# Redner the scene in a pop out window
p.show()