import omf
import omfvista
import pyvista as pv
import omf.compat.omf_v1 as o_omf

proj = omfvista.load_project("../assets/test_file.omf")
assay = proj["wolfpass_WP_assay"]
topo = proj["Topography"]
dacite = proj["Dacite"]
vol = proj["Block Model"]
proj.plot(multi_colors=True)
# Create a plotting window
p = pv.Plotter(notebook=False)
# Add the bounds axis
p.show_bounds()
# p.add_bounding_box()

# Add our datasets
p.add_mesh(topo, opacity=0.5)
p.add_mesh(
    dacite,
    color="orange",
    opacity=0.6,
)
thresh_vol = vol.threshold([0.2, 5])

p.add_mesh(thresh_vol, cmap="coolwarm", clim=vol.get_data_range())

# Add the assay logs: use a tube filter that varius the radius by an attribute
p.add_mesh(assay.tube(radius=3), cmap="viridis")

p.show()
