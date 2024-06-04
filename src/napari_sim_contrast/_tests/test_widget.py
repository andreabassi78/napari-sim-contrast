from napari_roi_registration import contrast_measurement, phase_estimation
import numpy as np
from napari.layers import Points, Image

def test_constrast(make_napari_viewer, capsys):
    
    viewer = make_napari_viewer()
    im_data = np.random.random((10, 200, 200))   
    image_layer = viewer.add_image(im_data)

    contrast_widget = contrast_measurement()
    
    # contrast_widget(viewer, viewer.layers[0], viewer.layers[1])
    # out, err = capsys.readouterr()
    # assert err == ''
    