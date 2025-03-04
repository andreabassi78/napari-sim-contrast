from src.napari_sim_contrast._widget import contrast_measurement, phase_estimation, H5opener
'''
Script that runs the napari plugin from the IDE. 
It is not executed when the plugin runs.
'''
if __name__ == '__main__':
    
    import napari
    from magicgui import magicgui
    viewer = napari.Viewer()
    contrast_widget = contrast_measurement()
    phase_widget = phase_estimation()

    h5widget = H5opener(viewer) 

    h5_opener = magicgui(h5widget.open_h5_dataset, call_button='Open h5 dataset')
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(contrast_widget, name = 'Contrast calculation',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(phase_widget, name = 'Phase estimation',
                                  area='right', add_vertical_stretch=True)

    napari.run() 
    