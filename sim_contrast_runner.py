from src.napari_sim_contrast._widget import contrast_measurement, phase_estimation
'''
Script that runs the napari plugin from the IDE. 
It is not executed when the plugin runs.
'''
if __name__ == '__main__':
    
    import napari
    viewer = napari.Viewer()
    contrast_widget = contrast_measurement()
    phase_widget = phase_estimation()
    viewer.window.add_dock_widget(contrast_widget, name = 'Contrast calculation',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(phase_widget, name = 'Phase estimation',
                                  area='right', add_vertical_stretch=False)
    napari.run()
    