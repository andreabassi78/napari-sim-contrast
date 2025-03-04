from src.napari_sim_contrast._phase_profilometry_widget import phase_profilometry, H5opener,select_phases,count_cycles,preprocesser
'''
Script that runs the napari plugin from the IDE. 
It is not executed when the plugin runs.
'''
if __name__ == '__main__':
    
    import napari
    from magicgui import magicgui
    viewer = napari.Viewer()
    phase_widget = phase_profilometry()
    select_widget = select_phases()
    peak_counter_widget = count_cycles()
    preprocesser_widget = preprocesser()

    h5widget = H5opener(viewer) 

    h5_opener = magicgui(h5widget.open_h5_dataset, call_button='Open h5 dataset')
    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(select_widget, name = 'Select phases',
                                  area='right', add_vertical_stretch=True)
    
    viewer.window.add_dock_widget(peak_counter_widget, name = 'Peak counter',
                                  area='right', add_vertical_stretch=True)
    
    viewer.window.add_dock_widget(preprocesser_widget,name='Preprocessing',
                                  area='right',add_vertical_stretch=True)
    
    viewer.window.add_dock_widget(phase_widget, name = 'Phase profilometry',
                                  area='right', add_vertical_stretch=True)

    napari.run() 
    