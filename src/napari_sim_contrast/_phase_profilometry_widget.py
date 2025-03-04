"""
Created on Tue Jun 6 15:58:58 2024

@author: Paolo Maran and Andrea Bassi @Polimi

Napari widget for SIM images contrast estimation and phase calibration

"""

import numpy as np
from magicgui import magic_factory, magicgui
from napari import Viewer
from napari.layers import Image, Shapes
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.signal as sc
from enum import Enum
from src.napari_sim_contrast.get_h5_data import get_h5_dataset, get_h5_attr, get_datasets_index_by_name, get_group_name
import os
from qtpy.QtWidgets import  QWidget
import pathlib
from scipy import fftpack as fft
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
 


def find_max(img):
    return divmod(np.argmax(img), img.shape[1])

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def naive_unwrap2d(image):
    sy,sx = np.shape(image)

    #the horizontal behaviour of the image is obtained via the np.unwrap function
    _unwrapped = np.unwrap(image,axis=1)

    unwrap_h_ref = _unwrapped[:,0]
    unwrap_h_meas = np.unwrap(image[:,0])



    unwrapped = _unwrapped + unwrap_h_meas[:,None] - unwrap_h_ref[:,None]

    return unwrapped

def _image_preprocesser(image,frame_factor):
    sy,sx = np.shape(image)

    img_fft = fft.fftshift(fft.fft2(image))
    
    img_fft[0:int(sx*frame_factor),:] = 0
    img_fft[int(sx*(1-frame_factor)),:] = 0
    img_fft[:,0:int(sy*frame_factor)] = 0
    img_fft[:,int(sy*(1-frame_factor))] = 0

    filtered = fft.ifft2(img_fft)

    return np.abs(filtered)




@magic_factory(call_button="Phase profilometry")
def phase_profilometry(viewer: Viewer,
                    image: Image, 
                    roi: Shapes,
                    num_phases:int= 6,
                    angle:float=30,
                    try_unwrapping:bool = False,
                    use_phase_estimate:bool = True,
                    cycles_estimate:float = 60,
                    ):
    '''
    Parameters
    ----------
    image: napari.layers.Image
        Select the image to use for caontrast measurement.
    roi: napari.layers.Shapes
        Use a rectangolar roi to calculate the contrast.
    num_phases: number of phases used
    show_figures:flag used to determine whether to show
    all figures or just the resulting dephasing
    '''    
    
    
    ymin = int(np.min(roi.data[0][...,1]))
    ymax = int(np.max(roi.data[0][...,1]))
    xmin = int(np.min(roi.data[0][...,2]))
    xmax = int(np.max(roi.data[0][...,2]))

    data = image.data[...,ymin:ymax,xmin:xmax]
    sp,sy,sx = data.shape

    m = np.zeros((sy,sx),dtype=np.complex128) 
    for phase_idx in range(sp):
        phase_step = 2*np.pi/sp
        m += 2/sp * data[phase_idx,:,:]*np.exp(+1.j*phase_step*phase_idx)

    _recovered_phase = np.angle(m)
    amplitude = np.abs(m)
    sum_image = np.sum(data,axis=0)

    linear_phase = np.linspace(0, cycles_estimate*2*np.pi, sx)

    recovered_phase = _recovered_phase - linear_phase[None,:]
    recovered_phase = recovered_phase % (2*np.pi) - np.pi 

    viewer.add_image(sum_image)
    viewer.add_image(recovered_phase)
    viewer.add_image(amplitude)

    if try_unwrapping:
        unwrapped_phase = np.unwrap(recovered_phase,axis=1)
        naive_unwrapped_2D  = naive_unwrap2d(recovered_phase)
        viewer.add_image(unwrapped_phase)
        viewer.add_image(naive_unwrapped_2D)

@magic_factory(call_button="Estimate cycles number")
def count_cycles(viewer: Viewer,
                    image: Image, 
                    roi: Shapes,
                    min_distance:int = 10,
                    cycles_estimate:int = 0
                    ):
    '''
    Parameters
    ----------
    image: napari.layers.Image
        Select the image to use for caontrast measurement.
    roi: napari.layers.Shapes
        Use a rectangolar roi to calculate the contrast.
    num_phases: number of phases used
    show_figures:flag used to determine whether to show
    all figures or just the resulting dephasing
    '''    
    
    ymin = int(np.min(roi.data[0][...,1]))
    ymax = int(np.max(roi.data[0][...,1]))
    xmin = int(np.min(roi.data[0][...,2]))
    xmax = int(np.max(roi.data[0][...,2]))

    data = image.data[...,ymin:ymax,xmin:xmax]
    sp,sy,sx = data.shape

    line = data[0,0,:]
    peaks, _ = sc.find_peaks(line, prominence=1,distance=min_distance)
    count_cycles.cycles_estimate.value =  len(peaks) 
    print('cycles in the first line:', len(peaks) )





@magic_factory(call_button="Select phases")
def select_phases(viewer: Viewer,
                    image: Image, 
                    ph0:int = 0,
                    ph1:int = 49,
                    ph2:int = 68,
                    ph3:int = 83,
                    ph4:int = 97,
                    ph5:int = 112,
                    ):
    
    stack= image.data[[ph0,ph1,ph2,ph3,ph4,ph5],:,:]
    viewer.add_image(stack)    
    

@magic_factory(call_button = "Preprocessing")
def preprocesser(viewer:Viewer,
                 image: Image,
                 frame_factor: float = 0.25
                 ):
    preprocessed = np.empty_like(image.data)
    for i in range(np.size(image.data,0)):
        preprocessed[i] = _image_preprocesser(image.data[i],frame_factor)

    viewer.add_image(preprocessed)



class H5opener(QWidget):

    full_path = os.path.realpath(__file__)
    _folder, _ = os.path.split(full_path) 
    for level in range(3):
        _folder = os.path.join(_folder, os.pardir)
    
    def __init__(self, napari_viewer):
        self.viewer = napari_viewer
        super().__init__()
        
    def open_h5_dataset(self, path: pathlib.Path = _folder,
                        dataset:int = 0, 
                        ):
        # open file
        directory, filename = os.path.split(path)
        stack,found = get_h5_dataset(path, dataset)
        
        #updates settings
        measurement_names,_ = get_group_name(path, 'measurement')
        measurement_name = measurement_names[0]
        for key in ['magnification','n','NA','pixelsize','wavelength']:
            val = get_h5_attr(path, key, group = measurement_name)
            if len(val)>0 and hasattr(self,key):
                new_value = val[0]
                setattr(getattr(self,key), 'val', new_value)
                print(f'Updated {key} to: {new_value} ')
        fullname = f'dts{dataset}_{filename}'
        self.show_image(stack, im_name=fullname)
                 
    def show_image(self, image_values, im_name, **kwargs):
        '''
        creates a new Image layer with image_values as data
        or updates an existing layer, if 'hold' in kwargs is True 
        '''
        if 'scale' in kwargs.keys():    
            scale = kwargs['scale']
        else:
            scale = [1.]*image_values.ndim
        if 'colormap' in kwargs.keys():
            colormap = kwargs['colormap']
        else:
            colormap = 'gray'    
        if kwargs.get('hold') is True and im_name in self.viewer.layers:
            layer = self.viewer.layers[im_name]
            layer.data = image_values
            layer.scale = scale
        else:  
            layer = self.viewer.add_image(image_values,
                                            name = im_name,
                                            scale = scale,
                                            colormap = colormap)
        self.center_stack(image_values)
        if kwargs.get('autoscale') is True:
            layer.reset_contrast_limits()
        return layer

    def center_stack(self, image_layer):
        '''
        centers a >3D stack in z,y,x 
        '''
        data = image_layer.data
        if data.ndim >2:
            current_step = list(self.viewer.dims.current_step)
            for dim_idx in [-3,-2,-1]:
                current_step[dim_idx] = data.shape[dim_idx]//2
            self.viewer.dims.current_step = current_step    

if __name__ == '__main__':
    
    import napari
    viewer = napari.Viewer()
    
    peak_counter_widget = count_cycles()

    phase_widget = phase_profilometry()

    h5widget = H5opener(viewer) 

    h5_opener = magicgui(h5widget.open_h5_dataset, call_button='Open h5 dataset')


    viewer.window.add_dock_widget(h5_opener,
                                  name = 'H5 file selection',
                                  add_vertical_stretch = True)
    
    viewer.window.add_dock_widget(peak_counter_widget, name = 'Peak counter',
                                  area='right', add_vertical_stretch=True)
    
    viewer.window.add_dock_widget(phase_widget, name = 'Phase estimation',
                                  area='right', add_vertical_stretch=True)


    napari.run() 