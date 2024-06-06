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
import scipy.signal as sc
from enum import Enum
from get_h5_data import get_h5_dataset, get_h5_attr, get_datasets_index_by_name, get_group_name
import os
from qtpy.QtWidgets import  QWidget
import pathlib
from scipy import fftpack as fft

 
@magic_factory(call_button="Contrast measurement")
def contrast_measurement(viewer: Viewer, image: Image, roi: Shapes,
                         pixel_size:float=6.5,
                         Magnification:float=20,
                         frame_index:int=0,
                         distance_between_peaks:int=4,
                         background:float = 321.0,
                         ):
    '''
    Parameters
    ----------
    image: napari.layers.Image
        Select the image to use for caontrast measurement.
    roi: napari.layers.Shapes
        Use a rectangolar roi to calculate the contrast.
    pixel_size: size of the pixel at the detector
    Magnification: M of the microscope
    frame_index: frame to be used for the contrast measurement
    distance_between_pixels: minimum distance, in pixels, between peaks to be recovered
    background: estimate of the detector dark noise
    '''
    
    ymin = int(np.min(roi.data[0][...,1]))
    ymax = int(np.max(roi.data[0][...,1]))
    xmin = int(np.min(roi.data[0][...,2]))
    xmax = int(np.max(roi.data[0][...,2]))

    sim_data = image.data[...,ymin:ymax,xmin:xmax]
    sz,sy,sx = sim_data.shape
    slices = np.mean(sim_data, axis=1)

    x = np.linspace(xmin,xmax,sx) * pixel_size/Magnification
    y = slices[frame_index]-background

    peaks, _ = sc.find_peaks(y, prominence=1,distance=distance_between_peaks)
    valley, _  = sc.find_peaks(-y, prominence=1,distance=distance_between_peaks)

    xp = x[peaks]
    yp = y[peaks]
    xv = x[valley]
    yv = y[valley]

    Imax = np.mean(yp)
    Imin = np.mean(yv)

    C = (Imax-Imin)/(Imax+Imin)
    print('Contrast:', C) 

    char_size = 12
    linewidth = 0.7
    #plt.rc('font', family='serif', size=char_size)
    xsize=3
    ysize=0.75*xsize
    fig = plt.figure(figsize=(xsize, ysize), dpi=300)
    ax = fig.add_subplot(111)
    ax.plot(x, y, 
            linewidth=linewidth,
            linestyle='solid',
            color='gray')
    ax.plot(x, y, 
            marker='o',
            markersize=2,
            linewidth=0,
            linestyle='solid',
            color='black')
    ax.plot(xp, yp, 
            marker='o',
            markersize=3,
            linewidth=0,
            linestyle='solid',
            color='red')
    ax.plot(xv, yv, 
            marker='o',
            markersize=3,
            linewidth=0,
            linestyle='solid',
            color='blue')
    ax.plot(x, Imax*np.ones_like(x), 
            linewidth=1.5,
            linestyle='dotted',
            color='red')
    ax.plot(x, Imin*np.ones_like(x), 
            linewidth=1.5,
            linestyle='dotted',
            color='blue')
    xlabel = 'Position [\u03BCm]'
    ylabel = 'Intensity [a.u.]'
    #ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.set_title(title, size=char_size*0.8)   
    ax.set_xlabel(xlabel, size=char_size*0.8)
    ax.set_ylabel(ylabel, size=char_size*0.8)
    ax.xaxis.set_tick_params(labelsize=char_size*0.6)
    ax.yaxis.set_tick_params(labelsize=char_size*0.6)
    plt.ylim(0.0,np.amax(y)*1.1)
    plt.xlim(np.amin(x), np.amax(x))
    fig.tight_layout()
    plt.show()
    

class Phase_estimation_modes(Enum):
    PEAKS = 0
    FOURIER = 1


def find_max(img):
    return divmod(np.argmax(img,),np.shape(img)[0])

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

@magic_factory(call_button="Phase estimation")
def phase_estimation(viewer: Viewer,
                    image: Image, roi: Shapes,
                    phase_estimation_modes: Phase_estimation_modes,
                    max_voltage:float = 12.0,
                    num_phases:int= 6,
                    mask_size:float= 0.2,
                    ):
    '''
    Parameters
    ----------
    image: napari.layers.Image
        Select the image to use for caontrast measurement.
    roi: napari.layers.Shapes
        Use a rectangolar roi to calculate the contrast.
    filter_mode: filtering method for background removal
    max_voltage: maximum voltage in the acquired scan
    num_phases: number of phases to be recovered
    '''    
    
    max_pos = []
    max_num = -1
    dephasing = [0]

    ymin = int(np.min(roi.data[0][...,1]))
    ymax = int(np.max(roi.data[0][...,1]))
    xmin = int(np.min(roi.data[0][...,2]))
    xmax = int(np.max(roi.data[0][...,2]))

    sim_data = image.data[...,ymin:ymax,xmin:xmax]
    sz,sy,sx = sim_data.shape

    volts = np.linspace(0,max_voltage,sz)
    slices = np.mean(sim_data, axis=1)

    plt.figure()
    plt.plot(slices[0,:])


    if phase_estimation_modes == Phase_estimation_modes.PEAKS:

        for i in range(sz):
            
            this_slice = slices[i,:]
            
            baseline = sc.savgol_filter(this_slice,sx//10,3)
            this_slice = baseline-this_slice

            if i==1:
                plt.figure()
                plt.plot(this_slice)

            max_pos.append(list(sc.argrelmax(this_slice,order=2)[0]))
            if max_num == -1 or max_num > len(max_pos[i]):
                max_num = len(max_pos[i])
            period = np.average(np.diff(max_pos[i]))
            rad_per_pixel = 2*np.pi/period

        #compute dephasing
        for i in range(sz-1):
            dephasing.append(np.average(np.array(max_pos[i+1][0:max_num-1])-np.array(max_pos[0][0:max_num-1]))*rad_per_pixel)

        dephasing = np.unwrap(dephasing)

    elif phase_estimation_modes == Phase_estimation_modes.FOURIER:
        print('Fourier')

        phases = []
        
        for im_idx, im in enumerate(sim_data):
            ft = fft.fftshift(fft.fft2(im))
            
            # find the coordinates of the peak in Fourier space
            if im_idx == 0:
                mask_radius = sy*mask_size
                x = np.array(np.linspace(-sx/2,sx/2,sx))
                y = np.array(np.linspace(-sy/2,sy/2,sy))
                XX,YY = np.meshgrid(x,y)
                ft_filtered = ft * (XX**2+YY**2>mask_radius**2)
                peak = find_max(np.abs(ft_filtered*(XX-YY>0)))

            #phase = np.arctan(np.imag(ft[peak])/np.real(ft[peak]))
            phase = np.angle(ft[peak])   
            phases.append(phase)
        dephasing = np.unwrap(np.array(phases))
        dephasing = dephasing - dephasing[0]
    
    y0 = np.arange(0,2*np.pi,2*np.pi/num_phases)

    voltages_idx_to_use =[]

    for y in y0:
        voltage_idx_to_use = find_nearest(dephasing,y)
        voltages_idx_to_use.append(voltage_idx_to_use)

    print('Voltages to apply:', *np.round(volts[voltages_idx_to_use],1))
    print('Phase (deg)      :', *np.round(dephasing[voltages_idx_to_use] * 180/np.pi,1))
    print('Index            :', *voltages_idx_to_use)


    plt.figure()
    plt.plot(volts,dephasing)
    for y,idx in zip(y0,voltages_idx_to_use):
        plt.hlines(y, min(volts),max(volts),'r',linestyles  ='dotted')
        plt.vlines(volts[idx], min(dephasing),max(dephasing),'g',linestyles  ='dotted')
    plt.xlabel('Applied voltage [V]')
    plt.ylabel('Dephasing [rad]')
    plt.title('Voltage-dephasing characteristic')
    plt.show()    


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