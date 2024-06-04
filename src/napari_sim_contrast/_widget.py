"""
Created on Tue Jun 6 15:58:58 2024

@author: Paolo Maran and Andrea Bassi @Polimi

Napari widget for SIM images contrast estimation and phase calibration

"""

import numpy as np
from magicgui import magic_factory
from napari import Viewer
from napari.layers import Image, Shapes
import matplotlib.pyplot as plt
import scipy.signal as sc
from enum import Enum

 
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
    

class Filter_modes(Enum):
    SAVGOL = 0
    BUTTERWORTH = 1

@magic_factory(call_button="Phase estimation")
def phase_estimation(viewer: Viewer,
                    image: Image, roi: Shapes,
                    filter_mode: Filter_modes,
                    max_voltage:float = 12.0,
                    num_phases:int=6):
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

    for i in range(sz):
        
        this_slice = slices[i,:]
        
        if filter_mode == Filter_modes.SAVGOL:
            baseline = sc.savgol_filter(this_slice,40,3)
            this_slice = baseline-this_slice
        
        if filter_mode == Filter_modes.BUTTERWORTH:
            F = sc.butter(4,25,'high',analog=False,output='sos',fs=200)
            this_slice = sc.sosfilt(F,this_slice)

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

    y0 = np.arange(0,2*np.pi,2*np.pi/num_phases)
    
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return idx

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

    

if __name__ == '__main__':
    
    import napari
    viewer = napari.Viewer()
    contrast_widget = contrast_measurement()
    phase_widget = phase_estimation()
    
    viewer.window.add_dock_widget(contrast_widget, name = 'Contrast calculation',
                                  area='right', add_vertical_stretch=True)
    viewer.window.add_dock_widget(phase_widget, name = 'Phase estimation',
                                  area='right', add_vertical_stretch=True)

    napari.run() 