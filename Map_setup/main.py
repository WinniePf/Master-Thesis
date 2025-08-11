import sys
sys.path.append(r'C:\Users\localadmin\Desktop\Data\Winnie\setup')
import os
import time
import platform
import numpy as np
from datetime import datetime
#from PyQt5.QtWidgets import (QApplication, QMainWindow, QAction,
#                             QFileDialog, QVBoxLayout, QWidget, QSlider, QLabel, QDialog, QLineEdit, QCheckBox, QDialogButtonBox, QMessageBox)
#from PyQt5.QtCore import Qt
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import pyqtgraph as pg
from avaspec import *
import globals
from pipython import GCSDevice, pitools
import matplotlib.pyplot as plt



class MainWindow(QMainWindow):
    timer = QTimer()
    newdata = pyqtSignal(int, int)


    def __init__(self):
        super().__init__()
        

        self.init = False
        self.x_span = 4096
        self.y_span = 4096
        self.meas_time_R = 12
        self.meas_time_T = 50

        self.nr_averages_R = 5
        self.nr_averages_T = 5
        
        # if Reflection (1) or Transmission (0) or Absorption (-1)
        self.Refl = 1
        # set if absorption map is calculated
        self.firstplot = True
        basepath = 'C:/Users/localadmin/Desktop/Data/Winnie/setup/measurements/'            
        self.folder_path = basepath+'Map_'+datetime.now().strftime("%H-%M-%S-%d-%b-%Y")


        # stage params:        
        self.xmin = 0     # minimum x coordinate
        self.ymin = 0     # minimum y coordinate
        self.xmax = 5   # maximum x coordinate
        self.ymax = 5   # maximum y coordinate
        self.nx = 5      # steps in x
        self.ny = 5      # steps in y
        
        
        global x_span
        global y_span
        global xmin   # minimum x coordinate
        global ymin   # minimum y coordinate
        global xmax   # maximum x coordinate
        global ymax   # maximum y coordinate
        global nx     # steps in x
        global ny     # steps in y

        global meas_time

        x_span = self.x_span
        y_span = self.y_span
        
        self.pidevice = None
        self.x = np.zeros(globals.pixels)
        self.y = np.zeros(globals.pixels)
        self.mapdata_R = np.zeros((self.nx, self.ny, 2048))
        self.mapdata_T = np.zeros((self.nx, self.ny, 2048))
        self.mapdata_A = np.zeros((self.nx, self.ny, 2048))

        meas_time_R = self.meas_time_R
        meas_time_T = self.meas_time_T

        self.baseline_R = []
        self.baseline_T = []

        self.darkcount_R = []
        self.darkcount_T = []

        
        self.initUI()
        self.newdata.connect(self.handle_newdata)


    def measure_cb(self, pparam1, pparam2):
        param1 = pparam1[0] # dereference the pointers
        param2 = pparam2[0]
        self.newdata.emit(param1, param2) 

    #@pyqtSlot()
    def initUI(self):
        # Set up the main window
        self.setWindowTitle("Measurement Diagram")
        #self.setGeometry(100, 100, 1200, 800)

        # Central widget
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)

        # Layout
        self.layout = QHBoxLayout()
        self.layout2 = QVBoxLayout()

        self.centralWidget.setLayout(self.layout)

        self.plotWidget = pg.PlotWidget()
        self.mapWidget = pg.PlotWidget()
        
        self.modeLabel = QLabel('Mode: None')

        self.plotWidget.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)
        self.mapWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.layout2.addWidget(self.modeLabel)
        self.layout.addWidget(self.mapWidget, stretch=3)
        self.layout2.addWidget(self.plotWidget, stretch=1)

        self.x = np.linspace(0.8, 4.6, 2048)
        self.ticks_slider = np.arange(0, 2048)

        # Slider to change frequency
        self.ESlider = QSlider(Qt.Horizontal)
        self.ESlider.setRange(0, 2047)
        
        # set initial display
        E = 1
        E_index = np.argmin(np.abs(self.x-E))

        self.ESlider.setValue(E_index)

        # change plot when sliding energy
        self.ESlider.valueChanged.connect(self.update_plot)
        
        self.layout2.addWidget(self.ESlider)
        
        self.layout.addLayout( self.layout2, stretch=1 )

        # Energy label

        self.ELabel = QLabel(f"Energy: {E} eV")
        self.layout2.addWidget(self.ELabel)

        # Initialize spectrum
        self.y = np.zeros(2048)
        self.plotData = self.plotWidget.plot(self.x, self.y)
        self.plotWidget.setLabel('left', 'counts')
        self.plotWidget.setLabel('bottom', 'energy (eV)')
        
        
        # initialize map
        self.img = pg.ImageItem(image = self.mapdata_R[:,:,0].copy())
        self.mapWidget.addItem(self.img)
        self.colorbar = self.mapWidget.addColorBar(self.img, colorMap='viridis', limits=(0.00,100.00))
        self.mapWidget.setLabel('left', 'y (um)')
        self.mapWidget.setLabel('bottom', 'x (um)')
        
        # set axis dimensions, not pixels
        yaxis = self.mapWidget.getAxis('left')
        yaxis.setScale((self.ymax - self.ymin) / self.ny)
        xaxis = self.mapWidget.getAxis('bottom')
        xaxis.setScale((self.xmax - self.xmin) / self.nx)

        # Create menu bar
        self.createMenuBar()

        self.show()

    @pyqtSlot()
    def createMenuBar(self):
        # Create a menu bar
        menuBar = self.menuBar()

        # Create the File menu
        fileMenu = menuBar.addMenu('File')

        # Add "Load Data" action
        loadAction = QAction('Load Data', self)
        loadAction.triggered.connect(self.load_data)
        fileMenu.addAction(loadAction)


        # Add "load baseline" action
        saveAction = QAction('Load Baseline', self)
        saveAction.triggered.connect(self.load_baseline)
        fileMenu.addAction(saveAction)

        # Add "save baseline" action
        saveAction = QAction('Save Baseline', self)
        saveAction.triggered.connect(self.save_baseline)
        fileMenu.addAction(saveAction)

        # Add "save baseline" action
        saveAction = QAction('Save Image', self)
        saveAction.triggered.connect(self.save_image)
        fileMenu.addAction(saveAction)

        # Create the File menu
        RMenu = menuBar.addMenu('Reflection Measurement')

        # Add "create dark count" action
        loadAction = QAction('create dark count', self)
        loadAction.triggered.connect(lambda: self.dark_count(True))
        RMenu.addAction(loadAction)
        
        # Add "create baseline" action
        loadAction = QAction('create Reflection reference', self)
        loadAction.triggered.connect(lambda: self.base_line(True))
        RMenu.addAction(loadAction)
        
        # Add "Load Data" action
        loadAction = QAction('Create Reflection Map', self)
        loadAction.triggered.connect(lambda: self.run_measurement('Refl'))
        
        RMenu.addAction(loadAction)

        TMenu = menuBar.addMenu('Transmission Measurement')
        
        # Add "create dark count" action
        loadAction = QAction('create dark count', self)
        loadAction.triggered.connect(lambda: self.dark_count(False))
        TMenu.addAction(loadAction)


        loadAction = QAction('create Transmission reference', self)
        loadAction.triggered.connect(lambda: self.base_line(False))
        TMenu.addAction(loadAction)
        
        loadAction = QAction('Create Transmission Map', self)
        loadAction.triggered.connect(lambda: self.run_measurement('Trans'))
        TMenu.addAction(loadAction)
        
        
        
        # Add "Settings" menu
        
        #settings_action = QAction(QIcon.fromTheme("preferences-system"), "Settings", self)
        #settings_action.setStatusTip("Open Settings")
        #settings_action.triggered.connect(self.settings)

        
        settings_icon_action = QAction(QIcon.fromTheme("system"), "Settings", self)
        settings_icon_action.setToolTip("Settings")
        settings_icon_action.triggered.connect(self.settings)
        menuBar.addAction(settings_icon_action)

        plot_abs_map = QAction(QIcon.fromTheme("system"), "Plot Abs. Map", self)
        plot_abs_map.triggered.connect(self.create_abs_map)
        menuBar.addAction(plot_abs_map)


        #loadAction = QAction('measurement settings', self)
        #loadAction.triggered.connect(self.settings)
        #RMenu.addAction(loadAction)
    def update_plot(self, index=None):
        E = self.x[self.ESlider.value()]
        E_index = np.argmin(np.abs(self.x-E))
        
        if self.init:
            
            # plot current spectrum
            self.x = 1239.8/np.array(globals.wavelength)[0:2048][::-1]
            #self.y = np.array(globals.spectraldata)[0:2048][::-1]

        self.plotData.setData(self.x, self.y)

        if self.init or self.mapdata_R[0][0][0] != 0 or self.mapdata_T[0][0][0] != 0 or self.mapdata_A[0][0][0] != 0:
            # plot map
            self.ELabel.setText(f"Energy: {round(self.x[E_index],3)} eV")
            if self.Refl == 1:
                plot_slice = self.mapdata_R[:,:,E_index]
                renorm_index = 100
            elif self.Refl == 0:
                plot_slice = self.mapdata_T[:,:,E_index]
                renorm_index = 100
            elif self.Refl == -1:
                plot_slice = self.mapdata_A[:,:,E_index]

                renorm_index = 100


            # plot R/F from 0 to 100 % if baseline was created
        
            # set the plot range for the first map plot:
            if index == 0:
                self.img.setImage(plot_slice*renorm_index, levels=(np.min(plot_slice)*renorm_index, np.max(plot_slice)*renorm_index))
                self.colorbar.setLevels((np.min(plot_slice)*renorm_index, np.max(plot_slice)*renorm_index))
            else:
                levels = self.colorbar.levels()
                self.img.setImage(plot_slice*renorm_index, levels=levels)
                self.colorbar.setLevels(levels)
        

    
    '''
    @pyqtSlot()
    def save_data(self):
        # Save the current spectrum as a .txt file
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "Save Spectrum as .txt",
                                                  "",
                                                  "Text Files (*.txt)",
                                                  options=options)
        if fileName:
            np.savetxt(fileName, np.column_stack((self.x, self.y)),
                       header="X Values\tY Values", delimiter="\t")
            print(f"Data saved to {fileName}")
    '''
            
    def save_image(self):
        if self.mapdata_T[0][0][0] != 0 or self.mapdata_R[0][0][0] != 0 or self.mapdata_A[0][0][0] != 0: 
            # create colormap at current energy:
            E = self.x[self.ESlider.value()]
            E_index = np.argmin(np.abs(self.x-E))

        
            plt.figure()
            if self.Refl == 1:
                plt.imshow(self.mapdata_R[:,:,E_index], cmap='viridis', aspect='equal', extent=(self.xmin, self.xmax, self.ymin, self.ymax))
            elif self.Refl == 0:
                plt.imshow(self.mapdata_T[:,:,E_index], cmap='viridis', aspect='equal', extent=(self.xmin, self.xmax, self.ymin, self.ymax))
            elif self.Refl == -1:
                plt.imshow(self.mapdata_A[:,:,E_index], cmap='viridis', aspect='equal', extent=(self.xmin, self.xmax, self.ymin, self.ymax))
            plt.xlabel('x (\u03BCm)')
            plt.ylabel('y (\u03BCm)')
            cbar = plt.colorbar()
            if self.Refl == 0:
                prefix = 'Transmission'  
            elif self.Refl == 1:
                prefix = 'Reflection'
            else:
                prefix = 'Absorption'  
        
            cbar.ax.set_ylabel(prefix)

            plt.savefig(self.folder_path + '/' + f'{prefix}_map_at_{round(E,2)}_eV.pdf')
        else:
            print('No Data to Map')

    @pyqtSlot()
    def load_data(self):
        # Load data from a .txt file
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Load Spectrum Data",
                                                  "",
                                                  "Data Files (*.txt *.npy)",
                                                  options=options)
        # handle imports of map files in binary .npy format
        if fileName[-4:] == '.npy':
            if fileName[-5] == 'R':
                try:
                    self.mapdata_R = np.load(fileName)
                    self.folder_path = fileName[:fileName.rfind('/')]
                    self.x = np.loadtxt(self.folder_path + '/R_map_part_0_0.txt', dtype='float', delimiter=',')[:,0]
                    self.Refl = 1
                    self.update_plot()
                except Exception:
                    QMessageBox.critical(self, "", "Error importing data")
                    
            if fileName[-5] == 'T':
                try:
                    self.mapdata_T = np.load(fileName)
                    self.folder_path = fileName[:fileName.rfind('/')]
                    self.x = np.loadtxt(self.folder_path + '/T_map_part_0_0.txt', dtype='float', delimiter=',')[:,0]
                    self.Refl = 0
                    self.update_plot()
                except Exception:
                    QMessageBox.critical(self, "", "Error importing data")
                    
            if fileName[-5] == 'A':
                try:
                    self.mapdata_A = np.load(fileName)
                    self.folder_path = fileName[:fileName.rfind('/')]
                    self.x = np.loadtxt(self.folder_path + '/R_map_part_0_0.txt', dtype='float', delimiter=',')[:,0]
                    self.Refl = -1
                    self.update_plot()
                except Exception:
                    QMessageBox.critical(self, "", "Error importing data")
        # handle imports of single spectra in .txt format
        elif fileName[-4:] == '.txt':
            try:
                data = np.loadtxt(fileName, delimiter="\t", skiprows=0)
                self.x = data[:, 0]
                self.y = data[:, 1]
                self.update_plot()
            except Exception:
                QMessageBox.critical(self, "", "Error importing data")

                

    @pyqtSlot()
    def settings(self):
        # open and handle input from settings dialogue
        settings_dialog = SettingsDialog(self)
        # Open the dialogue and check if the user accepted it
        if settings_dialog.exec_() == QDialog.Accepted:
            # Retrieve the parameters and update them if input was correct
            if settings_dialog.get_data():
                xmin, xmax, ymin, ymax, nx, ny, t_R, t_T, n_avg_R, n_avg_T = settings_dialog.get_data()
                if xmin:
                    self.xmin = xmin
                if ymin:
                    self.ymin = ymin
                if xmax:
                    self.xmax = xmax
                if ymax:
                    self.ymax = ymax
                if t_R:
                    self.meas_time_R = t_R
                if t_T:
                    self.meas_time_T = t_T
                if nx:
                    self.nx = nx
                if ny:
                    self.ny = ny
                if n_avg_R:
                    self.nr_averages_R = n_avg_R
                if n_avg_T:
                    self.nr_averages_T = n_avg_T

    def initialize_stage(self):
        # initialize stage before doing a measurement
        self.pidevice = GCSDevice('E-727')
        # could prob remove the dlg.
        self.pidevice.InterfaceSetupDlg()
        pitools.startup(self.pidevice)
        
        # check if map extends beyond axis range:
        xminrange = pitools.getmintravelrange(self.pidevice, '1')['1']
        yminrange = pitools.getmintravelrange(self.pidevice, '2')['2']
        xmaxrange = pitools.getmaxtravelrange(self.pidevice, '1')['1']
        ymaxrange = pitools.getmaxtravelrange(self.pidevice, '2')['2']
                
        
        if xminrange < self.xmin:
            self.xmin = xminrange
            print(f'Invalid x min, setting x min to {xminrange} um')
        if yminrange < self.ymin:
            self.ymin = yminrange
            print(f'Invalid y min, setting y min to {yminrange} um')
        if xmaxrange < self.xmax:
            self.xmax = xmaxrange
            print(f'Invalid x max, setting x max to {xmaxrange} um')
        if ymaxrange < self.ymax:
            self.ymax = ymaxrange
            print(f'Invalid y max, setting y max to {ymaxrange} um')

            
        
    @pyqtSlot()
    def move_stage(self, index):
        # move the stage to a location x, y while keeping interface responsive
        #pitools.moveandwait(self.pidevice, ('1','2'), (self.path[index][0], self.path[index][1]))
        self.pidevice.MOV(('1','2'), (self.path[index][0], self.path[index][1]))
        while True:
            ontarget = self.pidevice.qONT(['1', '2'])
            
            if ontarget['1'] and ontarget['2']:
                break
            else:
                time.sleep(0.01)
                app.processEvents()
        
    def save_spectrum(self, index):
        xpos = self.path[index][0]
        ypos = self.path[index][1]

        if self.Refl == 1:
            np.savetxt(self.folder_path + r'/' + f'R_map_part_{index // self.nx}_{index % self.nx}.txt', np.column_stack((self.x, self.y[::-1])), delimiter=',')
        elif self.Refl == 0:
            np.savetxt(self.folder_path + r'/' + f'T_map_part_{index // self.nx}_{index % self.nx}.txt', np.column_stack((self.x, self.y)), delimiter=',')


    def save_map(self):
        if self.Refl == 1:
            np.save(self.folder_path + r'/' + f'completed_map_R.npy', self.mapdata_R)
        elif self.Refl == 0:
            np.save(self.folder_path + r'/' + f'completed_map_T.npy', self.mapdata_T)
        elif self.Refl == -1:
            np.save(self.folder_path + r'/' + f'completed_map_A.npy', self.mapdata_A)

        
    def load_map(self, path):
        self.mapdata = np.load(path)

    @pyqtSlot()
    def run_measurement(self, mode=None):
        if mode == 'Refl':
            self.Refl = True
        if mode == 'Trans':
            self.Refl = False
        
        if self.Refl == 1:
            self.modeLabel.setText('Mode: Reflection')
        elif self.Refl == 0:
            self.modeLabel.setText('Mode: Transmission')

        self.initialize_spectrometer()
        self.initialize_stage()
        
        try:
            os.mkdir(self.folder_path)
        except FileExistsError:
            pass
        
        with open(self.folder_path + '/' + 'info.txt', 'w') as f:
            f.write('Map Settings \n')
            f.write('Spectrometer settings \n')
            f.write('length of energy array (0.87-4.48 eV):' + str(self.x_span) + '\n')
            

        xpath = np.linspace(self.xmin, self.xmax, self.nx)

        # set path you want to go. Array with x,y,n coordinates for n spectra
        ypath = np.linspace(self.ymin, self.ymax, self.ny)


        self.path = np.zeros((self.nx,self.ny, 2))

        # could be vectorized:
        for ix in range(self.nx):
            for iy in range(self.ny):
                self.path[ix][iy] = [xpath[ix], ypath[iy]]
        
        self.path = self.path.reshape(self.nx*self.ny, 2)
        
        # update axes for map:
        
        if self.Refl == 1:
            self.mapdata_R = np.zeros((self.nx, self.ny, 2048))
            print('Starting reflection map:')

        elif self.Refl == 0:
            self.mapdata_T = np.zeros((self.nx, self.ny, 2048))
            print('Starting transmission map:')

        
        yaxis = self.mapWidget.getAxis('left')
        yaxis.setScale((self.ymax - self.ymin) / self.ny)
        xaxis = self.mapWidget.getAxis('bottom')
        xaxis.setScale((self.xmax - self.xmin) / self.nx)

        
        print('Pos 1:', self.path[0])

        # main loop for the map: Loops through each coordinate and measures spectrum at that position        
        for index in range(len(self.path)):
            
            
            print(f'Moving to location x = {round(self.path[index][0], 2)}, y = {round(self.path[index][1], 2)}')
            self.move_stage(index)
            
            self.measure_spectrum()
            
            if self.Refl == 1:
                self.mapdata_R[index // self.nx, index % self.nx, :] = self.y
            elif self.Refl == 0:
                self.mapdata_T[index // self.nx, index % self.nx, :] = self.y
            
            self.update_plot(index=index)
            
            self.save_spectrum(index)
            
        self.close()
        self.init = False
        
        self.save_map()
        
        
        
    def close(self):
        self.pidevice.CloseConnection()
        AVS_Done()
        
            
    def create_abs_map(self):
        try:
            self.Refl = -1
            self.mapdata_A = 1 - self.mapdata_R - self.mapdata_T
            
            self.save_map()
            self.update_plot(index=0)
            
            
        except:
            print('Absorption map could not be created')


    @pyqtSlot()
    def dark_count(self, isRefl):
        self.initialize_spectrometer()
        QMessageBox.critical(self, "", "creating dark count spectrum")
        if isRefl:
            self.Refl = 1
            self.darkcount_R = self.measure_spectrum(opt=1)
        else:
            self.Refl = 0
            self.darkcount_T = self.measure_spectrum(opt=1)

        self.update_plot()

    @pyqtSlot()
    def base_line(self, refl=None):
        self.initialize_spectrometer()
        QMessageBox.critical(self, "", "creating baseline spectrum")
        
        if refl:
            self.baseline_R = self.measure_spectrum(opt=2)
            try:
                if len(self.baseline_R.nonzero()) != len(self.baseline_R):
                    print('Baseline has entries with 0 counts! Adding 1 count to entries with 0 counts')   
                    zero_pos = np.where(self.baseline_R == 0)
                    self.baseline_R[zero_pos] += 1
            except:
                pass
        else:
            self.baseline_T = self.measure_spectrum(opt=2)
            try:
                if len(self.baseline_T.nonzero()) != len(self.baseline_T):
                    print('Baseline has entries with 0 counts! Adding 1 count to entries with 0 counts') 
                    zero_pos = np.where(self.baseline_T == 0)
                    self.baseline_T[zero_pos] += 1
		 
            except:
                pass
        self.update_plot()
        
      

    def load_baseline(self):
        try:
            self.baseline_T = np.loadtxt(self.folder_path + r'/' + f'baseline_T.txt', np.column_stack((self.x[::-1], self.y[::-1])), delimiter=',')
        except Exception:
            print('Transmission baseline not found at ', self.folder_path)
        
        try:
            self.baseline_R = np.loadtxt(self.folder_path + r'/' + f'baseline_R.txt', np.column_stack((self.x, self.y)), delimiter=',')
        except Exception:
            print('Reflection baseline not found at ', self.folder_path)


    def save_baseline(self):
        if self.baseline_R:
            try:
                np.savetxt(self.folder_path + r'/' + f'baseline_R.txt', np.column_stack((self.x, self.y)), delimiter=',')
            except Exception:
                print('Baseline not saved, error occured')
                
        if self.baseline_T:
            try:
                np.savetxt(self.folder_path + r'/' + f'baseline_T.txt', np.column_stack((self.x, self.y)), delimiter=',')
            except Exception:
                print('Baseline not saved, error occured')


    @pyqtSlot()
    def measure_spectrum(self, opt=None):
        # opt is identifier for type of measurement:
        # None for normal spectrum with background and baseline correction, 
        # 1    for darkcount
        # 2    for baseline
        if self.init:
            # If initialized, perform measurement
            ret = AVS_UseHighResAdc(globals.dev_handle, True)
            measconfig = MeasConfigType()

            # predefined vars:
            if self.Refl == 1:
                measconfig.m_IntegrationTime = self.meas_time_R
                measconfig.m_NrAverages = self.nr_averages_R

            elif self.Refl == 0:
                measconfig.m_IntegrationTime = self.meas_time_T
                measconfig.m_NrAverages = self.nr_averages_T


            nummeas = 1 
            sleeptime = 0.01 # change responsiveness of application

            # unchanged vars:
            measconfig.m_StartPixel = 0
            measconfig.m_StopPixel = globals.pixels - 1
            measconfig.m_IntegrationDelay = 0
            measconfig.m_CorDynDark_m_Enable = 0  
            measconfig.m_CorDynDark_m_ForgetPercentage = 0
            measconfig.m_Smoothing_m_SmoothPix = 0
            measconfig.m_Smoothing_m_SmoothModel = 0
            measconfig.m_SaturationDetection = 0
            measconfig.m_Trigger_m_Mode = 0
            measconfig.m_Trigger_m_Source = 0
            measconfig.m_Trigger_m_SourceType = 0
            measconfig.m_Control_m_StrobeControl = 0
            measconfig.m_Control_m_LaserDelay = 0
            measconfig.m_Control_m_LaserWidth = 0
            measconfig.m_Control_m_LaserWaveLength = 0.0
            measconfig.m_Control_m_StoreToRam = 0

            globals.NrScanned = 0
            
            # do we have to prepare measconfig once or before each measurement?
            ret = AVS_PrepareMeasure(globals.dev_handle, measconfig)
            avs_cb = AVS_MeasureCallbackFunc(self.measure_cb)
            AVS_MeasureCallback(globals.dev_handle, avs_cb, nummeas)

            while nummeas > globals.NrScanned:  # wait until data has arrived
                time.sleep(sleeptime)
                qApp.processEvents()

            ret = AVS_StopMeasure(globals.dev_handle)
            
            self.y = np.array(globals.spectraldata)[0:2048][::-1]
        
        if self.Refl == 1:
            baseline = self.baseline_R
            darkcount = self.darkcount_R
        elif self.Refl == 0:
            baseline = self.baseline_T
            darkcount = self.darkcount_T

        if not opt:
            try:
                self.y = (self.y - darkcount) / baseline
            except ValueError:
                print('Could not find baseline/darkcount to subtract')
                try:
                    self.y = self.y / baseline
                except ValueError:
                    pass
                
        if opt == 1:
            return self.y
        if opt == 2:
            try:
                self.y -= darkcount # subtract darkcount from baseline measurement
            except ValueError:
                print('Could not find darkcount to subtract') # catch if no darkcount recorded, then just spectrum is returned
            return self.y

    @pyqtSlot(int, int)
    def handle_newdata(self, lparam1, lparam2):
        ret = AVS_GetScopeData(globals.dev_handle)
        globals.NrScanned += 1
        globals.spectraldata = ret[1]
        #QMessageBox.information(self,"Info","Received data")
        sat_array = AVS_GetSaturatedPixels(globals.dev_handle)
        if np.sum(sat_array) > 1:
            print(f"Warning! {np.sum(sat_array)} pixels are saturated")
        
        return

    @pyqtSlot()
    def initialize_spectrometer(self):
        if not self.init:
            ret = AVS_Init(0)
            # QMessageBox.information(self,"Info","AVS_Init returned:  {0:d}".format(ret))
            ret = AVS_GetNrOfDevices()
            # QMessageBox.information(self,"Info","AVS_GetNrOfDevices returned:  {0:d}".format(ret))
            if (ret > 0):
                mylist = AVS_GetList(1)
                serienummer = str(mylist[0].SerialNumber.decode("utf-8"))
                QMessageBox.information(self,"Info","Found Serialnumber: " + serienummer)
                globals.dev_handle = AVS_Activate(mylist[0])
                # QMessageBox.information(self,"Info","AVS_Activate returned:  {0:d}".format(globals.dev_handle))
                #devcon = DeviceConfigType()
                devcon = AVS_GetParameter(globals.dev_handle, 63484)
                globals.pixels = devcon.m_Detector_m_NrPixels
                globals.wavelength = AVS_GetLambda(globals.dev_handle)
                #self.StartMeasBtn.setEnabled(True)
                self.init = True
            else:
                QMessageBox.critical(self,"Error","No devices were found!")
                
        return self.init

# Create the settings dialog
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Settings")
        self.setGeometry(300, 300, 200, 150)

        # Create layout and add widgets
        layout = QVBoxLayout()
        #self.region_label = QLabel("Set measurement region")

        self.meas_time_label_T = QLabel("measurement time T (ms)")
        self.meas_time_input_T = QLineEdit(self)
        self.meas_time_input_T.setText(str(window.meas_time_T))
        
        self.meas_time_label_R = QLabel("measurement time R (ms)")
        self.meas_time_input_R = QLineEdit(self)
        self.meas_time_input_R.setText(str(window.meas_time_R))


        self.xmin_label = QLabel("x min")
        self.xmin_input = QLineEdit(self)
        self.xmin_input.setText(str(window.xmin))
        
        self.xmax_label = QLabel("x max")
        self.xmax_input = QLineEdit(self)
        self.xmax_input.setText(str(window.xmax))
        
        self.ymin_label = QLabel("y min")
        self.ymin_input = QLineEdit(self)
        self.ymin_input.setText(str(window.ymin))
        
        self.ymax_label = QLabel("y max")
        self.ymax_input = QLineEdit(self)
        self.ymax_input.setText(str(window.ymax))

        self.nx_label = QLabel("nx")
        self.nx_input = QLineEdit(self)
        self.nx_input.setText(str(window.nx))
        
        self.ny_label = QLabel("ny")
        self.ny_input = QLineEdit(self)
        self.ny_input.setText(str(window.ny))

        self.nr_avg_label_T = QLabel("nr avg T")
        self.nr_avg_input_T = QLineEdit(self)
        self.nr_avg_input_T.setText(str(window.nr_averages_T))
        
        self.nr_avg_label_R = QLabel("nr avg R")
        self.nr_avg_input_R = QLineEdit(self)
        self.nr_avg_input_R.setText(str(window.nr_averages_R))

        
        self.button = QPushButton('Create Folder', self)
        self.button.clicked.connect(self.select_folder)
        
        layout.addWidget(self.xmin_label)
        layout.addWidget(self.xmin_input)
        layout.addWidget(self.xmax_label)
        layout.addWidget(self.xmax_input)
        layout.addWidget(self.ymin_label)
        layout.addWidget(self.ymin_input)
        layout.addWidget(self.ymax_label)
        layout.addWidget(self.ymax_input)
        layout.addWidget(self.nx_label)
        layout.addWidget(self.nx_input)
        layout.addWidget(self.ny_label)
        layout.addWidget(self.ny_input)
        layout.addWidget(self.nr_avg_label_R)
        layout.addWidget(self.nr_avg_input_R)
        layout.addWidget(self.nr_avg_label_T)
        layout.addWidget(self.nr_avg_input_T)
        layout.addWidget(self.button)
        layout.addWidget(self.meas_time_label_R)
        layout.addWidget(self.meas_time_input_R)
        layout.addWidget(self.meas_time_label_T)
        layout.addWidget(self.meas_time_input_T)


        #layout.addWidget(self.show_notifications)

        # OK and Cancel buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(buttons)
        
        
        
        # Set the layout for the dialog
        self.setLayout(layout)
        
    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Create folder for data storage')
        
        if folder_path:
            window.folder_path = folder_path


    def get_data(self):
        # return data entered in the settings dialogue
        try:
            xmin = float(self.xmin_input.text())
            xmax = float(self.xmax_input.text())
            ymin = float(self.ymin_input.text())
            ymax = float(self.ymax_input.text())
            nx = int(self.ny_input.text())
            ny = int(self.nx_input.text())
            n_avg_R = int(self.nr_avg_input_R.text())
            n_avg_T = int(self.nr_avg_input_T.text())
            meas_time_R = float(self.meas_time_input_R.text())
            meas_time_T = float(self.meas_time_input_T.text())

        # catch wrong inputs
        except ValueError:
            error_message = QMessageBox(self)
            error_message.setIcon(QMessageBox.Critical)
            error_message.setWindowTitle("Error")
            error_message.setText("Couldn't convert to integer")
            error_message.setInformativeText("Please check your input and try again.")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()  # Show the message box
            
            #xspan = False
            #yspan = False
            xmin = False
            xmax = False
            ymin = False
            ymax = False
            nx = False
            ny = False
            n_avg_R = False
            n_avg_T = False
            meas_time_R = False
            meas_time_T = False

        
        # return parameters to main function for processing
        return [xmin, xmax, ymin, ymax, nx, ny, meas_time_R, meas_time_T, n_avg_R, n_avg_T]

# Run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())

        

