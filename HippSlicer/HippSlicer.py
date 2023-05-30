import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np
import re
from qt import QStandardItem
from pathlib import Path
import copy
from vtk.util import numpy_support

# Packages that might need to be installed
try:
    from bids import BIDSLayout
except:
    os.system('PythonSlicer -m pip install pybids')
    from bids import BIDSLayout
        
try:
    import nibabel as nb
except:
    os.system('PythonSlicer -m pip install nibabel')
    import nibabel as nb

try:
    import nrrd
except:
    os.system('PythonSlicer -m pip install pynrrd')
    import nrrd

try:
    import pandas as pd
except:
    os.system('PythonSlicer -m pip install pandas')
    import pandas as pd

try:
    import yaml
except:
    os.system('PythonSlicer -m pip install pyyaml')
    import yaml

#
# HippSlicer. Module to connect 3D Slicer with vCastSender application.
#

class HippSlicer(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "HippSlicer"
        self.parent.categories = ["Segmentation"]
        self._dir_chosen = "" # Saves path to vCast exe file
        self.parent.dependencies = []
        self.parent.contributors = ["Mauricio Cespedes Tenorio (Western University)"]
        self.parent.helpText = """
        This tool is made to connect vCastSender application with 3D Slicer.
        """
        self.parent.acknowledgementText = """
        This module was originally developed by Mauricio Cespedes Tenorio (Western University) as part
        of the extension Multiviews.
        """ # <a href="https://github.com/mnarizzano/SEEGA">Multiviews</a>
    


#
# HippSlicer Widget
#

class HippSlicerWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """
    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._updatingGUIFromParameterNode = False  
        self._bool_subj = False
        self._dir_selected = False
        self.atlas_labels = self.resourcePath('Data/desc-subfields_atlas-bigbrain_dseg.tsv')
        self.config = self.resourcePath('Config/config.yml')
        # Condition to set apply botton to true based on files selection
        self.files_selected = False
        self.checkboxes = [[],[]] # convert and visible
        self.labels_color = {}

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        self._loadUI()
        self.logic = HippSlicerLogic()

        # Connections
        self._setupConnections()
    
    def _loadUI(self):
        """
        Load widget from .ui file (created by Qt Designer).
        """
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath('UI/HippSlicer.ui'))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        # UI boot configuration of 'Apply' button and the input box. 
        self.ui.applyButton.toolTip = "Please select a path to Hippunfold results"
        self.ui.applyButton.enabled = False
        # TableWidget
        header = self.ui.tableFiles.horizontalHeader()
        header.setDefaultSectionSize(80)
        header.setSectionResizeMode(0, qt.QHeaderView.Stretch)
        header.setSectionResizeMode(1, qt.QHeaderView.Fixed)
        header.setSectionResizeMode(2, qt.QHeaderView.Fixed)

        # Subj dropdown
        self.ui.subj.addItems(['Select subject'])

        self.ui.configFileSelector.setCurrentPath(self.config)

        # Combo boxes
        # print(self.ui.VisibleAll.currentText)
        # self.ui.VisibleAll.setCurrentText('---')
        

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)


    def _setupConnections(self):
        # Connections
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
        # (in the selected parameter node).
        self.ui.HippUnfoldDirSelector.connect("directoryChanged(QString)", self.onDirectoryChange)
        self.ui.OutputDirSelector.connect("directoryChanged(QString)", self.onDirectoryChange)
        self.ui.subj.connect('currentIndexChanged(int)', self.onSubjChange)
        self.ui.configFileSelector.connect("currentPathChanged(QString)", self.onConfigChange)
        self.ui.VisibleAll.connect('checkedIndexesChanged()', self.onVisibleAllChange)
        self.ui.ConvertAll.connect('checkedIndexesChanged()', self.onConvertAllChange)
        # print('ca')
        # Buttons
        self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()
    
    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self.removeObservers()

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """
        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def onSubjChange(self):
        """
        This method is called whenever subject object is changed.
        The module GUI is updated to show the current state of the parameter node.
        """
        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return
        self.checkboxes = [[],[]]
        # Set state of button
        if self.ui.subj.currentIndex > 0:
            self._bool_subj = True
            if self._dir_selected:
                # Clear the table
                while (self.ui.tableFiles.rowCount > 0):
                    self.ui.tableFiles.removeRow(0)
                # Load the files
                for file,_ in self.files[self.ui.subj.currentText]:
                    rowPosition = self.ui.tableFiles.rowCount
                    self.ui.tableFiles.insertRow(rowPosition)
                    # Use file path without selected parent folder
                    filename = re.sub(str(self.ui.HippUnfoldDirSelector.directory), '.', file)
                    self.ui.tableFiles.setItem(rowPosition, 0, qt.QTableWidgetItem(filename))
                    for i in range(1,3):
                        # Construct checkbox and add to table
                        cell_widget = qt.QWidget()
                        chk_bx = qt.QCheckBox()
                        chk_bx.setCheckState(qt.Qt.Checked)
                        lay_out = qt.QHBoxLayout(cell_widget)
                        lay_out.addWidget(chk_bx)
                        lay_out.setAlignment(qt.Qt.AlignCenter)
                        lay_out.setContentsMargins(0,0,0,0)
                        cell_widget.setLayout(lay_out)
                        self.ui.tableFiles.setCellWidget(rowPosition, i, cell_widget)
                        # Add checkbox object to list
                        self.checkboxes[i-1].append(chk_bx)
                # Update state of All/None checkboxes
                comboChkBoxes = [self.ui.ConvertAll, self.ui.VisibleAll]
                for comboBox in comboChkBoxes:
                    model = comboBox.model()
                    for index in range(comboBox.count):
                        indexqt = model.index(index, 0)
                        if comboBox.itemText(index) == 'Check':
                            model.itemFromIndex(indexqt).setCheckState(qt.Qt.Checked)
                        else:
                            model.itemFromIndex(indexqt).setCheckState(qt.Qt.Unchecked)
                # Connect files checkboxes to function
                for chk_bx_conv in self.checkboxes[0]:
                    chk_bx_conv.stateChanged.connect(self.chkBoxConvertChange)
                for chk_box_vis in self.checkboxes[1]:
                    chk_box_vis.stateChanged.connect(self.chkBoxVisibleChange)
                # Enable button
                self.ui.applyButton.toolTip = "Run algorithm"
                self.ui.applyButton.enabled = True
        else: # The button must be disabled if the condition is not met
            self.ui.applyButton.toolTip = "Select the required inputs"
            self.ui.applyButton.enabled = False
            self._bool_subj = False
            # Clear the table
            while (self.ui.tableFiles.rowCount > 0):
                self.ui.tableFiles.removeRow(0)

    def onConfigChange(self):
        """
        Function to enable/disable 'Apply' button depending on the selected file
        """
        if (os.path.isfile(str(self.ui.configFileSelector.currentPath)) and self._dir_selected):
            self.config = str(self.ui.configFileSelector.currentPath)
            # Read yaml file
            with open(self.config) as file:
                inputs_dict = yaml.load(file, Loader=yaml.FullLoader)
            data_path = str(self.ui.HippUnfoldDirSelector.directory)
            layout = BIDSLayout(data_path, config = self.resourcePath('Data/bids.json'), validate=False)
            self.files = {}
            for subj in self.list_subj:
                self.files[subj] = []
                for type_file in inputs_dict['pybids_inputs']:
                    input_filters = {
                        'subject':subj
                    }
                    dict_input = inputs_dict['pybids_inputs'][type_file]
                    # Update filter
                    input_filters.update(dict_input['pybids_filters'])
                    # Look for files based on BIDS 
                    image_files = layout.get(**input_filters)
                    tmp_files = layout.get(**input_filters, return_type='filename')
                    # Check if there are scalars attached
                    # print(tmp_files)
                    if 'scalars' in dict_input:
                        tmp_files_color = []
                        for tmp_file, image_file in zip(tmp_files, image_files):
                            labels_color = []
                            for scalar in dict_input['scalars']:
                                input_filters = {
                                    'subject':subj
                                }
                                input_filters.update(dict_input['scalars'][scalar]['pybids_filters'])
                                # Get entities from surf file
                                if 'match_entities' in dict_input['scalars'][scalar]:
                                    for entity in dict_input['scalars'][scalar]['match_entities']:
                                        input_filters[entity] = image_file.get_entities()[entity]
                                # print(input_filters)
                                color_filenames = layout.get(**input_filters, return_type='filename')
                                if 'colortable' in dict_input['scalars'][scalar]:
                                    labels_color +=  [(file, dict_input['scalars'][scalar]['colortable']) for file in color_filenames]
                                else:
                                    labels_color +=  [(file, None) for file in color_filenames]
                            tmp_files_color.append((tmp_file, labels_color))
                    else:
                        tmp_files_color = [(tmp_file, []) for tmp_file in tmp_files]
                    # Add to list of files
                    self.files[subj] += tmp_files_color
    def onVisibleAllChange(self):
        """
        Function to select all or select none
        """
        model = self.ui.VisibleAll.model()
        index_int = self.ui.VisibleAll.currentIndex
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if model.data(index, qt.Qt.CheckStateRole) == qt.Qt.Checked and index.row()!=index_int:
                model.itemFromIndex(index).setCheckState(qt.Qt.Unchecked)
            elif model.data(index, qt.Qt.CheckStateRole) == qt.Qt.Checked and index.row()==index_int:
                if self.ui.VisibleAll.currentText == 'Check':
                    for chk_bx in self.checkboxes[1]:
                        chk_bx.setCheckState(qt.Qt.Checked)
                elif self.ui.VisibleAll.currentText == 'Uncheck':
                    for chk_bx in self.checkboxes[1]:
                        chk_bx.setCheckState(qt.Qt.Unchecked)
        if len(self.checkboxes[1])>0:
            self.chkBoxVisibleChange()

    def onConvertAllChange(self):
        """
        Function to select all or select none
        """
        model = self.ui.ConvertAll.model()
        index_int = self.ui.ConvertAll.currentIndex
        for row in range(model.rowCount()):
            index = model.index(row, 0)
            if model.data(index, qt.Qt.CheckStateRole) == qt.Qt.Checked and index.row()!=index_int:
                model.itemFromIndex(index).setCheckState(qt.Qt.Unchecked)
            elif model.data(index, qt.Qt.CheckStateRole) == qt.Qt.Checked and index.row()==index_int:
                if self.ui.ConvertAll.currentText == 'Check':
                    for chk_bx in self.checkboxes[0]:
                        chk_bx.setCheckState(qt.Qt.Checked)
                elif self.ui.ConvertAll.currentText == 'Uncheck':
                    for chk_bx in self.checkboxes[0]:
                        chk_bx.setCheckState(qt.Qt.Unchecked)
        if len(self.checkboxes[0])>0:
            self.chkBoxConvertChange()

    def onDirectoryChange(self):
        """
        Function to enable/disable 'Apply' button depending on the selected file
        """
        _tmp_dir_input = str(self.ui.HippUnfoldDirSelector.directory) 
        _tmp_dir_output = str(self.ui.OutputDirSelector.directory) 

        # Bool to change button status
        # If the selected file is a valid one, the button is enabled.   
        if (os.path.exists(_tmp_dir_input) and os.path.exists(_tmp_dir_output)):
            try:
                # Update dropdown
                data_path = _tmp_dir_input
                self.BIDSLayout = BIDSLayout(data_path, validate=False)
                self.list_subj = self.BIDSLayout.get(return_type='id', target='subject')
                self.ui.subj.clear()
                self.ui.subj.addItems(['Select subject']+self.list_subj)
            except ValueError:
                self.ui.applyButton.toolTip = "Please select a valid directory"
                self.ui.applyButton.enabled = False
            # Set to true condition indicating that we have valid input directories
            self._dir_selected = True
            # Re-run config file
            self.onConfigChange()
            # Update button if both conditions are true
            if self._bool_subj:
                self.ui.applyButton.toolTip = "Run algorithm"
                self.ui.applyButton.enabled = True
        # Else, it is disabled.
        else:
            self.ui.applyButton.toolTip = "Please select a valid directory"
            self.ui.applyButton.enabled = False
            self._dir_selected = False
        
    def chkBoxVisibleChange(self):
        # Look for amount of items checked
        if len(self.checkboxes[0]) == len(self.checkboxes[1]):
            item_checked = 0
            for chk_bx_conv, chk_box_vis in zip(self.checkboxes[0], self.checkboxes[1]):
                if chk_box_vis.checkState() == qt.Qt.Checked and chk_bx_conv.checkState() == qt.Qt.Unchecked:
                    chk_box_vis.setCheckState(qt.Qt.Unchecked)
                elif chk_box_vis.checkState() == qt.Qt.Checked:
                    item_checked += 1
        else:
            item_checked = 0
            for chk_bx in self.checkboxes[1]:
                if chk_bx.checkState() == qt.Qt.Checked:
                    item_checked += 1
        # Update all/none state
        model = self.ui.VisibleAll.model()
        if item_checked == len(self.checkboxes[1]):
            state = 'Check'
        elif item_checked == 0:
            state = 'Uncheck'
        else:
            state = 'Some'
        for index in range(self.ui.VisibleAll.count):
            indexqt = model.index(index, 0)
            if self.ui.VisibleAll.itemText(index) == state:
                self.ui.VisibleAll.setCurrentIndex(index)
                model.itemFromIndex(indexqt).setCheckState(qt.Qt.Checked)
            else:
                model.itemFromIndex(indexqt).setCheckState(qt.Qt.Unchecked)
    
    def chkBoxConvertChange(self):
        # Look for amount of items checked
        item_checked = 0
        for chk_bx in self.checkboxes[0]:
            if chk_bx.checkState() == qt.Qt.Checked:
                item_checked += 1
        # Update all/none state
        model = self.ui.ConvertAll.model()
        if item_checked == len(self.checkboxes[0]):
            state = 'Check'
        elif item_checked == 0:
            state = 'Uncheck'
        else:
            state = 'Some'
        for index in range(self.ui.ConvertAll.count):
            indexqt = model.index(index, 0)
            if self.ui.ConvertAll.itemText(index) == state:
                self.ui.ConvertAll.setCurrentIndex(index)
                model.itemFromIndex(indexqt).setCheckState(qt.Qt.Checked)
            else:
                model.itemFromIndex(indexqt).setCheckState(qt.Qt.Unchecked)
        # Update visible boxes
        self.chkBoxVisibleChange()

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

        self._parameterNode.SetNodeReferenceID("HippUnfoldDir", self.ui.HippUnfoldDirSelector.currentNodeID)
        self._parameterNode.SetNodeReferenceID("OutputDir", self.ui.OutputDirSelector.currentNodeID)

        self._parameterNode.EndModify(wasModified)

    def onApplyButton(self):
        """
        Configures the behavior of 'Apply' button by connecting it to the logic function.
        """
        # Retrieve files to be converted 
        files_convert = []
        for index, chk_bx in enumerate(self.checkboxes[0]):
            if chk_bx.checkState() == qt.Qt.Checked:
                files_convert.append(self.files[self.ui.subj.currentText][index])
        # print(files_convert)
        # Retrieve files to be visible 
        files_visible = []
        for index, chk_bx in enumerate(self.checkboxes[1]):
            if chk_bx.checkState() == qt.Qt.Checked:
                files_visible.append(self.files[self.ui.subj.currentText][index][0])
        # print(files_convert)
        HippSlicerLogic().convertToSlicer(str(self.ui.OutputDirSelector.directory), 
                                           self.atlas_labels, files_convert, files_visible)

#########################################################################################
####                                                                                 ####
#### HippSlicerLogic                                                          ####
####                                                                                 ####
#########################################################################################
class HippSlicerLogic(ScriptedLoadableModuleLogic):
    """
  """

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        # Create a Progress Bar
        self.pb = qt.QProgressBar()
    
    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("LUT"):
            parameterNode.SetParameter("LUT", "Select LUT file")

    def convertToSlicer(self, OutputPath, atlas_labels_file, files_convert, files_visible):
        """
        Updates this file by changing the default _dir_chosen attribute from
        the HippSlicer and HippSlicerWidget classes so that the next time
        3D Slicer is launched, the directory to vCastSender.exe is saved.
        """
        # Read atlas label
        atlas_labels = pd.read_table(atlas_labels_file)
        print(atlas_labels.head())
        atlas_labels['lut']=atlas_labels[['r','g','b']].to_numpy().tolist()
        # Create dictionary of file types
        files_dict = {}
        for file, scalars in files_convert:
            filename = Path(file)
            ext = ''
            while filename.suffix:
                ext = filename.suffix + ext
                filename = filename.with_suffix('')
            if ext in files_dict:
                files_dict[ext].append((file,scalars))
            else:
                files_dict[ext] = [(file,scalars)]
        
        # For each type of file, run the corresponding function
        for extension in files_dict:
            if extension == '.surf.gii':
                self.convert_surf(files_dict[extension], OutputPath, files_visible)
            elif extension == '.nii.gz':
                files_dseg, _ = zip(*files_dict[extension])
                self.convert_dseg(files_dseg, OutputPath, atlas_labels, files_visible)
            else:
                print(f'File type {extension} is not supported.')   

    def convert_dseg(self, dseg_files, OutputPath, atlas_labels, files_visible):
        for dseg in dseg_files:
            # Find base file name to create output
            filename_with_extension = os.path.basename(dseg)
            base_filename = filename_with_extension.split('.', 1)[0]
            # Find parent dir
            dir_re = r'sub-.+/[a-z]+/'
            parent_dir = re.findall(dir_re, dseg, re.IGNORECASE)[0]
            # Create sub and anat folder if it doesn't exist  
            if not os.path.exists(os.path.join(OutputPath, parent_dir)):
                os.makedirs(os.path.join(OutputPath, parent_dir))
            seg_out_fname = os.path.join(OutputPath, parent_dir, f'{base_filename}.seg.nrrd')
            # print(seg_out_fname)
            # Load data from dseg file
            data_obj=nb.load(dseg)
            self.write_nrrd(data_obj, seg_out_fname, atlas_labels)
            if dseg in files_visible:
                seg = slicer.util.loadSegmentation(seg_out_fname)
                seg.CreateClosedSurfaceRepresentation()
    
    def convert_surf(self, surf_files, OutputPath, files_visible):
        for surf, label_files in surf_files:
            # Find base file name to create output
            filename_with_extension = os.path.basename(surf)
            base_filename = filename_with_extension.split('.', 1)[0]
            # Find parent dir
            dir_re = r'sub-.+/[a-z]+/'
            parent_dir = re.findall(dir_re, surf, re.IGNORECASE)[0]
            # Create surf folder if it doesn't exist
            if not os.path.exists(os.path.join(OutputPath, parent_dir)):
                os.makedirs(os.path.join(OutputPath, parent_dir))
            outFilePath = os.path.join(OutputPath, parent_dir, f'{base_filename}.vtk')
            # Extract geometric data
            gii_data = nb.load(surf)
            vertices = gii_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
            faces = gii_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
            # Extract color data
            print('aqui')
            # Add scalars
            if len(label_files)>0:
                active_scalar = None
                scalar_range = []
                arrayScalars = []
                labelsScalars = []
                label_files_df = pd.DataFrame(label_files)
                for index in label_files_df.index:
                    vert_colors_idx = nb.load(label_files_df.loc[index, 0]).agg_data()
                    name_label = os.path.basename(label_files_df.loc[index, 0]).split('.', 1)[0].split('-')[-1]
                    # print(vert_colors_idx[0:10])
                    # Extract colors from df if exists
                    if label_files_df.loc[index, 1] != None:
                        df_colors = pd.read_table(label_files_df.loc[index, 1], index_col='index')
                        print('a')
                        # Create color table
                        colorTableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLProceduralColorNode", "HippUnfoldColors")
                        colorTableNode.SetType(slicer.vtkMRMLColorTableNode.User)
                        colorTransferFunction = vtk.vtkDiscretizableColorTransferFunction()
                        for index_color in df_colors.index:
                            r = df_colors.loc[index_color, 'r']/255.0
                            g = df_colors.loc[index_color, 'g']/255.0
                            b = df_colors.loc[index_color, 'b']/255.0
                            colorTransferFunction.AddRGBPoint(index_color, r, g, b)
                        colorTableNode.SetAndObserveColorTransferFunction(colorTransferFunction)
                        # Test 
                        if len(arrayScalars) == 0:
                            arrayScalars = [tuple([scalar]) for scalar in vert_colors_idx]
                        else:
                            for idx in range(len(vert_colors_idx)):
                                arrayScalars[idx] += tuple([vert_colors_idx[idx]])
                        # Name 
                        labelsScalars.append(name_label)
                        active_scalar = name_label
                        indexes = df_colors.index.values.tolist()
                        scalar_range = (indexes[0], indexes[-1])
                    else:
                        print('b')
                        # Test 
                        if len(arrayScalars) == 0:
                            arrayScalars = [tuple([scalar]) for scalar in vert_colors_idx]
                        else:
                            for idx in range(len(vert_colors_idx)):
                                arrayScalars[idx] += tuple([vert_colors_idx[idx]])
                        # Name 
                        labelsScalars.append(name_label)
                        if active_scalar == None and index == list(label_files_df.index)[-1]:
                            active_scalar = name_label
            # Create model
            surf_pv = self.makePolyData(vertices, faces, labelsScalars, arrayScalars)
            modelNode = slicer.modules.models.logic().AddModel(surf_pv)
            # Set name
            modelNode.SetName(base_filename)
            # Set active scalar
            if len(scalar_range) > 0 and active_scalar != None:
                modelNode.GetDisplayNode().SetActiveScalar(active_scalar, vtk.vtkAssignAttribute.POINT_DATA)
                modelNode.GetDisplayNode().SetAndObserveColorNodeID(colorTableNode.GetID())
                modelNode.GetDisplayNode().SetAutoScalarRange(False)
                modelNode.GetDisplayNode().SetScalarRange(scalar_range[0], scalar_range[1])
                modelNode.GetDisplayNode().SetScalarVisibility(True)
            elif active_scalar != None:
                modelNode.GetDisplayNode().SetActiveScalar(active_scalar, vtk.vtkAssignAttribute.POINT_DATA)
                modelNode.GetDisplayNode().SetAutoScalarRange(True)
                modelNode.GetDisplayNode().SetScalarVisibility(True)
            # Set visibility 
            if surf in files_visible:
                modelNode.SetDisplayVisibility(True)
            else:
                modelNode.SetDisplayVisibility(False)
            # Export
            writer = vtk.vtkPolyDataWriter()
            writer.SetInputData(surf_pv)
            writer.SetFileName(outFilePath)
            writer.Write()

    # Functions to compute files
    def bounding_box(self, seg):
        x = np.any(np.any(seg, axis=0), axis=1)
        y = np.any(np.any(seg, axis=1), axis=1)
        z = np.any(np.any(seg, axis=1), axis=0)
        ymin, ymax = np.where(y)[0][[0, -1]]
        xmin, xmax = np.where(x)[0][[0, -1]]
        zmin, zmax = np.where(z)[0][[0, -1]]
        bbox = np.array([ymin,ymax,xmin,xmax,zmin,zmax])
        return bbox

    def get_shape_origin(self, img_data):
        bbox = self.bounding_box(img_data)
        ymin, ymax, xmin, xmax, zmin, zmax = bbox
        shape = list(np.array([ymax-ymin, xmax-xmin, zmax-zmin]) + 1)
        origin = [ymin, xmin, zmin]
        return shape, origin

    def write_nrrd(self, data_obj, out_file, atlas_labels):
        
        data=data_obj.get_fdata()
        
        keyvaluepairs = {}
        keyvaluepairs['dimension'] = 3
        keyvaluepairs['encoding'] = 'gzip'
        keyvaluepairs['kinds'] = ['domain', 'domain', 'domain']
        keyvaluepairs['space'] = 'right-anterior-superior'
        keyvaluepairs['space directions'] = data_obj.affine[:3,:3].T
        keyvaluepairs['type'] = 'double'
        
        box = self.bounding_box(data)
        seg_cut = data[box[0]:box[1]+1,box[2]:box[3]+1,box[4]:box[5]+1]
        shape, origin = self.get_shape_origin(data)
        origin = nb.affines.apply_affine(data_obj.affine, np.array([origin]))

        keyvaluepairs['sizes'] = np.array([*shape])
        keyvaluepairs['space origin'] = origin[0]
        
        for i in range(int(np.max(data))):
            col_lut=np.array(atlas_labels[atlas_labels['index']==i+1]['lut'].values[0]+[255])/255
            name = 'Segment{}'.format(i)
            keyvaluepairs[name + '_Color'] = ' '.join([f"{a:10.3f}" for a in col_lut])
            keyvaluepairs[name + '_ColorAutoGenerated'] = '1'
            keyvaluepairs[name + '_Extent'] = f'0 {shape[0]-1} 0 {shape[1]-1} 0 {shape[2]-1}'
            keyvaluepairs[name + '_ID'] = 'Segment_{}'.format(i+1)
            keyvaluepairs[name + '_LabelValue'] = '{}'.format(i+1)
            keyvaluepairs[name + '_Layer'] = '0'
            keyvaluepairs[name + '_Name'] = atlas_labels[atlas_labels['index']==i+1]['abbreviation'].values[0]
            keyvaluepairs[name + '_NameAutoGenerated'] = 1
            keyvaluepairs[name + '_Tags'] = 'TerminologyEntry:Segmentation category' +\
                ' and type - 3D Slicer General Anatomy list~SRT^T-D0050^Tissue~SRT^' +\
                'T-D0050^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|'

        keyvaluepairs['Segmentation_ContainedRepresentationNames'] = 'Binary labelmap|'
        keyvaluepairs['Segmentation_ConversionParameters'] = 'placeholder'
        keyvaluepairs['Segmentation_MasterRepresentation'] = 'Binary labelmap'
        
        nrrd.write(out_file, seg_cut, keyvaluepairs)

    def write_ply(self, filename, vertices, faces, comment=None):
        # infer number of vertices and faces
        number_vertices = vertices.shape[0]
        number_faces = faces.shape[0]
        # make header dataframe
        header = ['ply',
                'format ascii 1.0',
                'comment %s' % comment,
                'element vertex %i' % number_vertices,
                'property float x',
                'property float y',
                'property float z',
                'element face %i' % number_faces,
                'property list uchar int vertex_indices',
                'end_header'
                ]
        header_df = pd.DataFrame(header)
        # make dataframe from vertices
        vertex_df = pd.DataFrame(vertices)
        # make dataframe from faces, adding first row of 3s (indicating triangles)
        triangles = np.reshape(3 * (np.ones(number_faces)), (number_faces, 1))
        triangles = triangles.astype(int)
        faces = faces.astype(int)
        faces_df = pd.DataFrame(np.concatenate((triangles, faces), axis=1))
        # write dfs to csv
        header_df.to_csv(filename, header=None, index=False)
        with open(filename, 'a') as f:
            vertex_df.to_csv(f, header=False, index=False,
                            float_format='%.3f', sep=' ')
        with open(filename, 'a') as f:
            faces_df.to_csv(f, header=False, index=False,
                            float_format='%.0f', sep=' ')
    
    # Function to create vtkPolyData object
    def makePolyData(self, verts, faces, labelsScalars, arrayScalars):
        # https://github.com/stephan1312/SlicerEAMapReader/blob/2798100fe2aebf482a83b347c1cef18135f2df87/EAMapReader-Slicer-4.11/lib/Slicer-4.11/qt-scripted-modules/EAMapReader.py#L218-L290
        # https://programtalk.com/python-examples/vtk.vtkPolyData/
        mesh = vtk.vtkPolyData()
        pts = vtk.vtkPoints()
        for pt in verts:
            pts.InsertNextPoint( pt[0], pt[1], pt[2] )
        cells = vtk.vtkCellArray()
        for f in faces:
            cells.InsertNextCell( len(f) )
            for v in f: 
                cells.InsertCellPoint( v )
        mesh.SetPoints(pts)
        mesh.SetPolys(cells)

        # Add scalars
        scalars = []
        for j in range(len(labelsScalars)):
            scalars.append(vtk.vtkFloatArray())
            scalars[j].SetNumberOfComponents(1)
            scalars[j].SetNumberOfTuples(len(arrayScalars))
            for i in range(len(arrayScalars)):
                scalars[j].SetTuple1(i,arrayScalars[i][j])
            scalars[j].SetName(labelsScalars[j])
            mesh.GetPointData().AddArray(scalars[j])

        return mesh
        



class HippSlicerTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_HippSlicer1()

  def test_HippSlicer1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("No tests are implemented")