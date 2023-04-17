import os
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import numpy as np
import re
from qt import QStandardItem

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
        self.ui.subj.addItems(['Select subject'])
            

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
        # Set state of button
        if self.ui.subj.currentIndex != 0:
            self._bool_subj = True
            if self._dir_selected:
                self.ui.applyButton.toolTip = "Run algorithm"
                self.ui.applyButton.enabled = True
        else: # The button must be disabled if the condition is not met
            self.ui.applyButton.toolTip = "Select the required inputs"
            self.ui.applyButton.enabled = False
            self._bool_subj = False

    def onConfigChange(self):
        """
        Function to enable/disable 'Apply' button depending on the selected file
        """
        if os.path.isfile(str(self.ui.configFileSelector.currentPath)): # Add case where the input is not a bids dir
            self.config = str(self.ui.configFileSelector.currentPath)
            # Read yaml file
            with open(self.config) as file:
                inputs_dict = yaml.load(file, Loader=yaml.FullLoader)
            data_path = os.path.join(str(self.ui.HippUnfoldDirSelector.directory), 'hippunfold')
            layout = BIDSLayout(data_path, validate=False)
            files = []
            for type_file in inputs_dict['pybids_inputs']:
                input_filters = {
                    'subject':'P022'
                }
                config_filters = inputs_dict['pybids_inputs'][type_file]['filters']
                #Remove regex wc
                regex_filter = None
                if 'custom_regex' in config_filters:
                    regex_filter = config_filters['custom_regex']
                    del config_filters['custom_regex']
                # Update filter
                input_filters.update(config_filters)
                # Look for files based on BIDS 
                tmp_files = layout.get(**input_filters, return_type='filename')
                # Filter based on regex if requested
                if regex_filter != None:
                    r = re.compile(regex_filter)
                    tmp_files = list(filter(r.match, tmp_files))
                # Add to list of files
                files += tmp_files
            for file in files:
                print(file)
        # Button should be activated only after files have been choosen
        # # If the selected file is a valid one, the button is enabled.
        # if (len(self._tmp_dir)>0 and os.path.isfile(self._tmp_dir)) and self._tmp_dir.endswith('vCastSender.exe') and (self._tmp_dir != self._dir_chosen):
        #     self.ui.applyButton.toolTip = "Set directory"
        #     self.ui.applyButton.enabled = True
        # # A path has already been set. 
        # elif len(self._tmp_dir)>0 and (self._tmp_dir == self._dir_chosen):
        #     self.ui.applyButton.toolTip = "Please select a new path to vCastSender.exe"
        #     self.ui.applyButton.enabled = False
        # # Else, it is disabled.
        # else:
        #     self.ui.applyButton.toolTip = "Please select a valid directory for vCastSender.exe"
        #     self.ui.applyButton.enabled = False
    

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
                data_path = os.path.join(_tmp_dir_input, 'hippunfold')
                layout = BIDSLayout(data_path, validate=False)
                list_subj = layout.get(return_type='id', target='subject')
                self.ui.subj.clear()
                self.ui.subj.addItems(['Select subject']+list_subj)
            except ValueError:
                self.ui.applyButton.toolTip = "Please select a valid directory"
                self.ui.applyButton.enabled = False
            # Set to true condition indicating that we have valid input directories
            self._dir_selected = True
            # Update botton if both conditions are true
            if self._bool_subj:
                self.ui.applyButton.toolTip = "Run algorithm"
                self.ui.applyButton.enabled = True
        # Else, it is disabled.
        else:
            self.ui.applyButton.toolTip = "Please select a valid directory"
            self.ui.applyButton.enabled = False
            self._dir_selected = False
        


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
        HippSlicerLogic().convertToSlicer(str(self.ui.HippUnfoldDirSelector.directory), 
                                           str(self.ui.OutputDirSelector.directory), 
                                           self.atlas_labels, self.config)
    

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

    def convertToSlicer(self, HippUnfoldDirPath, OutputPath, atlas_labels_file, config_file):
        """
        Updates this file by changing the default _dir_chosen attribute from
        the HippSlicer and HippSlicerWidget classes so that the next time
        3D Slicer is launched, the directory to vCastSender.exe is saved.
        """
        # Read atlas label
        atlas_labels = pd.read_table(atlas_labels_file)
        atlas_labels['lut']=atlas_labels[['r','g','b']].to_numpy().tolist()
        # pybids
        data_path = os.path.join(HippUnfoldDirPath, 'hippunfold')
        layout = BIDSLayout(data_path, validate=False)

        # # Test yml
        # with open(config_file) as file:
        #     inputs_dict = yaml.load(file, Loader=yaml.FullLoader)
        # Retrieve filenames of dseg files
        dseg_files = layout.get(subject='P022', extension='nii.gz', suffix='dseg', datatype='anat', return_type='filename')
        # Retrieve filenames of surf files
        surf_files = []
        for isurf in ('inner','midthickness','outer'):
            surf_files += layout.get(subject='P022', extension='.surf.gii', suffix=f'{isurf}', return_type='filename')
        r = re.compile("^.+_space-T1w_den-0p5mm_label-hipp_.+.surf.gii$")
        surf_files = list(filter(r.match, surf_files))

        
        
        

    def convert_dseg(self, dseg_files, OutputPath, atlas_labels):
        for dseg in dseg_files:
            # Find base file name to create output
            filename_with_extension = os.path.basename(dseg)
            base_filename = filename_with_extension.split('.', 1)[0]
            # Create anat folder if it doesn't exist
            if not os.path.exists(os.path.join(OutputPath, 'anat/')):
                os.mkdir(os.path.join(OutputPath, 'anat/'))
            seg_out_fname = os.path.join(OutputPath, 'anat/', f'{base_filename}.seg.nrrd')
            # Load data from dseg file
            data_obj=nb.load(dseg)
            self.write_nrrd(data_obj, seg_out_fname, atlas_labels)
            seg = slicer.util.loadSegmentation(seg_out_fname)
            seg.CreateClosedSurfaceRepresentation()
    
    def convert_surf(self):
        for surf in surf_files:
            # Find base file name to create output
            filename_with_extension = os.path.basename(surf)
            base_filename = filename_with_extension.split('.', 1)[0]
            # Create surf folder if it doesn't exist
            if not os.path.exists(os.path.join(OutputPath, 'surf/')):
                os.mkdir(os.path.join(OutputPath, 'surf/'))
            gii_out_fname = os.path.join(OutputPath, 'surf/', f'{base_filename}.ply')
            gii_data = nb.load(surf)
            vertices = gii_data.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
            faces = gii_data.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
            self.write_ply(gii_out_fname,vertices,faces,'SPACE=RAS')
            slicer.util.loadModel(gii_out_fname)
            
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