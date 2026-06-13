import sys

# state file generated using paraview version 5.13.2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *

def main(path, time_step):
    #### disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [3528, 1888]
    renderView1.AxesGrid = 'Grid Axes 3D Actor'
    renderView1.OrientationAxesVisibility = 0
    renderView1.CenterOfRotation = [3.141594886779785, 3.141594886779785, 3.141594886779785]
    renderView1.StereoType = 'Crystal Eyes'
    #renderView1.CameraPosition = [3.141594886779785, 24.16555762702477, 3.141594886779785]
    #renderView1.CameraFocalPoint = [3.141594886779785, 3.141594886779785, 3.141594886779785]
    #renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    center = [3.141594886779785, 3.141594886779785, 3.141594886779785]
    dist = 15.0
    
    renderView1.CameraPosition = [
        center[0] + dist,
        center[1] + dist,
        center[2] + dist
    ]
    renderView1.CameraFocalPoint = center
    #renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraViewUp = [0.0, 1.0, 0.0]
    #renderView1.CameraViewUp = [0.0, 0.0, 1.0]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 5.4414019607011825
    renderView1.LegendGrid = 'Legend Grid Actor'
    renderView1.PolarGrid = 'Polar Grid Actor'
    renderView1.BackEnd = 'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary = materialLibrary1

    SetActiveView(None)

    # ----------------------------------------------------------------
    # setup view layouts
    # ----------------------------------------------------------------

    # create new layout object 'Layout #1'
    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    layout1.SetSize(3528, 1888)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView1)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'XML Partitioned Rectilinear Grid Reader'
    time_step_data = XMLPartitionedRectilinearGridReader(
        registrationName=f'time_step-{time_step}.pvtr', 
        FileName=[f'{path}/time_step-{time_step}.pvtr']
    )
    time_step_data.CellArrayStatus = ['phi1']
    time_step_data.TimeArray = 'None'

    # create a new 'Cell Data to Point Data'
    cellDatatoPointData1 = CellDatatoPointData(registrationName='CellDatatoPointData1', Input=time_step_data)
    cellDatatoPointData1.CellDataArraytoprocess = ['phi1']

    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=cellDatatoPointData1)
    contour1.ContourBy = ['POINTS', 'phi1']
    contour1.Isosurfaces = [0.5]
    contour1.PointMergeMethod = 'Uniform Binning'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from cellDatatoPointData1
    cellDatatoPointData1Display = Show(cellDatatoPointData1, renderView1, 'UniformGridRepresentation')

    # trace defaults for the display properties.
    cellDatatoPointData1Display.Representation = 'Outline'
    cellDatatoPointData1Display.ColorArrayName = [None, '']
    cellDatatoPointData1Display.LineWidth = 3.0
    cellDatatoPointData1Display.SelectNormalArray = 'None'
    cellDatatoPointData1Display.SelectTangentArray = 'None'
    cellDatatoPointData1Display.SelectTCoordArray = 'None'
    cellDatatoPointData1Display.TextureTransform = 'Transform2'
    cellDatatoPointData1Display.OSPRayScaleArray = 'phi1'
    cellDatatoPointData1Display.OSPRayScaleFunction = 'Piecewise Function'
    cellDatatoPointData1Display.Assembly = ''
    cellDatatoPointData1Display.SelectedBlockSelectors = ['']
    cellDatatoPointData1Display.SelectOrientationVectors = 'None'
    cellDatatoPointData1Display.ScaleFactor = 0.6283190000000001
    cellDatatoPointData1Display.SelectScaleArray = 'None'
    cellDatatoPointData1Display.GlyphType = 'Arrow'
    cellDatatoPointData1Display.GlyphTableIndexArray = 'None'
    cellDatatoPointData1Display.GaussianRadius = 0.031415950000000005
    cellDatatoPointData1Display.SetScaleArray = ['POINTS', 'phi1']
    cellDatatoPointData1Display.ScaleTransferFunction = 'Piecewise Function'
    cellDatatoPointData1Display.OpacityArray = ['POINTS', 'phi1']
    cellDatatoPointData1Display.OpacityTransferFunction = 'Piecewise Function'
    cellDatatoPointData1Display.DataAxesGrid = 'Grid Axes Representation'
    cellDatatoPointData1Display.PolarAxes = 'Polar Axes Representation'
    cellDatatoPointData1Display.ScalarOpacityUnitDistance = 0.042510954350033964
    cellDatatoPointData1Display.OpacityArrayName = ['POINTS', 'phi1']
    cellDatatoPointData1Display.ColorArray2Name = ['POINTS', 'phi1']
    cellDatatoPointData1Display.SliceFunction = 'Plane'
    cellDatatoPointData1Display.Slice = 128
    cellDatatoPointData1Display.SelectInputVectors = [None, '']
    cellDatatoPointData1Display.WriteLog = ''

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    cellDatatoPointData1Display.ScaleTransferFunction.Points = [-1.010543921457628e-18, 0.0, 0.5, 0.0, 1.000000000008294, 1.0, 0.5, 0.0]

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    cellDatatoPointData1Display.OpacityTransferFunction.Points = [-1.010543921457628e-18, 0.0, 0.5, 0.0, 1.000000000008294, 1.0, 0.5, 0.0]

    # init the 'Plane' selected for 'SliceFunction'
    cellDatatoPointData1Display.SliceFunction.Origin = [3.141595, 3.141595, 3.141595]

    # show data from contour1
    contour1Display = Show(contour1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    contour1Display.Representation = 'Surface'
    contour1Display.ColorArrayName = ['POINTS', '']
    contour1Display.SelectNormalArray = 'Normals'
    contour1Display.SelectTangentArray = 'None'
    contour1Display.SelectTCoordArray = 'None'
    contour1Display.TextureTransform = 'Transform2'
    contour1Display.OSPRayScaleArray = 'phi1'
    contour1Display.OSPRayScaleFunction = 'Piecewise Function'
    contour1Display.Assembly = ''
    contour1Display.SelectedBlockSelectors = ['']
    contour1Display.SelectOrientationVectors = 'None'
    contour1Display.ScaleFactor = 0.4663903594017029
    contour1Display.SelectScaleArray = 'phi1'
    contour1Display.GlyphType = 'Arrow'
    contour1Display.GlyphTableIndexArray = 'phi1'
    contour1Display.GaussianRadius = 0.023319517970085146
    contour1Display.SetScaleArray = ['POINTS', 'phi1']
    contour1Display.ScaleTransferFunction = 'Piecewise Function'
    contour1Display.OpacityArray = ['POINTS', 'phi1']
    contour1Display.OpacityTransferFunction = 'Piecewise Function'
    contour1Display.DataAxesGrid = 'Grid Axes Representation'
    contour1Display.PolarAxes = 'Polar Axes Representation'
    contour1Display.SelectInputVectors = ['POINTS', 'Normals']
    contour1Display.WriteLog = ''

    # init the 'Piecewise Function' selected for 'ScaleTransferFunction'
    contour1Display.ScaleTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

    # init the 'Piecewise Function' selected for 'OpacityTransferFunction'
    contour1Display.OpacityTransferFunction.Points = [0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]

    # ----------------------------------------------------------------
    # setup animation scene, tracks and keyframes
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # get time animation track
    timeAnimationCue1 = GetTimeTrack()

    # initialize the animation scene

    # get the time-keeper
    timeKeeper1 = GetTimeKeeper()

    # initialize the timekeeper

    # initialize the animation track

    # get animation scene
    animationScene1 = GetAnimationScene()

    # initialize the animation scene
    animationScene1.ViewModules = renderView1
    animationScene1.Cues = timeAnimationCue1
    animationScene1.AnimationTime = 0.0

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(cellDatatoPointData1)
    # ----------------------------------------------------------------


    ##--------------------------------------------
    ## You may need to add some code at the end of this python script depending on your usage, eg:
    #
    ## Render all views to see them appears
    # RenderAllViews()
    #
    ## Interact with the view, usefull when running from pvpython
    # Interact()
    #
    ## Save a screenshot of the active view
    SaveScreenshot(
        f"{path}/time_step-{time_step}-phi0.5_contour_3D.png", 
        ImageResolution=[1600, 1600], 
        OverrideColorPalette="WhiteBackground"
    )
    #
    ## Save a screenshot of a layout (multiple splitted view)
    # SaveScreenshot("path/to/screenshot.png", GetLayout())
    #
    ## Save all "Extractors" from the pipeline browser
    # SaveExtracts()
    #
    ## Save a animation of the current active view
    # SaveAnimation()
    #
    ## Please refer to the documentation of paraview.simple
    ## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
    ##--------------------------------------------

if __name__ == '__main__':

    path      = sys.argv[1]
    time_step = sys.argv[2]

    print('Initializing the pvpython script...')
    main(path, time_step)
    print('Done for', path, 'and time step', time_step)
