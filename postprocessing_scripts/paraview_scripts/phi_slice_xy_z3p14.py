import sys

# state file generated using paraview version 5.13.2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

from paraview.simple import *


def _set_first_existing_property(proxy, names, value):
    last_error = None
    for name in names:
        try:
            setattr(proxy, name, value)
            return name
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Could not set any of {names}; last error: {last_error}")


def main(path, time_step):
    paraview.simple._DisableFirstRenderCameraReset()

    zslice = 3.14159265
    center = [3.14159265, 3.14159265, zslice]

    materialLibrary1 = GetMaterialLibrary()

    renderView1                                 =   CreateView('RenderView')
    renderView1.ViewSize                        =   [1600, 1600]
    renderView1.AxesGrid                        =   'Grid Axes 3D Actor'
    renderView1.OrientationAxesVisibility       =   0
    renderView1.CenterOfRotation                =   center
    renderView1.CameraPosition                  =   [3.14159265, 3.14159265, 20.0]
    #renderView1.CameraFocalPoint = [3.14159265, 3.14159265, 3.14159] 
    renderView1.CameraFocalPoint                =   center
    renderView1.CameraViewUp                    =   [0.0, 1.0, 0.0]
    renderView1.CameraParallelProjection        =   1
    #renderView1.CameraParallelScale             =   3.25
    renderView1.CameraParallelScale             =   5.0
    renderView1.BackEnd                         =   'OSPRay raycaster'
    renderView1.OSPRayMaterialLibrary           =   materialLibrary1

    layout1 = CreateLayout(name='Layout #1')
    layout1.AssignView(0, renderView1)
    #layout1.SetSize(1600, 1600)
    layout1.SetSize(1600, 1600)
    SetActiveView(renderView1)

    time_step_data = XMLPartitionedRectilinearGridReader(
        registrationName=f'time_step-{time_step}.pvtr',
        FileName=[f'{path}/time_step-{time_step}.pvtr']
    )
    _set_first_existing_property(time_step_data, ['CellArrays', 'CellArrayStatus'], ['phi1'])
    time_step_data.TimeArray = 'None'

    cellDatatoPointData1 = CellDatatoPointData(
        registrationName='CellDatatoPointData1',
        Input=time_step_data
    )
    _set_first_existing_property(
        cellDatatoPointData1,
        ['CellDataArraytoprocess', 'CellDataArrays'],
        ['phi1']
    )

    slice1 = Slice(registrationName='Slice_z_3p14', Input=cellDatatoPointData1)
    slice1.SliceType = 'Plane'
    slice1.SliceType.Origin = center
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]

    sliceDisplay = Show(slice1, renderView1, 'GeometryRepresentation')
    sliceDisplay.Representation = 'Surface'
    ColorBy(sliceDisplay, ('POINTS', 'phi1'))

    #Debug
    print("Slice cells:", slice1.GetDataInformation().GetNumberOfCells())
    pinfo = slice1.GetDataInformation().GetPointDataInformation()
    arr = pinfo.GetArrayInformation("phi1")
    print(arr.GetComponentRange(0))

    phi1LUT = GetColorTransferFunction('phi1')
    phi1LUT.RescaleTransferFunction(0.0, 1.0)
    sliceDisplay.SetScalarBarVisibility(renderView1, True)

    #Scalar bar transformation
    phi1Bar                      =   GetScalarBar(phi1LUT, renderView1)
    phi1Bar.WindowLocation       =   'Any Location'
    phi1Bar.Position             =   [0.88, 0.20]        #Move up/down
    phi1Bar.ScalarBarLength      =   0.45                #Taller
    phi1Bar.ScalarBarThickness   =   28                  #Wider
    phi1Bar.Title                =   'phi1'

    renderView1.Update()
    SaveScreenshot(
        f'{path}/time_step-{time_step}-phi1_xy_z3p14.png',
        renderView1,
        ImageResolution=[1600, 1600],
        OverrideColorPalette='WhiteBackground'
    )


if __name__ == '__main__':
    path = sys.argv[1]
    time_step = sys.argv[2]
    print('Initializing the pvpython script...')
    main(path, time_step)
    print('Done for', path, 'and time step', time_step)
