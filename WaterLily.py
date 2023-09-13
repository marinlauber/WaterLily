#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import subprocess
import os
import shutil
import vtk
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from matplotlib.colors import ListedColormap

# Choose colormap
cmap = plt.cm.RdBu
my_cmap = cmap(np.arange(cmap.N))
ls, ds = np.linspace(-1, 1, cmap.N, retstep=True)
my_cmap[:,-1] = np.where(abs(ls) < ds, 0, 1)
lotus_cmap = ListedColormap(my_cmap)

def run(n_proc=0,run_folder='test',read_folder=None):
    "setup and run Lotus using lotus.f90 and the files in postproc"

    print('Number of proccessors :{}'.format(n_proc))
    print('Run folder            :{}'.format(run_folder))

    if read_folder:
        print('Read folder           :{}'.format(read_folder))
    else:
        print('No read folder')

    if os.path.exists(run_folder):
        print('Folder '+run_folder+' exists!')
        if read_folder=='./':
            print('Resuming in place')
        else:
            print('Moving contents to trash')
            subprocess.call('trash '+run_folder+'/*', shell=True)
    else:
        print('Creating '+run_folder)
        os.makedirs(run_folder)

    print('Setting up in '+run_folder)
    if os.path.isdir('postproc'):
        for file_name in os.listdir('postproc'):
            full_file_name = os.path.join('postproc', file_name)
            if os.path.isfile(full_file_name):
                shutil.copy(full_file_name, run_folder)
    if os.path.isfile('lotus.f90'):
        shutil.copy('lotus.f90', run_folder)

    print('Making executable ')
    os.chdir(run_folder)
    subprocess.call('make -C $MGLHOME/geom_lib/ libgeom.a', shell=True)
    subprocess.call('make -C $MGLHOME/oop/ libfluid.a', shell=True)
    subprocess.call('make -f $MGLHOME/oop/Makefile lotus', shell=True)
    print('Finished executable ')

    print('Runeing executable ')
    if read_folder:
        if n_proc==0:
            subprocess.call('time ./lotus '+read_folder, shell=True)
        else:
            subprocess.call('mpirun -n '+str(n_proc)+' ./lotus '+read_folder, shell=True)
    else:
        if n_proc==0:
            subprocess.call('time ./lotus', shell=True)
        else:
            subprocess.call('mpirun -n '+str(n_proc)+' ./lotus', shell=True)

    print('Run all python files for postprocessing')
    for file_name in os.listdir('.'):
        if file_name.endswith('.py'):
            print(file_name)
            subprocess.call('python3 '+file_name, shell=True)

    print('Popping back up')
    os.chdir('../.')


def replace(template,dic):
    """
    Write lotus.f90 file by replacing the dic on the template.f90
    and return a potential folder name
    """
    f1 = open(template,'r')
    f2 = open('lotus.f90','w')
    for text in f1:
        for i, j in dic.items():
            text = text.replace(i, j)
        f2.write(text)
    f1.close()
    f2.close()
    return '_'.join([i+'_'+j for i,j in dic.items()])


def read_vti(fname):
	reader = vtk.vtkXMLPImageDataReader()
	reader.SetFileName(fname)
	reader.Update()
	data = reader.GetOutput()
	pointData = data.GetPointData()

	sh = data.GetDimensions()[::-1]
	ndims = len(sh)
    
	# get vector field
	v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims,))
	vec = []
	for d in range(ndims):
		a = v[..., d]
		vec.append(a)
	# get scalar field
	sca = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1,))

	# Generate grid
	# nPoints = data.GetNumberOfPoints()
	(xmin, xmax, ymin, ymax, zmin, zmax) = data.GetBounds()
	grid3D = np.mgrid[xmin:xmax + 1, ymin:ymax + 1, zmin:zmax + 1]

	return np.transpose(np.array(vec), (0,3,2,1)), np.transpose(sca, (3,2,1,0))[0,:,:,:], grid3D


def read_vtr(fname):
	reader = vtk.vtkXMLPRectilinearGridReader()
	reader.SetFileName(fname)
	reader.Update()
	data = reader.GetOutput()
	pointData = data.GetPointData()

	sh = data.GetDimensions()[::-1]
	ndims = len(sh)

	# get vector field
	v = np.array(pointData.GetVectors("Velocity")).reshape(sh + (ndims,))
	vec = []
	for d in range(ndims):
		a = v[..., d]
		vec.append(a)
	vec = np.array(vec)

	# get scalar field
	sca = np.array(pointData.GetScalars('Pressure')).reshape(sh + (1,))

	# get grid
	x = np.array(data.GetXCoordinates())
	y = np.array(data.GetYCoordinates())
	z = np.array(data.GetZCoordinates())

	return np.transpose(vec, (0,3,2,1)), np.transpose(sca, (3,2,1,0))[0,:,:,:], np.array([x, y, z], dtype=object)

def read_vtu(fname):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fname)
    reader.Update()
    data = reader.GetOutput()
    vtkData = data.GetPoints().GetData()
    
    Npoints = data.GetPointData().GetNumberOfTuples()

    p = np.array(data.GetPointData().GetArray("Pressure")).reshape((Npoints))
    v = np.array(data.GetPointData().GetArray("Forces")).reshape((Npoints,3))
    coords = np.array([vtkData.GetTuple3(x) for x in range(Npoints)])
    conectiv = []

    for i in range(data.GetNumberOfCells()):
        conectiv.append([data.GetCell(i).GetPointIds().GetId(j)
            for j in range(data.GetCell(i).GetPointIds().GetNumberOfIds())])

    conectiv = np.array(conectiv, dtype=np.float64)

    return coords, conectiv, p, np.transpose(v)

def read_vtk(data_set):
    if(data_set[-3:]=="vti"):
        return read_vti(data_set) 
    elif(data_set[-3:]=="vtr"):
        return read_vtr(data_set)

def face_to_center(vec, d):
    n = np.array(vec.shape,dtype=int)
    ndims=2 if n[2]==1 else 3
    off=np.zeros(3,dtype=int)
    if d>-1: off[d]=1 if d+1<=ndims else 0
    n[:ndims] = n[:ndims]-1
    ne=n[:3]+off
    return 0.5*(vec[off[0]:ne[0],off[1]:ne[1],off[2]:ne[2]]+
                vec[:n[0],:n[1],:n[2]])

def center(x):
    return 0.5*(x[:-1]+x[1:])

def read_collection(fname):
    f = open(fname)
    time=[]; file=[]
    for line in f:
        if line[1:8]=="DataSet":
            time.append(float(line.split()[1].
                        replace('timestep="','').replace('"','')))
            file.append(line.split()[-1].
                        replace('file="','').replace('"/>',''))
    time = np.array(time,dtype=float)
    filename_list = np.array(file)
    return time, filename_list

def _plot_XY_grid(ax, x, y, dist, L, every, xlab=r"x/L", ylab=r"y/L"):
    ax.vlines(x[::every], ymin=min(y), ymax=max(y), color='gray', lw=0.2)
    ax.hlines(y[::every], xmin=min(x), xmax=max(x), color='gray', lw=0.2)
    ax.contour(x, y, dist.T, colors='k', levels=[0.])
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_xlim(min(x-50/L),max(x+50/L))
    ax.set_ylim(min(y-50/L),max(y+50/L))
    ax.set_aspect('equal', adjustable='box')
    return ax

def _plot_table(ax, *args):
    tot = []; cell_text = []
    for i,arg in enumerate(args):
        dx = arg[1:]-arg[:-1]
        mn, mx = min(abs(dx)), max(abs(dx))
        cell_text.append([str(len(dx)),"%.4f" % mn,"%.4f" % mx, "%.1f" % (mx/mn)] )
        tot.append(len(dx))
    cell_text.append( [str(np.product(np.array(tot))),"n-a","n-a","n-a"] )
    r=["X","Y","Z"]; rowlab = [r[j]+"-axis" for j in range(i+1)]; rowlab.append(r"$\sum$")
    the_table = ax.table(cellText=cell_text,
                        rowLabels=rowlab,
                        colLabels=[r"$N_i$",r"$\Delta x_i$ min",r"$\Delta x_i$ max",r"$r$ max"],
                        loc='center')
    the_table.scale(1, 1.5)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.axis("off")
    return ax

# spatial Jacobian that is used to compute FTLE
def Jacobian(X,Y,dx,dy):

    # shapes
    nx, ny = X.shape

    # storage arrays
    J = np.empty([2,2],float)
    FTLE = np.empty([nx,ny],float)
    
    for i in range(1,nx-1):
        for j in range(1,ny-1):
            J[0,0] = (X[i+1,j]-X[i-1,j])/(2*dx)
            J[0,1] = (X[i,j+1]-X[i,j-1])/(2*dy)
            J[1,0] = (Y[i+1,j]-Y[i-1,j])/(2*dx)
            J[1,1] = (Y[i,j+1]-Y[i,j-1])/(2*dy)
			
			# Green-Cauchy tensor
            C = np.dot(np.transpose(J),J)
			# its largest eigenvalue
            lamda = np.linalg.eigvals(C)
            FTLE[i,j] = max(lamda)
    return FTLE

class DataSet(object):

    def __init__(self, data_path , data_set) -> None:

        self.data_path = data_path
        self.data_set = data_set
        self._read_collection()
        self.L = 1.

    def set_scale(self, scale):
        self.L = scale

    def get_snapshot(self, time):
        """
        return the id of the snapshot that is the closest to the time provided
        """
        return np.argmin(np.round((self.time-time),2)**2)

    def _read_collection(self) -> None:
        self.time, self.filename_list = read_collection(self.data_path+self.data_set)

    def __repr__(self):
        string = f"{self.__class__.__name__}(\n"
        for i in range(self.time.shape[0]):
            string += f"timestep = {self.time[i]} file = {self.filename_list[i]}\n"
        string += ")"
        return string


class LotusDataSet(DataSet):

    def __init__(self, data_path , data_set, list_of_variables=None) -> None:
        super().__init__(data_path , data_set)
        self.set_of_variables = ["u","v","w","p"]
        if list_of_variables!=None:
            self.set_of_variables = list_of_variables

        # initialize storage for snapshots
        self.data = xr.Dataset()
        

    def load_snapshot(self, numerical_identifier):
        return self._load_array(self.data_path+self.filename_list[numerical_identifier],
                                self.time[numerical_identifier])

    def load_snapshot_d(self, numerical_identifier):
        self.data = xr.concat(
            (self._load_array(self.data_path+self.filename_list[numerical_identifier],
                              self.time[numerical_identifier]),
            self.data),
            dim="time",
        )

    def load_time_series(self):
        self.data = xr.concat(
            (
                self._load_array(self.data_path+file, add_time=time)
                for file, time in zip(self.filename_list, self.time)
            ),
            dim="time",
        )

    def _load_array(self, filename: str, add_time: float = -1, attrs: dict = None):
        """This method reads a field from Lotus and wraps it into a 
        :obj:`xarray.DataArray` with the appropriate dimensions,
        coordinates and attributes.
        Parameters
        ----------
        filename : str
            Name of the file.
        add_time : bool, optional
            Add time as a coordinate (default is :obj:`True`).
        attrs : dict_like, optional
            Attributes to assign to the new instance :obj:`xarray.DataArray`.
        Returns
        -------
        :obj:`xarray.DataArray`
            Data array containing values loaded from the disc.
        """
        # create an empty dataset
        dataset = xr.Dataset()

        # read the lotus data, vti or vtr alike
        (u,v,w),p,(x,y,z) = read_vtk(filename)

        # we must put all the data field at the celll center
        p = face_to_center(p,-1)
        u = face_to_center(u,0)
        v = face_to_center(v,1)
        w = face_to_center(w,2)
        
        # same for the coordinates, exept for 2D sims
        if(len(x)>1): x=center(x)
        if(len(y)>1): y=center(y)
        if(len(z)>1): z=center(z)

        # generate a dictionnary of coordinates
        coords = dict(x=(["x"], x/self.L),y=(["y"], y/self.L),z=(["z"], z/self.L))
        
        # shape
        shape = u.shape

        # is this in a time-series?
        if(add_time!=-1):
            coords["time"] = [add_time]
            shape = (*u.shape,1)

        # append to dataset
        for var,field in zip(self.set_of_variables,[u,v,w,p]):
            dataset[var] = xr.DataArray(data=field.reshape(shape),
                                        dims=coords.keys(),
                                        coords=coords,
                                        name="Lotus Data",
                                        attrs=attrs
                                        )

        return dataset

    def buildInterpolator(self, L=1):
        # remove z axis which is useless
        data = np.squeeze(np.moveaxis(self.data.to_array().data,0,-1))
        data = data[...,:2]
        x = self.data.x.values/L
        y = self.data.y.values/L
        z = self.data.z.values/L
        time = self.data.time.values/L
        coords = (x,y,z,time) if len(z)>1 else (x,y,time)
        self.interp_ = RegularGridInterpolator(coords, 
                                               data,
                                               method="linear",
                                               bounds_error=False,
                                               fill_value=None)
    

    def at(self, r, t):
        return self.interp_((r[:,0],r[:,1],t))


    def computeFTLE(self, r, shape, grid, t0, t1, dt):

        # FORWARD vs BACKWARD
        sign = 1
        if(t1<t0): sign = -1

        # integrate in time
        for time in np.arange(t0, t1, sign*dt):
            k1 = dt*self.at(r,time)
            k2 = dt*self.at(r+0.5*k1,time+0.5*dt)
            k3 = dt*self.at(r+0.5*k2,time+0.5*dt)
            k4 = dt*self.at(r+k3,time+dt)
            r += sign*(k1+2*k2+2*k3+k4)/6

        # compute jacobian
        X = r[:,0].reshape(shape)
        Y = r[:,1].reshape(shape)
        FTLE = Jacobian(X, Y, grid[0], grid[1])

        return np.log(np.sqrt(FTLE[1:-1,1:-1]))


    def show_grid(self, save=False):
        grid = self._load_array(self.data_path+'datp/bodyF.0'+self.filename_list[0][-5:])
        if len(grid.z)==1:
            self._plot_2D_Grid(grid, every=2)
        else:
            self._plot_3D_Gird(grid, every=2)
        plt.tight_layout(pad=1.08, h_pad=1.2)
        if save: plt.savefig("LotusGrid.png", dpi=600)
        plt.show()


    def _plot_2D_Grid(self, grid, every):
        fig = plt.figure(figsize=(8,6))
        ax = plt.subplot2grid((3,3), (0,0), colspan=3,rowspan=2, fig=fig)
        t  = plt.subplot2grid((3,3), (2,1), fig=fig)
        ax = _plot_XY_grid(ax, grid.x, grid.y, grid.p[:,:,0], self.L, every)
        t  = _plot_table(t, grid.x.values, grid.y.values)


    def _plot_3D_Gird(self, grid, every):
        """
        1st angle projection
        """
        fig = plt.figure(figsize=(10,6))
        xz = plt.subplot2grid((2,3), (0,0), colspan=2)
        xy = plt.subplot2grid((2,3), (1,0), colspan=2)
        yz = plt.subplot2grid((2,3), (0,2))
        t  = plt.subplot2grid((2,3), (1,2)) 
        xy = _plot_XY_grid(xy, grid.x, grid.y, grid.p[:,:,np.argmin(abs(grid.z))], self.L, every)
        yz = _plot_XY_grid(yz, -grid.y[::-1],grid.z, grid.p[np.argmin(abs(grid.x)),::-1,::],
                           self.L, every, xlab=r"y/L", ylab=r"z/L")
        xz = _plot_XY_grid(xz, grid.x, grid.z, grid.p[:,np.argmin(abs(grid.y)),:], self.L,
                           every, ylab=r"z/L")
        t  = _plot_table(t, grid.x.values, grid.y.values, grid.z.values)


# @xr.register_dataset_accessor("Lotus")
# def FirstDerivative(DataArray,axis="x"):
#     lab = {"x":0,"y":1,"z":2}
#     off=np.zeros(3,dtype=int); off[lab[axis]]=1
#     n = np.array(DataArray.shape,dtype=int)
#     ndims=2 if n[2]==1 else 3
#     n[:ndims] = n[:ndims]-1
#     res = np.zeros(n)
#     coords = {}
#     for i in ["x","y","z"]:
#         x = 0.5*(DataArray[i].values[1:]+DataArray[i].values[:-1])
#         if(len(x)==0):
#             coords.update({i:DataArray.coords[i].values})
#         else:
#             coords.update({i:x})
#     coords.update({"time":DataArray.coords["time"].values})
#     # dx = abs(DataArray[axis].values[:-1]-DataArray[axis].values[1:])
#     dx =1.
#     ne=n[:3]+off
#     res[:,:,:,:] = (DataArray.values[off[0]:ne[0],off[1]:ne[1],off[2]:ne[2],:]-
#                     DataArray.values[:n[0],:n[1],:n[2],:])/dx
#     return xr.DataArray(data=res,
#                         dims=coords.keys(),
#                         coords=coords,
#                         name="First derivative of Data")

class Mesh(DataSet):

    def __init__(self, data_path) -> None:
        super().__init__(data_path, 'interF.vtu.pvd')

    def load_snapshot(self, numerical_identifier):
        return self._load_array(self.data_path+self.filename_list[numerical_identifier],
                                self.time[numerical_identifier])

    def _load_array(self, filename: str, add_time: float = -1):

        # read triangular data
        tri,conectiv,p,v  = read_vtu(filename)

        return tri
    
    