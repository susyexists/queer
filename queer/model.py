# Numerical tools
from scipy.constants import physical_constants
# Matrix inversion
from numpy.linalg import inv
import numpy as np
# Plotting
import matplotlib.pyplot as plt
# Data analysis
import pandas as pd
# Parallel computation
from joblib import Parallel, delayed
import multiprocessing
# Physical constants
import psutil
# plt.style.use('./queer/resources/neon.mplstyle')

from .functions import fd


from .epw import epw
from . import utils
from tqdm import tqdm

class model:
    def __init__(self, hr="wannier90_hr.dat", path="./",nscf=False,poscar=False, ef=0,read_ef=False,shift=0,num_core=False):
        if num_core!=False:
            self.num_cores= num_core
        else:
            self.num_cores = multiprocessing.cpu_count()
        self.shift = shift
        self.path = path
        if read_ef:
            self.fermi_energy = read_efermi(path+nscf)+self.shift
        else:
            self.fermi_energy=ef
        if nscf:
            self.g_vec = utils.read_gvec(path+nscf)
        if poscar:
            
            lattice_vector = utils.read_poscar(path+poscar)
            self.g_vec = utils.crystal2reciprocal(lattice_vector)
        self.data = read_hr(path+hr)
        self.hopping = self.data[0]
        self.nbnd = int(np.sqrt(len(self.data[0])/len(self.data[2])))
        self.points = len(self.data[2])
        self.sym = self.data[2]
        self.h = self.hopping.reshape(self.points, self.nbnd*self.nbnd)
        self.x = self.data[1].reshape(3, self.points, self.nbnd*self.nbnd)

    def fourier(self, k):
        kx = np.tensordot(k, self.x, axes=(0, 0))
        transform = np.dot(self.sym, np.exp(-1j*self.super_cell*kx)
                           * self.h).reshape(self.nbnd, self.nbnd)
        return(transform)

    def eig(self, k):
        val = []
        vec = []
        for i in range(len(k)):
            sol = np.linalg.eigh(self.fourier(k[i]))
            val.append(sol[0])
            vec.append(sol[1])
        return (val, vec)

    def solver(self, k):
        kx = np.tensordot(k, self.x, axes=(0, 0))
        transform = np.dot(self.sym, np.exp(-1j*kx*2*np.pi)
                           * self.h).reshape(self.nbnd, self.nbnd)
        val, vec = np.linalg.eigh(transform)
        return(val)

    def calculate_energy(self, path, band_index=False):
        path = path
        results = Parallel(n_jobs=self.num_cores)(
            delayed(self.solver)(i) for i in path)
        res = np.array(results).T-self.fermi_energy
        if band_index==False:
            return (res)
        else:
            return (res[band_index])

    def suscep(self, point, mesh, mesh_energy, mesh_fermi, bands,T=1,delta=0.0000001,fermi_shift =0 ):
        real= 0
        imag = 0
        shifted_energy = self.calculate_energy(point+mesh)
        shifted_fermi = fd(shifted_energy,T)
        for i in bands:
            for j in bands:
                num = mesh_fermi[i]-shifted_fermi[j]
                den = mesh_energy[i]-shifted_energy[j]+1j*delta
                real += np.average(num/den)
                imag += np.average(delta_function(mesh_energy[i])*delta_function(shifted_energy[j]))
        return([-real.real,imag])
    
    def suscep_path(self,q_path,k_mesh,band_index,T=1):
        en_k = self.calculate_energy(k_mesh)
        fd_k = fd(en_k,T)
        res = [self.suscep(point=q,mesh= k_mesh,mesh_energy=en_k,mesh_fermi = fd_k,bands=band_index) for q in tqdm(q_path)]
        return np.array(res).T

    def plot_electron_path(self, band, sym, labels, ylim=None, save=None, temp=None,title=False):
        # Plot band
        plt.figure(figsize=(6, 6))
        for i in band:
            plt.plot(i, c="blue",)
        plt.xticks(ticks=sym, labels=labels, fontsize=15)
        plt.xlim(sym[0], sym[-1])
        for i in sym[1:-1]:
            plt.axvline(i, c="black", linestyle="--")
        plt.axhline(0, linestyle="--", color="red")
        if ylim == None:
            plt.ylim(-0.6, 0.8)
        else:
            plt.ylim(ylim)
        if title!=False:
            plt.title(title)
        if temp != None:
            plt.title(f"σ = {temp}", fontsize=15)
        if self.shift != 0:
            plt.title(
                r"$\delta \epsilon_{Fermi} = $"f" {self.shift} eV", fontsize=15)
        plt.ylabel("Energy (eV)", fontsize=15)
        if save != None:
            plt.savefig(save)


def Symmetries(fstring):
    f = open(fstring, 'r')
    x = np.zeros(0)
    for i in f:
        x = np.append(x, float(i.split()[-1]))
    f.close()
    return x


def plot_fs(band, fs_thickness=0.01, title=None):
    # Imaging cross sections of fermi surface using a single calculation
    df = pd.DataFrame()
    x,y = mesh_crystal(int(np.sqrt(len(band))))
    df['x'] = x
    df['y'] = y
    df['E'] = band
    fs = df.query(f' {-fs_thickness} <= E <= {fs_thickness}')
    fig = plt.figure(figsize=(6, 6))
    plt.scatter(fs.x, fs.y)
    plt.ylim(-2, 2)
    plt.xlim(-2, 2)
    if title != None:
        plt.title(title, fontsize=15)
    plt.show()




def delta_function(x, epsilon=0.00001):
    return (1 / np.pi) * epsilon / (x ** 2 + epsilon ** 2)





def read_hr(path):
    lines = open(path, 'r').readlines()
    sym_line = int(np.ceil(float(lines[2].split()[0])/15))+3
    sym = np.array([int(lines[i].split()[j]) for i in range(3, sym_line)
                    for j in range(len(lines[i].split()))])
    hr_temp = np.array([float(lines[i].split()[j]) for i in range(
        sym_line, len(lines)) for j in range(len(lines[i].split()))])
    hr = hr_temp.reshape(-1, 7).T
    x = hr[0:3]
    hopping = hr[5]+1j*hr[6]
    return (hopping, x, sym)


def read_efermi(path):
    lines = open(path, 'r').readlines()
    e_fermi = 0
    for i in lines:
        if "the Fermi energy is" in i:
            e_fermi = float(i.split()[-2])
            return e_fermi



def density_of_states(energy, band_index=False, dE=1e-2):
    if band_index:
        E = energy[band_index]
    else:
        E=energy
    # Initial empty array for dos
    dos = np.zeros(len(E))
    # Iterate over each energy
    for i in range(len(E)):
        # Delta function approxiation for given value of energy over all states
        delta_array = np.where(abs(E[i]-E) < dE, np.ones(len(E)), 0)
        delta_average = np.average(delta_array)
        dos[i] = delta_average
    return dos

def ram_check():
    # Getting % usage of virtual_memory ( 3rd field)
    print('RAM memory % used:', psutil.virtual_memory()[2])
    # Getting usage of virtual_memory in GB ( 4th field)
    print('RAM Used (GB):', psutil.virtual_memory()[3]/1000000000)
    
def rotate(vector,angle):
    matrix = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])    
    transform = np.dot(matrix,vector.T)
    return(transform)

def triangle_mesh(N):
    x,y = mesh_cartesian(N).T
    df = pd.DataFrame()
    df['x']=x
    df['y']=y
    triangle = df.query("y<=sqrt(3)*x").query("y<=-sqrt(3)*x+sqrt(3)").values
    return triangle.T


    


def find_cross(band,parameter):
    xs=[]
    point_pair=[]
    for i in range(len(band)):
        grad = np.gradient(band[i])
        for j in range(1,len(band[i])):
            if abs(grad[j]-grad[j-1])>parameter:
                # print(i,j)
                point_pair.append([i,j])
    # print(point_pair)
    for i in point_pair:
        point=i[1]
        begin=i[0]
        for j in point_pair:
            if i[0]!=j[0]:
                if i[1]==j[1]:
                    end=j[0]
                    xs.append([begin,end,point])
    xs_sort = xs[xs[:, 2].argsort()]    
    return np.array(xs_sort)

