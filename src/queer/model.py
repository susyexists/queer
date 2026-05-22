# Numerical tools
from scipy.constants import physical_constants
# Matrix inversion
from numpy.linalg import inv
import numpy as np
from pathlib import Path
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
from .kpath import KPath, named_k_path
from .mesh import mesh_crystal
from tqdm import tqdm

class model:
    def __init__(self, hr="wannier90_hr.dat", path="./",nscf=False,poscar=False, ef=0,read_ef=False,shift=0,num_core=False):
        if num_core!=False:
            self.num_cores= num_core
        else:
            self.num_cores = multiprocessing.cpu_count()
        self.shift = shift
        self.path = Path(path).expanduser()
        self.hr = Path(hr)
        self.nscf = Path(nscf) if nscf else None
        self.poscar = Path(poscar) if poscar else None
        if read_ef:
            self.fermi_energy = read_efermi(self.path / self.nscf)+self.shift
        else:
            self.fermi_energy=ef
        if nscf:
            self.g_vec = utils.read_gvec(self.path / self.nscf)
        if poscar:
            lattice_vector = utils.read_poscar(self.path / self.poscar)
            self.g_vec = utils.crystal2reciprocal(lattice_vector)
        self.data = read_hr(self.path / self.hr)
        self.hopping = self.data[0]
        self.nbnd = int(np.sqrt(len(self.data[0])/len(self.data[2])))
        self.points = len(self.data[2])
        self.sym = self.data[2]
        self.h = self.hopping.reshape(self.points, self.nbnd*self.nbnd)
        self.x = self.data[1].reshape(3, self.points, self.nbnd*self.nbnd)

    def kpath(self, point_names, n_points=200, poscar=None, **seekpath_kwargs):
        """Build a high-symmetry k-path from point names using this model's POSCAR."""
        if poscar is None:
            if self.poscar is None:
                raise ValueError("model.kpath requires a POSCAR. Initialize the model with poscar=... or pass poscar=...")
            poscar_path = self.path / self.poscar
        else:
            poscar_path = Path(poscar)
            if not poscar_path.is_absolute():
                poscar_path = self.path / poscar_path
        return named_k_path(point_names, n_points, poscar_path, **seekpath_kwargs)

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
        transform = np.dot(self.sym, np.exp(-1j * kx * 2*np.pi) * self.h).reshape(self.nbnd, self.nbnd)
    
        # 1) Hard checks
        if not np.isfinite(transform).all():
            raise FloatingPointError(f"Non-finite in transform at k={k}")
    
        # 2) Enforce Hermitian (Wannier HR *should* give Hermitian H(k), but numerics can break it)
        transform = 0.5 * (transform + transform.conj().T)
    
        try:
            val = np.linalg.eigh(transform)[0]
            return val
        except np.linalg.LinAlgError:
            # fallback 1: slightly regularize diagonal (tiny)
            eps = 1e-12
            transform2 = transform + eps * np.eye(transform.shape[0], dtype=transform.dtype)
            try:
                return np.linalg.eigh(transform2)[0]
            except np.linalg.LinAlgError:
                # fallback 2: use eig (less stable/guaranteed real, but won't crash the whole run)
                w = np.linalg.eigvals(transform)
                return np.sort(w.real)
    
    def calculate_energy(self, path, band_index=False):
        path = np.asarray(path.path if isinstance(path, KPath) else path)
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

    def plot_electron_path(self, band, sym, labels, ylim=(-10, 10), save=None, temp=None,title=False, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        for i in band:
            ax.plot(i, c="blue")
        ax.set_xticks(sym, labels, fontsize=15)
        ax.set_xlim(sym[0], sym[-1])
        for i in sym[1:-1]:
            ax.axvline(i, c="black", linestyle="--")
        ax.axhline(0, linestyle="--", color="red")
        if ylim is not None:
            ax.set_ylim(ylim)
        if title!=False:
            ax.set_title(title)
        if temp != None:
            ax.set_title(f"σ = {temp}", fontsize=15)
        if self.shift != 0:
            ax.set_title(
                r"$\delta \epsilon_{Fermi} = $"f" {self.shift} eV", fontsize=15)
        ax.set_ylabel("Energy (eV)", fontsize=15)
        if save != None:
            ax.figure.savefig(save)
        return ax

    def plot_band_path(
        self,
        path_or_points,
        sym=None,
        labels=None,
        n_points=200,
        ylim=(-10, 10),
        save=None,
        band_index=False,
        ax=None,
        **seekpath_kwargs,
    ):
        """Calculate and plot electron bands along a path.

        ``path_or_points`` can be either an explicit k-point array, in which
        case ``sym`` and ``labels`` must be supplied, a :class:`queer.kpath.KPath`,
        or a named path string such as ``"GAMMA-X-W-K-GAMMA-L"``.
        """
        if isinstance(path_or_points, str):
            k_path = self.kpath(path_or_points, n_points, **seekpath_kwargs)
            sym, path, labels = k_path
        elif isinstance(path_or_points, KPath):
            path = path_or_points.path
            sym = path_or_points.sym if sym is None else sym
            labels = path_or_points.labels if labels is None else labels
        else:
            if sym is None or labels is None:
                raise ValueError("Explicit paths require sym and labels.")
            path = np.asarray(path_or_points)

        bands = self.calculate_energy(path, band_index=band_index)
        axis = self.plot_electron_path(bands, sym, labels, ylim=ylim, save=save, ax=ax)
        return bands, axis

    def plot_brillouin_zone(self, mesh=None, mesh_repeat=0, align_dir=None,
                            ax=None, figsize=(7, 7), title=None,
                            max_mesh_pts=2000, mesh_clip_bz=False,
                            mesh_style='scatter'):
        """Plot the Brillouin zone with optional k-point mesh overlay.

        Requires a POSCAR (pass ``poscar=`` when constructing the model).

        Parameters
        ----------
        mesh : (N, 3) array, optional
            k-points in Å⁻¹ to scatter on top of the BZ.
        mesh_repeat : int
            Number of BZ shells over which to tile the mesh using G-vector
            offsets.  0 = original mesh only; 1 = adds the 26 nearest G-vector
            images, etc.
        align_dir : array-like of length 3, optional
            Cartesian direction (Å⁻¹) for an arrow drawn from Γ to the BZ
            boundary.  The arrow length is clipped to the BZ face it first hits.
        ax : Axes3D, optional
            Existing axes to draw into; a new figure is created if omitted.
        figsize, title : passed to plt.figure / ax.set_title.

        Returns
        -------
        fig, ax
        """
        try:
            import ase.io
            from ase.dft.bz import bz_vertices
            from ase.dft.kpoints import get_special_points
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError("plot_brillouin_zone requires ase: pip install ase")

        if self.poscar is None:
            raise ValueError(
                "plot_brillouin_zone requires a POSCAR. "
                "Initialize the model with poscar=..."
            )

        atoms = ase.io.read(self.path / self.poscar)

        # self.g_vec rows are b1, b2, b3 in 2π/Å — exactly what bz_vertices expects
        faces = bz_vertices(self.g_vec)
        polys = [f[0] for f in faces]

        # fractional → Cartesian: k_cart = k_frac @ g_vec
        sp_cart = {
            name: frac @ self.g_vec
            for name, frac in get_special_points(atoms.cell).items()
        }

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
        else:
            fig = ax.figure

        ax.add_collection3d(
            Poly3DCollection(polys, alpha=0.08, facecolor='steelblue',
                             edgecolor='steelblue', linewidth=0.8)
        )

        if mesh is not None:
            m = np.asarray(mesh, dtype=float)
            n_pts = len(m)
            n_side = int(round(n_pts ** 0.5))
            is_structured = (n_side * n_side == n_pts)

            if mesh_clip_bz:
                # Find the G-vector nearest to the mesh centroid, then fold
                # every point back into the first BZ by subtracting that G
                # (reduced zone scheme).
                centroid = m.mean(axis=0)
                G_near = np.zeros(3)
                best_d = np.inf
                for n1 in range(-3, 4):
                    for n2 in range(-3, 4):
                        for n3 in range(-3, 4):
                            G = (n1 * self.g_vec[0]
                                 + n2 * self.g_vec[1]
                                 + n3 * self.g_vec[2])
                            d = np.linalg.norm(centroid - G)
                            if d < best_d:
                                best_d = d
                                G_near = G.copy()
                m = m - G_near  # fold into first BZ

            if mesh_style in ('surface', 'wireframe') and is_structured:
                stride = max(1, n_side // max(1, int(max_mesh_pts ** 0.5)))
                gs = m.reshape(n_side, n_side, 3)[::stride, ::stride].copy()
                if mesh_clip_bz:
                    flat = gs.reshape(-1, 3)
                    inside = np.ones(len(flat), dtype=bool)
                    for verts, normal in faces:
                        inside &= (flat @ normal <= verts[0] @ normal + 1e-10)
                    flat[~inside] = np.nan
                    gs = flat.reshape(gs.shape)
                X, Y, Z = gs[..., 0], gs[..., 1], gs[..., 2]
                if mesh_style == 'wireframe':
                    ax.plot_wireframe(X, Y, Z, color='orange', alpha=0.6,
                                      linewidth=0.5, rstride=1, cstride=1)
                else:
                    ax.plot_surface(X, Y, Z, color='orange', alpha=0.3,
                                    linewidth=0, antialiased=True)
            else:
                # Scatter fallback (style='scatter' or unstructured input).
                if mesh_clip_bz:
                    inside = np.ones(len(m), dtype=bool)
                    for verts, normal in faces:
                        inside &= (m @ normal <= verts[0] @ normal + 1e-10)
                    m = m[inside]
                elif mesh_repeat > 0:
                    bz_radius = np.linalg.norm(np.vstack(polys), axis=1).max()
                    shells = [m]
                    for n1 in range(-mesh_repeat, mesh_repeat + 1):
                        for n2 in range(-mesh_repeat, mesh_repeat + 1):
                            for n3 in range(-mesh_repeat, mesh_repeat + 1):
                                if n1 == n2 == n3 == 0:
                                    continue
                                offset = (n1 * self.g_vec[0]
                                          + n2 * self.g_vec[1]
                                          + n3 * self.g_vec[2])
                                if np.linalg.norm(offset) <= 2 * bz_radius:
                                    shells.append(m + offset)
                    m = np.vstack(shells)
                if len(m) > max_mesh_pts:
                    rng = np.random.default_rng(0)
                    m = m[rng.choice(len(m), max_mesh_pts, replace=False)]
                ax.scatter(m.T[0], m.T[1], m.T[2],
                           s=3, alpha=0.4, color='orange', label='ARPES mesh')

        for name, pt in sp_cart.items():
            ax.scatter(*pt, s=40, color='red', zorder=5)
            ax.text(pt[0], pt[1], pt[2], f' {name}', fontsize=10, color='red')

        lim = np.abs(np.vstack(polys)).max() * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel(r'$k_x$ [Å$^{-1}$]')
        ax.set_ylabel(r'$k_y$ [Å$^{-1}$]')
        ax.set_zlabel(r'$k_z$ [Å$^{-1}$]')
        ax.set_title(title or 'Brillouin Zone')

        if align_dir is not None:
            d = np.asarray(align_dir, dtype=float)
            d_hat = d / np.linalg.norm(d)
            # Find how far the ray Γ + t*d_hat travels before hitting a BZ face
            t_bz = np.inf
            for verts, normal in faces:
                denom = d_hat @ normal
                if abs(denom) > 1e-10:
                    t = (verts[0] @ normal) / denom
                    if 0 < t < t_bz:
                        t_bz = t
            arrow_tip = d_hat * (t_bz if np.isfinite(t_bz) else lim * 0.8)
            ax.quiver(0, 0, 0, *arrow_tip,
                      color='cyan', linewidth=2, arrow_length_ratio=0.15)

        return fig, ax


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
