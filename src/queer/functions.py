import numpy as np

def fd(energy, T=0.025):
    """Fermi-Dirac distribution function."""
    import numpy as np
    return 1.0 / (np.exp(energy / T) + 1.0)




def get_rotation_matrix(from_vec, to_vec):
    from_vec = np.asarray(from_vec, dtype=float)
    to_vec = np.asarray(to_vec, dtype=float)
    u = from_vec / np.linalg.norm(from_vec)
    v = to_vec / np.linalg.norm(to_vec)
    c = np.dot(u, v)
    if np.isclose(c, 1.0):        # already aligned
        return np.eye(3)
    if np.isclose(c, -1.0):       # opposite; pick any perpendicular axis
        # here choose an axis perpendicular to u
        axis = np.array([1,0,0]) if abs(u[0]) < 0.9 else np.array([0,1,0])
        a = axis - u*np.dot(axis,u); a /= np.linalg.norm(a)
        K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        # 180°: R = I + 2 K^2
        return np.eye(3) + 2*(K@K)
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    a = axis / s
    K = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    # Rodrigues: R = I + K*s + K^2*(1 - c)
    R = np.eye(3) + K*s + (K@K)*(1 - c)
    return R
