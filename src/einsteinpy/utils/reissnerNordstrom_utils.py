import astropy.units as u
import numpy as np

from einsteinpy import constant

def charge_legth_scale(Q, c=constant.c.value, G=constant.G.value, Cc=constant.coulombs_const.value):
    """
    Returns a length scale corrosponding to the Electric Charge Q of the mass

    Parameters
    ----------
    Q : float
        Charge on the massive body
    c : float
        Speed of light. Defaults to 299792458 (SI units)
    G : float
        Gravitational constant. Defaults to 6.67408e-11 (SI units)
    Cc : float
        Coulumb's constant. Defaults to 8.98755e9 (SI units)

    Returns
    -------
    float
        returns (coulomb's constant^0.5)*(Q/c^2)*G^0.5

    """
    return (Q / (c ** 2)) * np.sqrt(G * Cc)


def metric(
    r,
    theta,
    M,
    Q,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns the Kerr-Newman Metric

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    M : float
        Mass of the massive body
    Q : float
        Charge on the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,4)

    """
    Rs = schwarzschild_radius_dimensionless(M, c, G)
    Rq = charge_legth_scale(Q, c=constant.c.value, G=constant.G.value, Cc=constant.coulombs_const.value)
    m = np.zeros((4, 4), dtype=float)
    c2 = c ** 2
    # set the diagonal/off-diagonal terms of metric
    m[0, 0] = (1.0 - Rs/r + Rq**2/r**2)
    m[1, 1] = -(1.0 -Rs/r +Rq**2/r**2)**(-1)
    m[2, 2] = -r**2
    m[3, 3] = -(r**2)*(np.sin(theta)**2)
    )
    return m


def metric_inv(
    r,
    theta,
    M,
    Q,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns the inverse of Kerr-Newman Metric

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    M : float
        Mass of the massive body
    Q : float
        Charge on the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,4)

    """
    return np.linalg.inv(metric(r, theta, M, Q, c, G, Cc))


def christoffels(
    r,
    theta,
    M,
    Q,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns the 3rd rank Tensor containing Christoffel Symbols for Kerr-Newman Metric

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    M : float
        Mass of the massive body
    Q : float
        Charge on the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,4,4)

    """
    Rs = schwarzschild_radius_dimensionless(M, c, G)
    chl = np.zeros(shape=(4, 4, 4), dtype=float)
    c2 = c ** 2
    Q2 = Q ** 2
    r2 = r ** 2
    chl[0, 1, 0] = (M*r + Q2)/r*(r*(r-2*M)- Q2)
    chl[1, 0, 0] = (Mr + Q2)*(r*(2*M - r)+Q2)/(r**5)
    chl[1, 1, 1] = (M*r + Q2)/(2*M*(r2) + (Q2)*r - r**3)
    chl[1, 2, 2] = 2*M - (Q2/r) + r 
    chl[1, 3, 3] = (np.sin(theta)**2)*(r2 - 2*M*r - Q2)
    chl[2, 2, 1] = 1/r
    chl[2, 3, 3] = -np.cos(theta) * np.sin(theta)
    chl[3, 3, 1] = 1/r
    chl[3, 3, 2] = 1 / np.tan(theta)
    return chl


def em_potential(
    r,
    theta,
    Q,
    M,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns a 4-d vector(for each component of 4-d space-time) containing the electromagnetic potential around a Kerr-Newman body

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    Q : float
        Charge on the massive body
    M : float
        Mass of the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,)
    
    """
    vec = np.zeros((4,), dtype=float)
    vec[0] = Q/r
    return vec


def maxwell_tensor_covariant(
    r,
    theta,
    Q,
    M,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns a 2nd rank Tensor containing Maxwell Tensor with lower indices for Kerr-Newman Metric

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    Q : float
        Charge on the massive body
    M : float
        Mass of the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,4)

    """
    c2 = c ** 2
    m = np.zeros((4, 4), dtype=float)
    # set the tensor terms
    m[0, 1] = ((-rq) / (rh2 ** 2)) * (rh2 - drh2dr * r)
    m[0, 2] = r * rq * drh2dth / (rh2 ** 2)
    m[3, 1] = (c2 * a * rq * (np.sin(theta) ** 2) / (G * M * (rh2 ** 2))) * (
        rh2 - r * drh2dr
    )
    m[3, 2] = (c2 * a * rq * r / (G * M * (rh2 ** 2))) * (
        (2 * np.sin(theta) * np.cos(theta) * rh2) - (drh2dth * (np.sin(theta) ** 2))
    )
    for i, j in [(0, 1), (0, 2), (3, 1), (3, 2)]:
        m[j, i] = -m[i, j]
    return m


def maxwell_tensor_contravariant(
    r,
    theta,
    Q,
    M,
    c=constant.c.value,
    G=constant.G.value,
    Cc=constant.coulombs_const.value,
):
    """
    Returns a 2nd rank Tensor containing Maxwell Tensor with upper indices for Kerr-Newman Metric

    Parameters
    ----------
    
    r : float
        Distance from the centre
    theta : float
        Angle from z-axis
    Q : float
        Charge on the massive body
    M : float
        Mass of the massive body
    c : float
        Speed of light
    G : float
        Gravitational constant
    Cc : float
        Coulomb's constant
    Returns
    -------
    ~numpy.array
        Numpy array of shape (4,4)

    """
    mcov = maxwell_tensor_covariant(r, theta, a, Q, M, c, G, Cc)
    ginv = metric_inv(r, theta, M, a, Q, c, G, Cc)
    # contravariant F = contravariant g X covariant F X transpose(contravariant g)
    # but g is symettric
    return np.matmul(np.matmul(ginv, mcov), ginv)


def reissnerNordstrom_time_velocity(pos_vec, vel_vec, mass, Q):
    """
    Velocity of coordinate time wrt proper metric

    Parameters
    ----------
    pos_vector : ~numpy.array
        Vector with r, theta, phi components in SI units
    vel_vector : ~numpy.array
        Vector with velocities of r, theta, phi components in SI units
    mass : ~astropy.units.kg
        Mass of the body
    Q : ~astropy.units.C
        Charge on the massive body

    Returns
    -------
    ~astropy.units.one
        Velocity of time

    """
    _scr = utils.schwarzschild_radius(mass).value
    Qc = Q.to(u.C)
    g = metric(pos_vec[0], pos_vec[1], mass.value, Qc.value)
    A = g[0, 0]
    B = 2 * g[0, 3]
    C = (
        g[1, 1] * (vel_vec[0] ** 2)
        + g[2, 2] * (vel_vec[1] ** 2)
        + g[3, 3] * (vel_vec[2] ** 2)
        - 1
    )
    D = (B ** 2) - (4 * A * C)
    vt = (B + np.sqrt(D)) / (2 * A)
    return vt * u.one

