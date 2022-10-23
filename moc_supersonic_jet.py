import numpy as np
import matplotlib.pyplot as plt


def m_to_nu(gamma, M):
    """
    Calculates the Prandtl-Meyer angle for a given Mach number.

    :param gamma: Ratio of specific heats
    :param M: Mach number
    :return: Prandtl-Meyer angle [rad]
    """
    return np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt(((gamma-1)/(gamma+1))*(M**2-1)))-np.arctan(np.sqrt(M**2-1))


def dnu_dm(gamma, M):
    """
    Calculates the derivative of the Prandtl-Meyer function for a given Mach number.

    :param gamma: Ratio of specific heats
    :param M: Mach number
    :return: Local derivative of the Prandtl-Meyer function [rad]
    """
    return ((1/(1+((gamma-1)/(gamma+1))*(M**2-1)))-1/(M**2))*M/np.sqrt(M**2-1)


def p_to_m_isentropic(gamma, p, p0):
    """
    Calculates the Mach number for a given static and total pressure, by using the isentropic relation.

    :param gamma: Ratio of specific heats
    :param p: Static pressure
    :param p0: Total pressure
    :return: Mach number
    """
    return np.sqrt((2/(gamma-1))*((p0/p)**((gamma-1)/gamma)-1))


def m_to_p_isentropic(gamma, p0, M):
    """
    Calculates the static pressure for a given Mach number and total pressure, by using the isentropic relation.

    :param gamma: Ratio of specific heats [-]
    :param p0: Total pressure [atm], [bar], [Pa]
    :param M: Mach number [-]
    :return: Static pressure [atm], [bar], [Pa]
    """
    return p0*(1+((gamma-1)/2)*M**2)**(-gamma/(gamma-1))


def nu_to_m(gamma, nu, M_init):
    """
    Performs a Newton method to find the Mach number for a given Prandtl-Meyer angle.

    :param gamma: Ratio of specific heats
    :param nu: Prandtl-Meyer angle [rad]
    :param M_init: Initial guess for the Mach number
    :return: The approximate Mach number for the given Prandtl-Meyer angle
    """
    x0 = M_init  # Initial guess (M = M_1)
    e = 1
    i = 0
    while e > 1e-10 and i < 100:
        xi = x0 - (m_to_nu(gamma, x0) - nu) / dnu_dm(gamma, x0)
        e = abs(xi - x0)
        x0 = xi
        i += 1
    return x0


def calc_xy(deriv1, deriv2, x1, y1, x2, y2):
    """
    Calculates the intersection of two straight lines, by using the locations of two points, and the slopes of the
    lines passing through those points. The slopes of both lines should be different.

    :param deriv1: Derivative (slope) of the line through the first point
    :param deriv2: Derivative (slope) of the line through the second point
    :param x1: X-location of the first point
    :param y1: Y-location of the first point
    :param x2: X-location of the second point
    :param y2: Y-location of the second point
    :return: X- and Y-location of the intersection point
    """
    return (deriv1*x1-deriv2*x2-y1+y2)/(deriv1-deriv2),\
           (deriv2*(-y1+x1*deriv1-x2*deriv1)+y2*deriv1)/(deriv1-deriv2)


def data_on_char(res, points, phi, nu, M, mu, p, x, y, x0, y0, points_char):
    """
    Add extra data points on the characteristic line (for visualisation purposes). All conditions are constant on these
    lines, so we take the conditions of the last point of each line (the point that is 'currently' being calculated).

    :param res: Resolution on the characteristics, i.e. the number of data points desired
    :param points: Array of all points that are calculated; only used to find the correct shape of the new array
    :param phi: Flow angle [rad]
    :param nu: Prandtl-Meyer angle [rad]
    :param M: Mach number
    :param mu: Mach angle [rad]
    :param p: Static pressure
    :param x: X-location of the current point
    :param y: Y-location of the current point
    :param x0: X-location of the first point
    :param y0: Y-location of the first point
    :param points_char: Array of all 'extra' points that have been added on the characteristics; each point contains
    all flow properties
    :return: The previously entered points_char, with additional points for the current characteristic
    """
    points_temp = np.zeros((res, len(points[0])))
    points_temp[:, :] = np.array([[phi, nu, M, mu, p, x, y]])
    points_temp[:, -2] = np.linspace(x0, x, res)  # x
    points_temp[:, -1] = np.linspace(y0, y, res)
    return np.vstack((points_char, points_temp))


def ccw(A, B, C):
    """
    Calculates whether three points are counterclockwise, based on their slopes. Is later used to calculate whether
    two line segments intersect. A and B are the streamline, while C is the first point of the characteristic.
    A, B, and C should be an array, in the form: [[x, y], [x, y], ...]

    :param A: An array containing all first points
    :param B: An array containing all second points
    :param C: An array containing all third points
    :return: An array of booleans stating whether the points A, B, and C are in counterclockwise order.
    """
    return np.array((C[:, 1]-A[:, 1])*(B[:, 0]-A[:, 0]) > (B[:, 1]-A[:, 1])*(C[:, 0]-A[:, 0]))


def intersect(A, B, C, D):
    """Calculates intersection based on whether both triangles have the same orientation
    A and B are the streamline, C and D are the characteristic. A, B, C, and D should be an array, in the form:
    [[x, y], [x, y], ...]

    :param A: An array containing all begin points of the first line
    :param B: An array containing all end points of the first line
    :param C: An array containing all begin points of the second line
    :param D: An array containing all end points of the second line
    :return: An array of booleans stating whether the lines A, B cross the lines C, D.
    """
    return np.logical_and(ccw(A, C, D) != ccw(B, C, D), ccw(A, B, C) != ccw(A, B, D))


# ---------------------------------- USER DEFINED PARAMETERS -----------------------------------
# Initial conditions and constants
gamma = 1.4
p_a = 1  # [atm]
M_1 = 2
p_1 = 2*p_a
H = 1  # Height (diameter) of the exit

# Parameters regarding the numerical scheme and plotting
N_char = 15  # Number of characteristics in which the expansion fan will be divided (at least 3)
res = 100  # "Resolution" of the straight lines that in itself do not contain data points
show_char = True  # Whether the characteristics are to be plotted or not
show_jb = True  # Whether the jet boundary is to be plotted or not
show_stream = True  # Whether the streamline is to be plotted or not
show_shock = True  # Whether the shock location should be shown
show_sym = True  # Whether the symmetry axis should be shown
show_interpolation = True  # Whether the interpolation 'heatmap' should be shown
show_points = False  # Whether all points of the characteristic intersections (non-simple regions) are shown
show_extra_points = False  # Whether all additionally calculated points on the characteristics and boundary are shown
show_text_regions = False  # Whether the text appears that shows the definition of region 1 and 2
show_text_points1 = False  # Whether the text appears that shows the data structure in non-simple region 1
show_text_points2 = False  # Whether the text appears that shows the data structure in non-simple region 2

# ------------------------- CALCULATION OF CONSTANT VALUES AND DEFINITION OF EXPANSION FAN ----------------------------
# Properties of uniform region 1
phi_1 = 0  # radians
nu_1 = m_to_nu(gamma, M_1)  # radians
mu_1 = np.arcsin(1/M_1)

# Properties of uniform region 2 (after expansion fan)
# NOTE: M_jetboundary, nu_jetboundary = M_2, nu_2 over the entire flow
p_0 = p_1*(1+((gamma-1)/2)*M_1**2)**(gamma/(gamma-1))  # Total pressure is constant, because of isentropic flow
p_2 = p_a
M_2 = p_to_m_isentropic(gamma, p_2, p_0)
nu_2 = m_to_nu(gamma, M_2)
mu_2 = np.arcsin(1/M_2)
phi_2 = nu_2-nu_1+phi_1

# Discretise expansion fan
fan = np.array([[phi_1, nu_1, M_1, mu_1, p_1, nu_1+phi_1]])  # Structure: [phi, nu, M, mu, p, V-]
for x in range(N_char)[1:]:  # All characteristics are gamma-
    phi = (phi_2-phi_1)*x/(N_char-1)+phi_1
    nu = nu_1 - phi_1 + phi
    M = nu_to_m(gamma, nu, M_1)
    mu = np.arcsin(1/M)
    p = m_to_p_isentropic(gamma, p_0, M)
    V_minus = nu+phi
    fan = np.vstack((fan, np.array([[phi, nu, M, mu, p, V_minus]])))

# Initialise arrays, for loop, and figure
points = np.array([[np.nan]*7])  # Dummy to be able to append to. Structure: [phi, nu, M, mu, p, x, y]
points_char = np.array([[np.nan]*7])  # Dummy, is used to make extra data points for the characteristic lines
lines_char = np.array([[np.nan]*10])  # Dummy: [phi, nu, M, mu, p, x0, y0, x1,  y1, gamma]. 0 = gamma-, 1 = gamma+
fig1 = plt.figure()
ax1 = fig1.add_subplot()

# -------------------------------- FIRST WAVE INTERACTION ----------------------------------------------
# i are the incoming gamma- characteristics, j are the outgoing gamma+ characteristics.
step = 0  # track progress of the for-loop
for j in range(N_char):
    for i in range(N_char)[j:]:
        # Calculate the properties
        if i == j:  # These are the points on the symmetry axis. This point is the start of a new j.
            phi = 0
            nu = fan[i, 5]  # nu = V_minus(i)
            V_plus = nu - phi  # constant over the entire gamma+ characteristic, so only calculate when j changes
        else:  # Is guaranteed to happen after the previous if-statement has already happened, so V_plus is defined.
            phi = (fan[i, 5]-V_plus)/2
            nu = (fan[i, 5]+V_plus)/2
        M = nu_to_m(gamma, nu, M_1)
        mu = np.arcsin(1/M)
        p = m_to_p_isentropic(gamma, p_0, M)

        # Calculate the location
        if j == 0:  # first row of points uses the pre-defined characteristics
            deriv_minus = np.tan(fan[i, 0] - fan[i, 3])  # Take the pre-defined characteristic
            if i == 0:  # at symmetry line, no gamma+ available. This if-condition corresponds to one point only.
                y = H/2
                x = (y-H)/deriv_minus
            else:
                deriv_plus = np.tan(0.5*(points[-1, 0]+points[-1, 3]+phi+mu))
                xi, yi, xj, yj = 0, H, points[-1, 5], points[-1, 6]
                x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 1]))
            # Add points on the characteristics
            points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, 0, H, points_char)
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, 0, H, x, y, 0]))
        else:  # the rest of the points uses not the pre-defined characteristics, but the previously calculated points
            deriv_minus = np.tan(0.5*(points[-(N_char-j), 0]-points[-(N_char-j), 3]+phi-mu))  # calculate the characteristic
            if i == j:  # at the symmetry line, no gamma+ available
                y = H/2
                x = points[-(N_char-j), 5]+(y-points[-(N_char-j), 6])/deriv_minus
            else:
                deriv_plus = np.tan(0.5*(points[-1, 0]+points[-1, 3]+phi+mu))
                xi, yi, xj, yj = points[-(N_char-j), 5], points[-(N_char-j), 6], points[-1, 5], points[-1, 6]
                x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 1]))
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-(N_char-j), 5], points[-(N_char-j), 6], x, y, 0]))
        # Add points to the final list
        points = np.vstack((points, np.array([[phi, nu, M, mu, p, x, y]])))
    if step == 0:
        points = points[1:]  # Remove dummy that was added at the start
        points_char = points_char[1:]
        lines_char = lines_char[1:]
    step += 1

# -------------------------------- SECOND WAVE INTERACTION ----------------------------------------------
# j are the incoming gamma+ characteristics, i are the outgoing gamma- characteristics.
for i in range(N_char):
    for j in range(N_char)[i:]:
        # Calculate the properties
        V_plus = fan[j, 1] - (-fan[j, 0])  # Characteristics come from the other symmetric side, where phi=-phi
        if i == j:  # These are the points on the jet boundary. This point is the start of a new j.
            nu = nu_2  # nu_2 is the value on the jet boundary, and is constant.
            phi = nu - V_plus
            V_minus = nu + phi
        else:
            phi = (V_minus-V_plus)/2
            nu = (V_minus+V_plus)/2
        M = nu_to_m(gamma, nu, M_1)
        mu = np.arcsin(1/M)
        p = m_to_p_isentropic(gamma, p_0, M)

        # Calculate the location
        if i == 0:  # first row of points uses the known characteristics
            idx = -1-sum(range(N_char-j))-j  # Index of the point from the previous non-simple region
            deriv_plus = np.tan(points[idx, 0]+points[idx, 3])  # phi and mu are the same on both points
            if j == 0:  # At jet boundary, no gamma- available. This if-condition corresponds to one point only.
                deriv_minus = np.tan(fan[-1, 0])  # Initial slope of jet boundary (call deriv_minus for convenience)
                xi, yi, xj, yj = 0, H, points[idx, 5], points[idx, 6]
            else:
                deriv_minus = np.tan(0.5*(points[-1, 0]-points[-1, 3]+phi-mu))
                xi, yi, xj, yj = points[-1, 5], points[-1, 6], points[idx, 5], points[idx, 6]
            x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
            if j == 0:  # jet boundary has a different colour
                if show_jb:
                    ax1.plot([0, x], [H, y], c='k', linestyle='--')
                # Add points on the characteristics
                points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, 0, H, points_char)
            else:
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 0]))
            # Add points on the characteristics
            points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, points[idx, 5], points[idx, 6], points_char)
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[idx, 5], points[idx, 6], x, y, 1]))
        else:  # all other points
            deriv_plus = np.tan(0.5*(points[-(N_char-i), 0]+points[-(N_char-i), 3]+phi+mu))
            if i == j:
                deriv_minus = np.tan(0.5*(points[-(N_char-i+1), 0]+phi))
                xi, yi, xj, yj = points[-(N_char-i+1), 5], points[-(N_char-i+1), 6], points[-(N_char-i), 5], points[-(N_char-i), 6]
            else:
                deriv_minus = np.tan(0.5*(points[-1, 0]-points[-1, 3]+phi-mu))
                xi, yi, xj, yj = points[-1, 5], points[-1, 6], points[-(N_char-i), 5], points[-(N_char-i), 6]
            x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
            if i == j and show_jb:  # Jet boundary
                ax1.plot([points[-(N_char-i+1), 5], x], [points[-(N_char-i+1), 6], y], c='k', linestyle='--')
            elif i != j:
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 0]))
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-(N_char-i), 5], points[-(N_char-i), 6], x, y, 1]))
        # Add points to the final list
        points = np.vstack((points, np.array([[phi, nu, M, mu, p, x, y]])))

        # Store the last point of the jet boundary. The value keeps overwriting until the required (last) point is reached.
        index_jb = len(points)-1

# -------------------------------- THIRD WAVE INTERACTION ----------------------------------------------
# i are the incoming gamma- characteristics, j are the outgoing gamma+ characteristics.
for j in range(N_char):
    for i in range(N_char)[j:]:
        # Calculate the properties
        idx = -1-sum(range(N_char-i))-i
        if j == 0:  # fan list cannot be used anymore, so use another method
            V_minus = points[idx, 1] + points[idx, 0]
        else:
            V_minus = points[-(N_char-j), 1]+points[-(N_char-j), 0]
        if i == j:  # These are the points on the symmetry axis. This point is the start of a new j.
            phi = 0
            nu = V_minus  # nu = V_minus(i)
            V_plus = nu - phi  # constant over the entire gamma+ characteristic, so only calculate when j changes
        else:  # Is guaranteed to happen after the previous if-statement has already happened, so V_plus is defined.
            phi = (V_minus-V_plus)/2
            nu = (V_minus+V_plus)/2
        M = nu_to_m(gamma, nu, M_1)
        mu = np.arcsin(1/M)
        p = m_to_p_isentropic(gamma, p_0, M)

        # Calculate the location
        if j == 0:  # first row of points uses the pre-defined characteristics
            deriv_minus = np.tan(points[idx, 0] - points[idx, 3])  # Take the gamma- characteristic
            if i == 0:  # at symmetry line, no gamma+ available. This if-condition corresponds to one point only.
                y = H/2
                x = points[idx, 5] + (y-points[idx, 6])/deriv_minus
            else:
                deriv_plus = np.tan(0.5*(points[-1, 0]+points[-1, 3]+phi+mu))
                xi, yi, xj, yj = points[idx, 5], points[idx, 6], points[-1, 5], points[-1, 6]
                x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 1]))
            # Add points on the characteristics
            points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, points[idx, 5], points[idx, 6], points_char)
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[idx, 5], points[idx, 6], x, y, 0]))
        else:  # the rest of the points uses not the pre-defined characteristics, but the previously calculated points
            deriv_minus = np.tan(0.5*(points[-(N_char-j), 0]-points[-(N_char-j), 3]+phi-mu))  # calculate the characteristic
            if i == j:  # at the symmetry line, no gamma+ available
                y = H/2
                x = points[-(N_char-j), 5]+(y-points[-(N_char-j), 6])/deriv_minus
            else:
                deriv_plus = np.tan(0.5*(points[-1, 0]+points[-1, 3]+phi+mu))
                xi, yi, xj, yj = points[-(N_char-j), 5], points[-(N_char-j), 6], points[-1, 5], points[-1, 6]
                x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
                lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 1]))
            lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-(N_char-j), 5], points[-(N_char-j), 6], x, y, 0]))
        # Add points to the final list
        points = np.vstack((points, np.array([[phi, nu, M, mu, p, x, y]])))

# -------------------------------- FOURTH WAVE INTERACTION ----------------------------------------------
# j are the incoming gamma+ characteristics, i are the outgoing gamma- characteristics.
i = 0  # Only the first row of points (i=0), since the rest is beyond the shock and unnecessary.
for j in range(N_char)[i:]:
    # Calculate the properties
    idx = -1-sum(range(N_char-j))-j
    V_plus = points[idx, 1] - points[idx, 0]

    if i == j:  # These are the points on the jet boundary. This point is the start of a new j.
        nu = nu_2  # nu_2 is the value on the jet boundary, and is constant.
        phi = nu - V_plus
        V_minus = nu + phi
    else:
        phi = (V_minus-V_plus)/2
        nu = (V_minus+V_plus)/2
    M = nu_to_m(gamma, nu, M_1)
    mu = np.arcsin(1/M)
    p = m_to_p_isentropic(gamma, p_0, M)

    # Calculate the location
    idx = -1-sum(range(N_char-j))-j  # Index of the point from the previous non-simple region
    deriv_plus = np.tan(points[idx, 0]+points[idx, 3])  # phi and mu are the same on both points
    if j == 0:  # At jet boundary, no gamma- available. This if-condition corresponds to one point only.
        deriv_minus = np.tan(points[index_jb, 0])  # index_jb is used here to indicate the required point at the jet boundary
        xi, yi, xj, yj = points[index_jb, 5], points[index_jb, 6], points[idx, 5], points[idx, 6]
    else:
        deriv_minus = np.tan(0.5*(points[-1, 0]-points[-1, 3]+phi-mu))
        xi, yi, xj, yj = points[-1, 5], points[-1, 6], points[idx, 5], points[idx, 6]
    x, y = calc_xy(deriv_minus, deriv_plus, xi, yi, xj, yj)
    if j == 0:  # jet boundary has a different colour
        if show_jb:
            ax1.plot([points[index_jb, 5], x], [points[index_jb, 6], y], c='k', linestyle='--')
        # Add points on the characteristics
        points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, points[index_jb, 5], points[index_jb, 6], points_char)
    else:
        lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[-1, 5], points[-1, 6], x, y, 0]))

    # Add points on the characteristics
    points_char = data_on_char(res, points, phi, nu, M, mu, p, x, y, points[idx, 5], points[idx, 6], points_char)
    lines_char = np.vstack((lines_char, [phi, nu, M, mu, p, points[idx, 5], points[idx, 6], x, y, 1]))

    # Add points to the final list
    points = np.vstack((points, np.array([[phi, nu, M, mu, p, x, y]])))

# ------------------------------------- COMPUTE HEATMAP ---------------------------------------
# Add the points of the beginning at x = 0, to fully define the initial condition for the heat map
points_begin = np.zeros((res, np.shape(points)[1]))
points_begin[:, :] = points_char[0]
points_begin[:, -1] = np.linspace(H/2, points_begin[0][-1], np.shape(points_begin)[0])

# Concatenate all defined points into x, y, and z. z can be changed to any desired flow property
x = np.concatenate((points[:, 5], points_begin[:, -2], points_char[:, -2]))
y = np.concatenate((points[:, 6], points_begin[:, -1], points_char[:, -1]))
z = np.concatenate((points[:, 2], points_begin[:, 2], points_char[:, 2]))

# -------------------------------- CALCULATE STREAMLINE -----------------------------------
x_init = 0
y_init = 0.75*H
streamline = np.array([points[0]])
streamline[0, 5] = x_init
streamline[0, 6] = y_init

# Propagate the streamline forward. x1 and y1 from the previous streamline point become x0 and y0 for the next.
"""A smaller step size yields more data points, but all data points in between characteristics are inaccurate, so
they do not yield extra information. Overshooting the intersection does not matter, since the new section starts
from the closes intersection point. Therefore, a slightly larger step size is chosen."""
delta = np.max(points[:, 5])/10  # Step size
lines_char_temp = np.array(lines_char)  # Do not use lines_char directly, since it is later used for other purposes
while streamline[-1, 5] < np.max(points[:, 5]):
    new_point = np.array(streamline[-1])  # It must become a numpy array, otherwise streamline itself will be changed
    new_point[5] += delta*np.cos(new_point[0])
    new_point[6] += delta*np.sin(new_point[0])
    # a and b are two points of the streamline and stay the same for one iteration, while the characteristics 'loop'
    # over all that are defined.
    a = np.array([[streamline[-1, 5], streamline[-1, 6]]]*np.shape(lines_char_temp)[0])  # [[x0, y0], [x0, y0], ...]
    b = np.array([[new_point[5], new_point[6]]]*np.shape(lines_char_temp)[0])  # [[x0, y0], [x0, y0], ...]
    c = lines_char_temp[:, 5:7]  # [[x0, y0], [x0, y0], ...]
    d = lines_char_temp[:, 7:9]  # [[x1, y1], [x1, y1], ...]
    bool_intersections = intersect(a, b, c, d)  # Boolean matrix
    idx_intersections = np.array(np.where(bool_intersections == True)).flatten()
    intersections = lines_char_temp[bool_intersections]  # Array of characteristics that intersect, containing all flow information

    if np.any(bool_intersections):  # If there are any intersections, we update the values on the new point
        x0, y0, x1, y1 = intersections[:, 5:9].T
        x_temp, y_temp = calc_xy((y1-y0)/(x1-x0), new_point[0], x0, y0, a[0, 0], a[0, 1])
        distance = np.sqrt(x_temp**2+y_temp**2)
        idx_min = np.argmin(distance)  # Index of the point that is the closest to the streamline
        new_point = intersections[idx_min][:-3]  # Cut the last three elements
        new_point[5], new_point[6] = x_temp[idx_min], y_temp[idx_min]
        lines_char_temp = np.delete(lines_char_temp, idx_intersections[idx_min], axis=0)
    streamline = np.vstack((streamline, new_point))

# ------------------------------- CALCULATE SHOCK LOCATION ------------------------------------
gamma_plus = lines_char[lines_char[:, 9] == 1]
gamma_minus = lines_char[lines_char[:, 9] == 0]
intersections = np.array([[np.nan]*7])
for w in [gamma_plus, gamma_minus]:  # Check whether characteristics of the same family intersect
    for char in range(len(w)):
        a = np.array([[w[char, 5], w[char, 6]]] * np.shape(w)[0])
        b = np.array([[w[char, 7], w[char, 8]]] * np.shape(w)[0])
        c = w[:, 5:7]
        d = w[:, 7:9]

        bool_intersections = intersect(a, b, c, d)  # Boolean matrix
        idx_intersections = np.array(np.where(bool_intersections == True)).flatten()
        intersecs = w[bool_intersections]
        if np.any(intersecs):
            x0, y0, x1, y1 = intersecs[:, 5:9].T
            x_temp, y_temp = calc_xy((y1 - y0) / (x1 - x0), (w[char, 8]-w[char, 6])/(w[char, 7]-w[char, 5]), x0, y0, a[0, 0], a[0, 1])
            distance = np.sqrt(x_temp ** 2 + y_temp ** 2)
            idx_min = np.argmin(distance)  # Index of the point that is the closest to the streamline
            new_point = intersecs[idx_min][:-3]  # Cut the last three elements
            new_point[5], new_point[6] = x_temp[idx_min], y_temp[idx_min]
            intersections = np.vstack((intersections, new_point))

intersections = intersections[1:]
intersecs_xy = intersections[:, 5:]
gamma_xy = np.vstack((gamma_plus[:, 5:7], gamma_plus[:, 7:9], gamma_minus[:, 5:7], gamma_minus[:, 7:9]))
indices = []
for xy in range(len(intersecs_xy)):
    xy_array = np.array([intersecs_xy[xy]]*len(gamma_xy))
    # Check whether the current intsersection point has any overlapping coordinates with the characteristics.
    # Overlapping coordinates mean that the line didn't intersect somewhere in the middle, but exactly at the endpoint.
    # When it overlaps exactly at the endpoint, it means that it is (probably) not a true intersection.
    index = np.array(np.any(np.logical_and(np.abs(xy_array[:, 0] - gamma_xy[:, 0]) < 1e-10, np.abs(xy_array[:, 1] - gamma_xy[:, 1]) < 1e-10))).flatten()
    if index:
        indices.append(xy)
intersections = np.delete(intersections, indices, axis=0)
first_shock_idx = np.argmin(intersections[:, 5])  # Separate the first shock from others, such that they can be displayed differently
first_shock = np.array(intersections[first_shock_idx])

# ------------------------------------ PLOTTING ------------------------------------
if show_interpolation:
    plt.tricontourf(x, y, z, levels=100, cmap='viridis')  # Interpolated values used as a background map

if show_points:
    plt.scatter(points[:, 5], points[:, 6], c=points[:, 2], s=20, zorder=30)  # All intersections of characteristics

if show_extra_points:
    plt.scatter(points_char[:, -2], points_char[:, -1], c=points_char[:, 2], s=20, zorder=30)  # All extra points

if show_char:
    ax1.plot([lines_char[:, 5], lines_char[:, 7]], [lines_char[:, 6], lines_char[:, 8]], c='dimgray', linewidth=1.5)

if show_sym:
    ax1.hlines(H/2, xmin=-0.5, xmax=np.max(points[:, 5])+0.5, linestyles='dashdot', colors='k')

if show_stream:
    ax1.plot(streamline[:, 5], streamline[:, 6], c='r', label='Streamline')

if show_shock:
    ax1.scatter(first_shock[5], first_shock[6], c='r', marker='*', s=200, zorder=21)

if show_text_regions:
    ax1.scatter([0, 1.2], [0.75, 1], c='pink', s=200)
    ax1.text(-0.08, 0.65, r'1')
    ax1.text(1.12, 0.9, r'2')
    ax1.arrow(-1, 0.75, 0.5, 0, width=0.05, color='lightseagreen')

if show_text_points1:  # Shows the numbering for 4 characteristics in non-simple region 1
    pos = np.array([[0.65, 0.75, 0.87, 1.00, 1.25, 1.45, 1.72, 2.05],
                    [0.55, 0.60, 0.65, 0.72, 0.80, 0.73, 0.66, 0.55]])  # [[x0, x1, ...], [y0, y1, ...]]
    txts = np.array(['i=0', 'i=1', 'i=2', 'i=3', 'j=0', 'j=1', 'j=2', 'j=3'])  # Texts to show
    for txt in range(8):
        ax1.text(pos[0, txt], pos[1, txt], txts[txt])

# Next text not plotted simultaneously with the previous text, because they will overlap
if show_text_points2:  # Shows the numbering for 4 characteristics in non-simple region 2
    pos = np.array([[3.67, 3.97, 4.33, 4.85, 1.82, 2.18, 2.57, 3.07],
                    [0.87, 0.97, 1.08, 1.20, 1.28, 1.21, 1.10, 0.95]])  # [[x0, x1, ...], [y0, y1, ...]]
    txts = np.array(['i=0', 'i=1', 'i=2', 'i=3', 'j=0', 'j=1', 'j=2', 'j=3'])  # Texts to show
    for txt in range(8):
        ax1.text(pos[0, txt], pos[1, txt], txts[txt])

# Show the colour bar when necessary
if show_extra_points or show_points or show_interpolation:
    cbar = plt.colorbar(pad=0.02)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(f'Mach number $M$', size=15)

# Pressure plot
fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.plot(streamline[:, 5], streamline[:, 4])
if show_shock:
    ax2.vlines(first_shock[5], np.min(streamline[:, 4]), np.max(streamline[:, 4]), colors='r', linestyles='--')

ax1.set_xlabel(r'$x$-position [m]', fontsize=15)
ax1.set_ylabel(r'$y$-position [m]', fontsize=15)
ax1.axis('equal')
ax1.xaxis.set_tick_params(labelsize=15)
ax1.yaxis.set_tick_params(labelsize=15)
fig1.tight_layout()

ax2.set_xlabel(r'$x$-position [m]', fontsize=15)
ax2.set_ylabel(r'Static pressure $p$ [atm]', fontsize=15)
ax2.xaxis.set_tick_params(labelsize=15)
ax2.yaxis.set_tick_params(labelsize=15)
fig2.tight_layout()

plt.show()
