from sympy import *
init_printing(use_latex=True)
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Circle


#SYMBOLIC EXPRESSION

t,r,theta,phi = symbols('t r theta phi', cls = Function)
td,rd,thetad,phid = symbols('td rd thetad phid', cls = Function)
lam = symbols('lam')
m = symbols('m', positive = True)

t = t(lam)
r = r(lam)
theta = theta(lam)
phi = phi(lam)

td = td(lam)
rd = rd(lam)
thetad = thetad(lam)
phid = phid(lam)

tdd = -(m/(r*(r - 2*m)) * td * rd)
rdd = -(m*(r-2*m)/r**3*td**2 - m/(r*(r-2*m))*rd**2 - 
          (r-2*m)*thetad**2 - (r-2*m)*sin(theta)**2*phid**2)
thetadd = -(2/r*thetad*rd - sin(2*theta)/2*phid**2)
phidd =  -(2/r*rd*phid + 2*cot(theta)*thetad*phid)


#DEFINING THE METRIC
m_val = 5

metric_t = -(1-2*m/r)
metric_r = 1/(1-2*m/r)
metric_theta = r**2
metric_phi = r**2*sin(theta)**2

metric = np.array([metric_t, metric_r, 
          metric_theta, metric_phi])


#TRANSLATING SYMBOLIC TO NUMERICAL
dd_m_sub = [e.subs(m,m_val) for e in [tdd,rdd,thetadd,phidd]]
g_m_sub = [e.subs(m,m_val) for e in metric]

f_dd = lambdify(
    (t,r,theta,phi,td,rd,thetad,phid),
    dd_m_sub,
    'numpy'
)

metric_lam = lambdify((r, theta), g_m_sub, 'numpy')

def null_curve(r_,th_, ders):
    g = np.array(metric_lam(r_,th_))
    v = np.array(ders)
    dt = np.sqrt(-(np.sum(g[1:]*v**2)/g[0]))
    return dt

def geod(lam,coord):
    [t,r,theta,phi] = coord[:4]
    v = coord[4:]
    return [ *v, *f_dd(*coord)]

def horizon(lam, y):
    return y[1] - 2*m_val*1.001

#Plotting
def plotter_3d(geos):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    u,v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    R = 2*m_val
    ax.plot_surface(
        R*np.cos(u)*np.sin(v),
        R*np.sin(u)*np.sin(v),
        R*np.cos(v),
        color = 'grey'
    )

    for j in geos:
        change_idx = np.where(j[2][:-1] * j[2][1:] < 0)[0]
        if  len(change_idx)>2:
            ax.plot(j[0],j[2],j[1], color = 'red') #photon ring
        else:
            if len(change_idx) == 2:
               ax.plot(j[0],j[2],j[1], color = 'orange') #lensing ring
            else:  
                ax.plot(j[0],j[2],j[1], color = 'lightblue') #direct

    """
    for j in geos:
        b = np.linalg.norm([j[0][0],j[1][0]])
        if  5.19*m_val < b  and b < 5.23*m_val:
            ax.plot(j[1],j[2],j[0], color = 'red')
        else:
            if 5.02*m_val < b  and b < 6.17*m_val:
                ax.plot(j[1],j[2],j[0], color = 'orange')
            else:
                if b <= 5.02*m_val:
                    ax.plot(j[1],j[2],j[0], color = 'lightgreen')
                else:  ax.plot(j[1],j[2],j[0], color = 'lightblue')
    """

    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    ax.set_zlim(-30,30)

    ax.set_box_aspect((1,1,1))

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    return