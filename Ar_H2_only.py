import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import interp1d
import scipy.constants as co
from bolos import solver, grid, parser
''' Mole Fractions'''
p = 2.0                               # Pressure [N/m2]
T = 300;                              # Temperature [K]
R = 8.314;                            # Universal Gas Constant [J/mol*K] 
N = 6.0221E23                         # Avagadro's number 
nf = (p/(R * T)) * N                  # plasma density  
''' MATLAB code completed under following conditions: p: 14 [Pa], rp: 7 [nm]'''
''' ELastic and Attachment Terms by the nanoparticles'''
##############################################################################
m_Ar = 6.6635209E-26                  # Mass of Argon [kg] 
m_H2 = 1.6737236E-27                  # Mass of Hydrogen [kg]
m_dom = 3 * m_H2                      # Mass of the Dominant Ion H3+ [kg]
#mi = (19.7/20)*(6.6335209E-26) + (0.3/20)*(1.6737236E-27) 
RA= 9E-2;                             # Radius of the Plasma [m]
d= 10.16E-2;                          # Distance between two plates [m]
Ae= 2 * math.pi * RA * d;             # Area Bounding the Plasma Volume [m^2]
V= (RA ** 2) * math.pi * d;           # Volume of the plasma region [m^3]
#m_particles = 2250 * ((4/3)*math.pi*(rp**3))# Mass per particle [kg]
Te =  3.095785236420782                              # Electron Temperature [eV] @ np: 1.0E15 [1/m3]
rp = 6.5E-9                             # radius of carbon nanoparticles
m_particles = 2266 * ((4/3)*math.pi*(rp**3))
eo = 8.8541878176E-12                 # Permittivity of Free Space [m^-3*kg^-1*s^4*A^2]
Ti = 300                              # Ion Temperature [K]
Te_K = 11604.52500617 * Te            # Electron Temperature [K]
ne = np.array([1.77E15])                 # Electron Density [1/m3]
ni = np.array([1.56E16])                 # Ion Density [1/m3]
e= 1.6021765E-19                      # Elementary Electron Charge [C]
kb= 1.38064853E-23                    # Boltzmann Constant [J/K]
me= 9.10938E-31                       # Mass of electron [kg]
gamma = ((2 * e)/me) ** 0.5           # Constant 
# Mass of ions [kg]
mi = (19.3/20)*(6.6335209E-26) + (0.7/20)*(1.6737236E-27)  

##############################################################################
''' Elastic Cross Section [m2]: Carbon Nanoparticles ''' 
'''####################################################################### '''
############################################################################'''
energy = np.linspace(0,30,3000)   
######################### Secondary Electron Emission ######################'''
xnergy = np.linspace(0,30,3000)    
###############################################################################
''' ######### SEM Cross Sections for Attachment and Excitation [m2]#########'''
###############################################################################
gs = grid.LinearGrid(0,30,3000)
titan = solver.BoltzmannSolver(gs)
''' ######### SEM Cross Sections for Attachment and Excitation [m2]#########'''     
###############################################################################
''' Open up the txt file and load processes and collisions'''
with open('Ar_H2_cs.txt') as fp:
    processes = parser.parse(fp)
    titan.load_collisions(processes)
    titan.target['Ar'].density = (19.3/20) * (1)
    titan.target['H2'].density = (0.7/20) * (1)
    titan.kT = 300 * co.k / co.eV
    titan.EN = 70 * solver.TOWNSEND
    titan.init()
    z0 = titan.maxwell(Te)
    z1 = titan.converge(z0, maxn=100, rtol=1e-7)

TeTBD = np.trapz(((xnergy ** 1.5)*z1), xnergy)
print('Electron Temperature ', np.multiply(0.66666666666666666666666,TeTBD))  
#np.savetxt('Ar_H2_only_70Td.txt', zip(xnergy,z1), delimiter=',', fmt='%.15f')
plt.plot(energy, z1)
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(8.5, 8.5)
plt.yscale('log')
plt.ylim(10 ** -2.0, 10 ** 0)
plt.xlim(-1,12)
plt.rcParams.update({'font.size': 9})





   