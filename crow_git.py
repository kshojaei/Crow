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
Te = 4.3                              # Electron Temperature [eV] @ np: 1.0E15 [1/m3]
rp = 6.5E-9                           # radius of carbon nanoparticles
m_particles = 2266 * ((4/3)*math.pi*(rp**3))
eo = 8.8541878176E-12                 # Permittivity of Free Space [m^-3*kg^-1*s^4*A^2]
Ti = 300                              # Ion Temperature [K]
Te_K = 11604.52500617 * Te            # Electron Temperature [K]
ne = np.array([5.77E15])              # Electron Density [1/m3]
ni = np.array([1.56E16])              # Ion Density [1/m3]
e= 1.6021765E-19                      # Elementary Electron Charge [C]
kb= 1.38064853E-23                    # Boltzmann Constant [J/K]
me= 9.10938E-31                       # Mass of electron [kg]
gamma = ((2 * e)/me) ** 0.5           # Constant 
# Mass of ions [kg]
mi = (19.3/20)*(6.6335209E-26) + (0.7/20)*(1.6737236E-27)  
# Electron Debye Length [m]
lam_e = ((eo * kb * Te_K)/(e * e * ne)) ** 0.5
# Ion Debye Length [m]
lam_i = ((eo * kb * Ti)/(e * e * ni)) ** 0.5
# Linearized Debye Length [m]
Tp = 300
##############################################################################
''' Elastic Cross Section [m2]: Carbon Nanoparticles '''
def elas(Vp, lam_f, x):
    result = list()
    for i in x:
        if i > 0:
            a =  (2 * np.pi * (rp ** 2) * ((Vp/(2*i)) ** 2))
            b = (((lam_f) ** 2) + (((Vp/(2*i)) ** 2) * (rp ** 2)))
            c = (rp ** 2) * ((((((Vp/(2 * i)) ** 2)))))
            result.append(a * np.log(b/c) + (math.pi * ((rp) ** 2) * (1 + (Vp/(i)))))
        else:
            s = 0.01
            d =  (2 * np.pi * (rp ** 2) * ((Vp/(2*s)) ** 2))
            e = (((lam_f) ** 2) + (((Vp/(2*s)) ** 2) * (rp ** 2)))
            f = (rp ** 2) * ((((((Vp/(2 * s)) ** 2)))))
            result.append(d * np.log(e/f)+ (math.pi * ((rp) ** 2) * (1 + (Vp/(s))))) 
    return result
''' Attachment (inelastic) Cross Section [m2]: Carbon Nanoparticles'''
def attach(Vp, x):
    result = list()
    for i in x:
        if i > -Vp:
            result.append((math.pi * ((rp) ** 2) * (1 + (Vp/(i)))))
        else:
            result.append(0)
    return result  
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
def SEM_DETACH(Vp, x):
    result = list()
    for i in x:
        if (-1*Vp + 0.01) >= i >= (-1*Vp - 0.01):
            result.append((-2.4E-15 * (1/((2 * math.pi * 0.001)) ** 0.5) * 
                           np.exp(-((i + 1 *Vp) ** 2)/0.002)))
        else:
            result.append(0)
    return result      
###############################################################################
''' Open up the txt file and load processes and collisions'''
with open('Ar_H2_cs.txt') as fp:
    processes = parser.parse(fp)
    titan.load_collisions(processes)
for Vpp in np.linspace(-0.5,-15):
    for npp in np.linspace(1E12, 1E16):
        Vp = -2.5502408979243154         # Particle potential @ np: 1.0E15 [1/m3]
        t = np.array([853907284136384.0])             # particle density [1/m3]
        x = t/(nf+t)                       # mole fraction
        kk = (ni-ne)/(t[0])
        lam_p = ((eo * kb * Tp)/(kk * e * e * t)) ** 0.5
        lam_f = ((1/lam_e)+(1/lam_i)+(1/lam_p))**(-1)
        #####################################################
        '''Load elastic and attachment cross sections of nanoparticles @ np: 1.0E15[1/m3]'''
        titan.add_process(kind = "EFFECTIVE", target="Karbon", mass_ratio = (me/(m_particles)), 
                          data=np.c_[energy, elas(Vp,lam_f[0], energy)])
        titan.add_process(kind = "ATTACHMENT", target="Karbon", 
                          data= np.c_[energy, attach(Vp,energy)])
        titan.add_process(kind= "ATTACHMENT", target="Karbon", 
                          data= np.c_[xnergy, SEM_DETACH(Vp, xnergy)])
        ###############################################################################
        titan.target['Ar'].density = (19.3/20) * (1- x[0])
        titan.target['H2'].density = (0.7/20) * (1-x[0])
        titan.target['Karbon'].density = x[0]
        titan.kT = 300 * co.k / co.eV
        titan.EN = 70 * solver.TOWNSEND
        titan.init()
        z0 = titan.maxwell(Te)
        z1 = titan.converge(z0, maxn=100, rtol=1e-7)
        ##############################################################################
        ##############################################################################
        z_norm  = np.trapz(z1,xnergy)
        z2 = z1 / z_norm
        freq_detach = gamma * x[0] * xnergy * z1 * (SEM_DETACH(Vp, xnergy))
        freq_attach = gamma * x[0] * xnergy * z1 * attach(Vp, xnergy) 
        vth  = ((e*xnergy)**0.5) * ((2/me)**0.5)
        area_detach = np.trapz(freq_detach,xnergy)
        area_attach = np.trapz(freq_attach, xnergy)
        hulk = area_detach/(-area_attach)
        print('normalized secondary yield', hulk)
        ''' ################################# OML theory ########################''' 
        S = 4 * math.pi * (rp ** 2)
        A1 = S * ni[0] * (((kb*Ti)/(2* math.pi * m_dom))** 0.5)
        area_e = ((area_attach+area_detach)/x[0]) * ne[0]
        Vpp = (1-(area_e/A1))*(kb*Ti / e)
        npp = (ne[0] - ni[0]) / ((4 * Vpp * math.pi * eo * rp)/(e))
        print('Vp and np',Vpp,npp)
        TeTBD = np.trapz(((xnergy ** 1.5)*z1), xnergy)
        print('Electron Temperature ', np.multiply(0.66666666666666666666666,TeTBD))        
        ''' Save the Normalized EEPF'''
        eqn11 = ne[0] - ni[0] - ((4 * Vpp * math.pi * eo * rp)/(e)) * npp
        eqn22 = A1 * (1 - ((e * Vpp)/(kb * Ti))) - (((area_attach+area_detach)/x[0]) * ne[0])
        ii = A1 * (1 - ((e * Vp)/(kb * Ti)))
        ed = ((-area_detach)/x[0]) * ne[0]
        ea = ((area_attach)/x[0]) * ne[0]
        print('Sec yield ed/ii', ed/ii)
        print('Sec yield ed/ea', ed/ea)
        print(eqn11,eqn22)
        print(np.absolute((Vp - Vpp)/Vp))
        Vp = (Vp + Vpp) / 2
        t = npp 
        if np.absolute((Vp - Vpp)/Vp) > 0.02:
            continue
        else:
            Vp = Vpp
            t = npp
            print('real Vp', Vpp)
            print('real npp', round(npp))
            print(round(npp))
            plt.plot(titan.grid.c, z1)
            TeTBD = np.trapz(((xnergy ** 1.5)*z1), xnergy)
            print('Electron Temperature ', np.multiply(0.66666666666666666666666,TeTBD))        
            ''' Save the Normalized EEPF'''
            print('Sec yield ed/ii', ed/ii)
            print('Sec yield ed/ea', ed/ea)
            #np.savetxt('photoemission_se_0.4.txt', zip(xnergy,z1), delimiter=',', fmt='%.15f')
            fig = plt.gcf()
            plt.grid(True)
            fig.set_size_inches(8.5, 8.5)
            plt.yscale('log')
            plt.ylim(10 ** -2.0, 10 ** 0)
            plt.xlim(-1,12)
            plt.rcParams.update({'font.size': 9})
            break 
        break
    break        

