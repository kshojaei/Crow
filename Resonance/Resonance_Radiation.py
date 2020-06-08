import numpy as np
import scipy as sp
from scipy.special import wofz
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d
import scipy.constants as co
# Analyzing Resonance Radiation 
p = 2.0                          # Pressure [N/m2]
eo = 8.8541878176E-12                    # Permittivity of Free Space [m^-3*kg^-1*s^4*A^2]
me= 9.10938E-31                  # Mass of electron [kg]
e= 1.6021765E-19                 # Elementary Electron Charge [C]
m_Ar = 6.6635209E-26             # Mass of Argon [kg] 
m_H2 = 3 * 1.6737236E-27             # Mass of Hydrogen [kg]
c = 299792458                    # Vacuum Velocity of light [m/s]
N_av = 6.0221E23                 # Avagadro's number [# atoms / mole]
M_Ar = 39.948E-3                 # Atomic Weight of Argon  [kg/mol]
M_H = 1.00794E-3                 # Atomic Weight of Hydrogen [kg/mol]
kb= 1.38064853E-23               # Boltzmann Constant [J/K]
T_Ar, T_H = 300,300              # Absolute Temperature of Ar and H2 [K]
R = 8.314                        # Universal Gas Constant Per mole [J/mol*K]
Q = 1                         # Quantum Yield 0.25-1.0 electron/ incident photon
rp = 15E-9                        # radius of carbon nanoparticles [m]
ne = np.array([275462801580917.97])         # Electron Density [1/m3]
ni = np.array([1.387e+16])         # Ion Density [1/m3]
Vp = -2.334376633908626                    # Particle potential
Te = 4.707                        # Electron Temperature [K]
Ti = 300                         # Ion Temperature [K]
Te_K = 11604.52500617 * Te       # Electron Temperature [K]
C = 0.73                         # Constant weakly dep(plasma properties, particle size)
A = 4.8                          # Electorn Affinity of Bulk Carbon [eV]
hh = 6.62607004E-34              # Planck's constant m2 kg / s
S = (4 * np.pi * (rp**2))        # Particle Surface Area
Coef = 0.5 * 1.20173E6;          # Richardson Constant 
Tp = 580                        # Particle Temperature [K]
##################################
vo_Ar = np.sqrt(2*R*T_Ar / M_Ar) # Average Argon Velocity [m/s]
vo_H = np.sqrt(2*R*T_H / M_H)    # Average Hydrogen Velocity [m/s]
N = (p/(R * T_Ar)) * N_av        # Gas Density  [1/m3]
##################################
# Resonance Line: The line of longest wavelength associated with a transition
# between the ground state and an excited state
lmda_1 = 106.7E-9                # Resonance line (3P1) excited argon [m]
lmda_2 = 104.8E-9                # Resonance line (1P1) excited argon [m]
##################################
gama_1 = 1.32E8                  # Transition Probability (3P1) [1/s]
gama_2 = 5.32E8                   # Transition Probability (1P1) [1/s]
##################################
vo_1 = c / lmda_1                # frequency at which k(v) is max (3P1) Ex Argon
vo_2 = c / lmda_2                # frequency at which k(v) is max (1P1) Ex Argon
##################################
tau_1 = 1 / gama_1               # Lifetime of the Ex Argon (3P1) [s]
tau_2 = 1 / gama_2               # Lifetime of the Ex Argon (1P1) [s]
##################################
g_1 = 3                          # Degeneracy (3P1) Ex Argon (2L+1)(2S+1)
g_2 = 3                          # Degeneracy (1P1) Ex Argon 
g_0 = 1                          # Degeneracy Argon Normal State (1S0)
# Spectroscopic Notation: (2S+1)LJ
k_01 = (((lmda_1 **2)* N)/(8*np.pi))*(g_1/g_0)*(1/tau_1)
k_02 = (((lmda_2 **2)* N)/(8*np.pi))*(g_2/g_0)*(1/tau_2)
##################################
f_1 = 0.0675                       # Oscillator Strength for 106.7 nm (3P1) Ex Argon
f_2 = 0.2629                       # Oscillator Strength for 104.8-9 nm (1P1) Ex Argon
#################################
# Average collision rate for the Ex Argon (3P1) [1/s]
gama_c_1 = (4 * co.e * co.e * f_1 * lmda_1 * N)/(3 * m_Ar * c)
# Average collision rate for the Ex Argon (1P1) [1/s]
gama_c_2 = (4 * co.e * co.e * f_2 * lmda_2 * N)/(3 * m_Ar * c)
##################################
a_1  = ((gama_1 + gama_c_1) * lmda_1) / (4 * np.pi * vo_Ar)
a_2  = ((gama_2 + gama_c_2) * lmda_2) / (4 * np.pi * vo_Ar)
##################################
# Impressed Frequency [1/s]
v_1 = np.linspace(vo_1 -1E11, vo_1 + 1E11,3000)
v_2 = np.linspace(vo_2 -1E11, vo_2 + 1E11,3000)
##################################
x_1 = (v_1 - vo_1)*(lmda_1 / vo_Ar)
x_2 = (v_2 - vo_2)*(lmda_2 / vo_Ar)
##################################
def Voig_1(u,y):
    top =  np.exp(-(y**2))
    bottom = a_1**2 + (u - y)**2
    return top/bottom 
def Voig_2(u,y):
    top =  np.exp(-(y**2))
    bottom = a_2**2 + (u - y)**2
    return top/bottom 
# Voigt Profile [3P1] Ex Argon
def Voigt_1(u):
    res = np.zeros_like(u)
    for i,val in enumerate(u):
        y,err = integrate.quad(Voig_1, -np.inf, np.inf, args=(val))
        res[i] = y
    return (a_1/np.pi) * res
# Voigt Profile [1p1] Ex Argon
def Voigt_2(u):
    res = np.zeros_like(u)
    for i,val in enumerate(u):
        y,err = integrate.quad(Voig_2, -np.inf, np.inf, args=(val))
        res[i] = y 
    return (a_2/np.pi) * res
##################################
Voigt_1_area = np.trapz(Voigt_1(x_1),v_1)
# Normalized Voigt_1 Function [3P1] Ex Argon
Voigt_2_area = np.trapz(Voigt_2(x_2),v_2)
# Normalized Voigt_1 Function [1P1] Ex Argon
Voigt_1_norm = Voigt_1(x_1) / Voigt_1_area
Voigt_2_norm = Voigt_2(x_2) / Voigt_2_area

#plt.plot(v_1,Voigt_1_norm)

# Absorption Coefficients accounting for natural, Doppler, and 
# pressure broadenings (3P1) Ex Argon (KP(v))
kv_1 = k_01 * Voigt_1_norm   
# Absorption Coefficients accounting for natural, Doppler, and 
# pressure broadenings (1P1) Ex Argon (KP(v))
kv_2 = k_02 * Voigt_2_norm  
###################################
# Probability of a quantum penentrating a distance rho in the gas 
# before being absorbed for constant rho 
def T_1(rho):
    return np.trapz((Voigt_1_norm * np.exp(-kv_1 * rho)),v_1)
# Probability of a quantum penentrating a distance rho in the gas 
# before being absorbed for constant rho 
def T_2(rho):
    return np.trapz((Voigt_2_norm * np.exp(-kv_2 * rho)),v_2)
# Probability of quantum penentrating a distance rho in a gas
# before being absorbed for series of rho
def T_11(x):
    result = list()
    for i in x: 
        if i >= 0:
            result.append(np.trapz((Voigt_1_norm * np.exp(-kv_1 * i)),v_1))
        else:
            result.append(0)
    return result
# Probability of quantum penetrating a distance rho in a gas
# before being absorbed for series of rho
def T_22(x):
    result = list()
    for i in x: 
        if i >= 0:
            result.append(np.trapz((Voigt_2_norm * np.exp(-kv_2 * i)),v_2))
        else:
            result.append(0)
    return result

#plt.plot(radial, T_11(radial))

RR = 0.3                           # Maximum Discharge Dimension [m]
radial = np.linspace(0, RR, 10000)
plt.plot(radial, T_11(radial))
plt.plot(radial, T_22(radial))
plt.yscale('log')
plt.xscale('log')
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(8.5, 8.5)
plt.rcParams.update({'font.size': 9})
''' ######################      Argon Excitation Frequency ################ '''
import scipy.constants as co
from bolos import solver, grid, parser
'''########################## Argon  Excitation Cross Section #########################'''
i = np.linspace(0,30,27000)
data = np.loadtxt('/Users/scienceman/Desktop/Langmuir/CrowMaster/Cross_Sections/Ar-03.txt')
d1 = data.T[0]
d2 = data.T[1]
Q_Ar_ex = np.interp(i,d1,d2)
'''  ######################## EEPF [eV^-1.5] ############################## '''
data1 = np.loadtxt('/Users/scienceman/Desktop/Langmuir/CrowMaster/CrowCode/SpiderMan/Figures/EEPF_1E15_resonance.txt', delimiter=',')
EEPF = data1.T[:]
#d22 = data1.T[1]
#EEPF = np.interp(i,d11,d22)
''' ######################### Argon Excitation Frequnecy ################# '''
xnergy = np.linspace(0,30,27000)
EEDF = EEPF * ((1*xnergy)**+0.5) 
EEDF_area = np.trapz(EEDF,xnergy)
EEDF_norm = EEDF / EEDF_area
vth  = ((co.e*xnergy)**0.5) * ((2/me)**0.5)
vexx = (20/20) * N * Q_Ar_ex * vth * EEDF_norm 
vex_Ar = np.trapz(vexx,xnergy)
#plt.plot(xnergy, EEPF)
#plt.yscale('log')
#print(xnergy, EEPF)
''' ######################## Ref 57 Eescape Factors ########################'''
g_escape_1 = 6E-2/5                  # Escape Factor (3P1) Ex Argon lmda_1: 106.7E-9 nm
g_escape_2 = 2E-3/5                  # Escape Factor (1P1) Ex Argon lmda_2: 104.8E-9 nm
''' ########## Population Density of Ex Argon (3P1) & (1P1) ############### '''
n_1 = (ne[0] * vex_Ar) / (g_escape_1 * gama_1) 
n_2 = (ne[0] * vex_Ar) / (g_escape_2 * gama_2) 
#print(round(n_1))
print((np.trapz(T_11(radial),radial)))
print((np.trapz(T_22(radial),radial)))
''' ######  Charging Frequency by resonance UV photodetachment ############ '''
freq_g_1 = Q * np.pi * (rp**2) * n_1 * gama_1 * (np.trapz(T_11(radial),radial))
freq_g_2 = Q * np.pi * (rp**2) * n_2 * gama_2 * (np.trapz(T_22(radial),radial))
freq_total = freq_g_1 + freq_g_2
print('UV Resonance Photodetachment',freq_total)
''' ################ Charging Frequency Due to Quenching Process ##########'''
freq_quench = (n_1 + n_2) * np.sqrt((kb*T_Ar)/m_Ar) * np.pi * (rp ** 2)
print('Quenching Frequency', freq_quench)
''' ################# Electron & Ion Frequency OML Theory ################# '''
# Ion Frequency [1/s]
def freq_i(k):
    S = 4 * np.pi * (rp**2)
    Vp = ((-k * co.e) / ( 4 * np.pi * eo * rp))
    A = S * ni[0] * (((kb*T_Ar)/(2* np.pi * m_Ar))** 0.5)
    nu = A * (1 - ((co.e * Vp)/(kb * T_Ar)))
    return nu
print(freq_i(12))
print(freq_total/ freq_i(12))
# Electron Frequency
def freq_e(k):
    S =  4 * np.pi * (rp**2)
    Vp = ((k * co.e) / ( 4 * np.pi * eo * rp))    
    A_e = S * ne[0] * (((kb*Te*11604.52500617)/(2* np.pi * me))** 0.5) 
    nue = A_e * np.exp((co.e*Vp)/(kb*Te*11604.52500617))
    return nue 


def fre_se(k):
    S =  4 * np.pi * (rp**2)
    Vp = ((k * co.e) / ( 4 * np.pi * eo * rp))    
    A_e = S * ne[0] * (((kb*Te*11604.52500617)/(2* np.pi * me))** 0.5) 
    nue = A_e * np.exp((co.e*Vp)/(kb*Te*11604.52500617))
    
# Electron Affinity of charged particles as a function of kk and rp 
''' ########################  Electron Affinity ############################'''
def A_u(k,r): 
    a = (5*co.e)/(8 * np.pi * eo * r)
    b = ((np.absolute(k)-1) * co.e) / (4 * np.pi * eo * r)
    return A - a - b
def W(k,r):
    return A_u(k,r)
# Thermionic Emission Frequency [1/s]  
def freq_therm(k,r,T):
    WW = e * W(k,r)
    return Coef * (T **2) * S * (1/e) * np.exp(-WW / (kb * T))
''' ##################### Particle Charge Distribution  ################## '''
'''
def F(k):
    if k == 0 :
        return 1
    elif k < 0:
        VENOM = (freq_i(k) + freq_g_1 + freq_g_2 + freq_quench + freq_therm(k,rp,Tp)) / (freq_e(k+1))
        return (1/VENOM) * F(k + 1)
    else:
        VENOM = (freq_i(k) + freq_g_1 + freq_g_2 + freq_quench + freq_therm(k,rp,Tp)) / (freq_e(k+1))
        return (VENOM) * F(k - 1)        
def FF(k):
    result = list()
    for i in k:
        result.append(F(i))
    return result
charge  = np.linspace(-20,5,26)
E = np.linspace(-10,-4,13)
#print(E)
F_area = np.trapz(FF(charge), charge)
F_norm = FF(charge) / F_area
pot = ((-4 * e) / ( 4 * np.pi * eo * rp))    
plt.plot(charge, F_norm)
#print(pot)
char = ( 4 * np.pi * eo * rp * -1.48) / e
#print(char)
'''
''' ############ Particle Balance and Charge Balance OML #################'''
'''
from scipy.optimize import fsolve
def equations(p):
    m, l = p
    # Particle Surface Area [m2]
    S= 4 * sp.pi * (rp**2)                
    # Electron Frequency OML [1/s]         
    ve= (((kb * m * 11604.52500617)/(2 * sp.pi * me))**0.5)* S * ne[0] * sp.exp((l*e)/(kb*m*11604.52500617)) 
    # Ion Frequency OML 
    vi= (((kb * T_Ar)/(2* sp.pi * m_Ar))**0.5)* S * ni[0] * (1-((l * e)/(kb*T_Ar)))
    eqn1= ve-vi
    k= (l * (4 * sp.pi * eo * rp))/(e)   
    t = np.array([300E14])                                 
    eqn2= ne[0]- ni[0] - (k * t[0])
    return(eqn1, eqn2)
    
m, l = fsolve(equations, (2, -2))
print('Te and Vp',m,l)
'''






















    




