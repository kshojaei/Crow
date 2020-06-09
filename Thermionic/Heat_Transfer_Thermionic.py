# Energy Balance for particle heating in a dusty plasma
import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
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
m_H22 = 2*1.6737236E-27                  # Mass of Hydrogen [kg]
m_avg = (19.3/20)*m_Ar + (0.7/20)*(m_H2*2)
m_ArH = m_Ar + m_H2
m_dom = 3 * m_H2                      # Mass of the Dominant Ion H3+ [kg]
#mi = (19.7/20)*(6.6335209E-26) + (0.3/20)*(1.6737236E-27) 
RA= 9E-2;                             # Radius of the Plasma [m]
d= 10.16E-2;                          # Distance between two plates [m]
Ae= 2 * np.pi * RA * d;             # Area Bounding the Plasma Volume [m^2]
V= (RA ** 2) * np.pi * d;           # Volume of the plasma region [m^3]
#m_particles = 2250 * ((4/3)*math.pi*(rp**3))# Mass per particle [kg]
Te = 4.3                              # Electron Temperature [eV] @ np: 1.0E15 [1/m3]
rp = 6.5E-9                           # radius of carbon nanoparticles
dp = 2 * rp                           # Diameter of carbon nanoparticles [m]
m_particles = 2266 * ((4/3)*np.pi*(rp**3))
eo = 8.8541878176E-12                 # Permittivity of Free Space [m^-3*kg^-1*s^4*A^2]
Ti = 300                              # Ion Temperature [K]
Te_K = 11604.52500617 * Te            # Electron Temperature [K]
ne = np.array([4.21E15])              # Electron Density [1/m3]
ni = np.array([1.56E16])              # Ion Density [1/m3]
e= 1.6021765E-19                      # Elementary Electron Charge [C]
kb= 1.38064853E-23                    # Boltzmann Constant [J/K]
me= 9.10938E-31                       # Mass of electron [kg]
gamma = ((2 * e)/me) ** 0.5           # Constant 
# Mass of ions [kg]
mi = (19.3/20)*(6.6335209E-26) + (0.7/20)*(1.6737236E-27)  
c = 2.998E8           # Speed of light [m/s]
H_pot = 13.6          # Ionization Potential of H2 [eV]
# Electron Debye Length [m]
lam_e = ((eo * kb * Te_K)/(e * e * ne)) ** 0.5
# Ion Debye Length [m]
lam_i = ((eo * kb * Ti)/(e * e * ni)) ** 0.5
# Linearized Debye Length [m]
h = 6.626E-34                    # Planck Constant [Js]
S = 4 * np.pi * (rp**2)
A = 5.1                          # Electorn Affinity of Bulk Carbon [eV]
T_gas = 300
################################################################
lamda = np.linspace(13E-9,100000E-9,3000)   
nm = 2.213 + 9.551E3 * lamda
km = 0.7528 + 1.265E4 * lamda
Em = (6*nm*km)/((((nm**2) - (km**2) + 2 )**2) + 4*((nm*km)**2))
emis = (4*np.pi*(2*rp)*Em) / lamda
def Cond(H):
    con_ar = (19.3/20)*(0.25 * nf * 1 * np.sqrt((8 * kb * T_gas)/(np.pi * m_Ar)) * 1.5 * kb * (1312.3328 -T_gas))
    con_h = (0.7/20)*(0.25 * nf * 1 * np.sqrt((8 * kb * T_gas)/(np.pi * m_H22)) * 1.5 * kb * (1312.3328 -T_gas))
    return (con_ar + con_h) * 4 * np.pi * (H**2)
''' Radiative Energy Loss [J/s] '''
def radiation(k,Tp):
    result = list()
    for i in Tp:
        if i > 0:
            radd = emis*np.pi*(dp**2)*(2*np.pi*h*c*c)/((lamda**5)*(np.exp((h*c)/(lamda*kb*i)) - 1))            
            result.append((np.trapz(radd,lamda))*(k/k))
        else:
            result.append(0)
    return result

def rad(H):
    result = list()
    for i in H:
        if i > 0:
            emiss = (4*np.pi*(2*i)*Em) / lamda
            radd = 4*emiss*np.pi*(i**2)*(2*np.pi*h*c*c)/((lamda**5)*(np.exp((h*c)/(lamda*kb*1312.3328)) - 1))            
            result.append((np.trapz(radd,lamda)))
        else:
            result.append(0)
    return result
T = np.linspace(1E-9,100E-9,3000)
plt.plot(T*1E9, rad(T))
plt.plot(T*1E9, Cond(T))
plt.legend(['Radiation', 'Conduction'])

plt.xlabel('temp')
plt.ylabel('radiation')
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(10,10)
np.savetxt('Radiation_Conduc.txt', zip(T,rad(T),Cond(T)), delimiter=',', fmt='%.15f')


''' Conduction Energy Loss [J/s] '''
def Conduction(k,Tp):
    con_ar = (19.3/20)*(0.25 * nf * S * np.sqrt((8 * kb * T_gas)/(np.pi * m_Ar)) * 1.5 * kb * (Tp -T_gas))*(k/k)
    con_h = (0.7/20)*(0.25 * nf * S * np.sqrt((8 * kb * T_gas)/(np.pi * m_H22)) * 1.5 * kb * (Tp -T_gas))*(k/k)
    return (con_ar + con_h)
''' Removal Energy [J/s] '''
def A_u(k): 
    a = 5.4/(rp*1E9*10)
    b = ((2*np.absolute(k)-1) * e) / (8 * np.pi * eo * rp)
    return A + a - b
'''  Thermionic Emission Frequency [J/s] ''' 
def Thermionic(k,Tp):
    WW = e * A_u(k)
    Coeff = 4 * me * ((2*rp*np.pi*kb*Tp)**2)
    return (Coeff/(h**3)) * WW * np.exp(- WW / (kb * Tp))
def Therm_Freq(k,Tp):
    WW = e * A_u(k)
    Coeff = 4 * me * ((2*rp*np.pi*kb*Tp)**2)
    return (Coeff/(h**3)) * 1 * np.exp(-WW / (kb * Tp))
# Ion Frequnency H3+ [1/s]
def freq_i_H3(k):
    S = 4 * np.pi * (rp**2)
    Vp = (-(k * e) / ( 4 * np.pi * eo * rp))
    A = S * (ni) * (((kb* T_gas)/(2* np.pi * m_dom))** 0.5)
    nu = A * (1 - ((e * Vp)/(kb * T_gas)))
    return nu
# Ion Frequency H2+ [1/s]
'''
def freq_i_H2(k):
    Sh = 4 * np.pi * (rp**2)
    Vph = (-(k * e) / ( 4 * np.pi * eo * rp))
    Ah = Sh * (1E13) * (((kb* T_gas)/(2* np.pi * (2*m_H2)))** 0.5)
    nh = Ah * (1 - ((e * Vph)/(kb * T_gas)))
    return nh
# Ion Frequency Ar+ [1/s]
def freq_i_Ar(k):
    SS = 4 * np.pi * (rp**2)
    Vpp = (-(k * e) / ( 4 * np.pi * eo * rp))
    AA = SS * (8.33E15) * (((kb* T_gas)/(2* np.pi * m_Ar))** 0.5)
    nuu = AA * (1 - ((e * Vpp)/(kb * T_gas)))
    return nuu
# Ion Frequency ArH+ [1/s]
def freq_i_ArH(k):
    SSS = 4 * np.pi * (rp**2)
    Vppp = (-(k * e) / ( 4 * np.pi * eo * rp))
    AAA = SSS * (8.33E15) * (((kb* T_gas)/(2* np.pi * m_ArH))** 0.5)
    w = AAA * (1 - ((e * Vppp)/(kb * T_gas)))
    return w  
'''
# Total Ion Frequency Ar+ & ArH+ & H3+ [1/s]
def freq_i_tot(k):
    return freq_i_H3(k)

# Electron Frequency [1/s]
def freq_e(k,Tp):
    return Therm_Freq(k,Tp) + freq_i_tot(k)

# Electron Kinetic Power [J/s]
def EKE(k,Tp):
    return (freq_e(k,Tp)*2*kb*Te_K)

# Ion Kinetic Power [J/s]
def IKE(k,Tp):
    return (freq_i_tot(k)*((0.5*kb*Te_K)+ (((k * e) / ( 4 * np.pi * eo * rp))*e)))*(Tp/Tp)
# Total Recombination [J/s]   
         
def Recombination(k,Tp):
    rec_h3 = freq_i_tot(k)*(13.6+(4.52/2))*e
    return (rec_h3)*(Tp/Tp)

# Association Energy (H + H => H2) [J/s]
surf = 4 * np.pi * (rp**2)
a_H2 = 1E13 * np.sqrt((8*kb*T_gas)/(np.pi * 2 * m_H2)) * 4.52 
a_ArH = 8.33E15 * np.sqrt((8*kb*T_gas)/(np.pi * m_ArH)) * 4.02
a_H3 = (3.332E15) * np.sqrt((8*kb*T_gas)/(np.pi*m_dom)) * 4.37
Assoc = ((0.5 * 1 * 0.25) * surf * (a_H2 + a_ArH + a_H3) * e)
def Association(Tp):
    return Assoc*(Tp/Tp)

# Total Energy Loss [W]
def Loss(k,Tp):
    return (Conduction(k,Tp) + Thermionic(k,Tp) + radiation(k,Tp))
# Total Energy Gain [W]
def Gain(k,Tp):
    return Recombination(k,Tp) + IKE(k,Tp) + EKE(k,Tp)

char = 11.6
# Net Energy
def Energy(Tp):
    return (Gain(char,Tp) - Loss(char,Tp))
T_opt = fsolve(Energy,1000)
print('Particle Temperature',T_opt[0])
# Net Frequency
def Frequency(al):
    return (-freq_i_tot(al) + Therm_Freq(al,T_opt[0]))
charge = fsolve(Frequency,30)
print(char,charge[0])
print(Therm_Freq(char,T_opt[0])/Therm_Freq(charge[0],T_opt[0]))
#################################################
'''
plt.subplot(211)
T = np.linspace(1000,1800,3000)
plt.plot(T,Loss(char,T))
plt.plot(T, Gain(char,T))
plt.yscale('log')
plt.xlabel('Temperature [K]')
plt.ylabel('Energy')
plt.legend(['Loss', 'Gain'])
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(10,10)
plt.rcParams.update({'font.size': 12})
plt.subplot(212)
kk = np.linspace(5,15,3000)
plt.plot(kk,freq_i_tot(kk))
plt.plot(kk,Therm_Freq(kk,T_opt))
plt.yscale('log')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.legend(['Ion Collection', 'Thermionic'])
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(10,10)
plt.rcParams.update({'font.size': 12})
'''
'''
#plt.subplot(211)
T = np.linspace(300,1500,3000)
plt.plot(T, Conduction(11.6,T))
plt.plot(T, radiation(11.6,T))
plt.plot(T, Thermionic(11.6,T))
plt.plot(T, Recombination(11.6,T))
plt.plot(T, IKE(11.6,T))
plt.plot(T, EKE(11.6,T))
plt.xlabel(' Particle Temperature [K]')
plt.ylabel('Energy [W]')
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(12,12)
plt.rcParams.update({'font.size': 12})
plt.legend(['Conduction Heat Loss', 'Radiative Heat Loss', 'Thermionic Emission Heat Loss', 'Charge Recombination Heat Gain', 'IKE (Ion Kinetic Energy Deposition) Heat Gain', 'EKE (Electron Kinetic Energy Desposition)  Heat ain'])
fig = plt.gcf()
plt.grid(True)
fig.set_size_inches(12,12)
plt.rcParams.update({'font.size': 12})
'''
#print(Conduction(11.6,T))
#np.savetxt('Heat_2.txt', zip(T,Conduction(11.6,T),radiation(11.6,T),Thermionic(11.6,T),Recombination(11.6,T),IKE(11.6,T),EKE(11.6,T)), delimiter=',', fmt='%.15f')
'''
Te = np.array([4.1,1.1, 0.5])
Charges = np.array([11.5, 12.86, 13.22])
Particle_temp = np.array([1330,1203,1168])
therm_ion = np.array([0.96,0.99,0.99])


Tee = np.array([4.1, 3.4, 2.5, 1.5, 0.5])
Chargess = np.array([11.52, 11.52, 11.52, 11.52])
Particle_tempp = np.array([1330, 1261, 1221, 1181, 1139])
therm_ionn = np.array([0.97, 0.22, 0.09, 0.03, 0.01])
'''


'''radiation > conduction > Thermionic '''
''' EKE > RECOM > ION'''


    





























