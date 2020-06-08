import numpy as np
import matplotlib.pyplot as plt
eo = 8.8541878176E-12                 # Permittivity of Free Space [m^-3*kg^-1*s^4*A^2]
rp = 7E-9                             # radius of carbon nanoparticles
e= 1.6021765E-19                      # Elementary Electron Charge [C]



Nii = np.array([1.56E16, 1.56E16,1.56E16])
Nee = np.array([5.77E15, 3.77E15, 1.77E15]) 
Vpo = np.array([-1.41280609737307, -1.1581288283609794,-0.5325289082093315])
npo = np.array([1431280256472731.0, 2101268720204989.0, 5342354649209894.0])
sye = [0.72, 0.643, 0.666]
sy = [2.62, 1.802, 1.93]
Tee = [3.1, 2.755, 2.29]
EN = [70]
SDA = [-1.52E15, -1.3E15, -1.93E15]
neni = Nee / Nii
ch_de = -(4 * np.pi * eo * rp * Vpo) /e
charge_detach = np.array([7,6,3])


# Without SEE 
Vpwo = np.array([-3.7230883876049456, -3.01413627742787, -1.8724805955907564])
npwo = np.array([543130128236142.0, 807375532163812.0, 1519352614553116.0])
Teewo = np.array([4.2, 4.088744647778105, 4.210633956669226])
ch_at = -(4 * np.pi * eo * rp * Vpwo) /e
charge_attach = np.array([18,15,10])

# OML Theory for given ne, ni, Te
Vpoml = np.array([-3.712, -2.867, -1.79])
ch_oml = -(4 * np.pi * eo * rp * Vpoml) /e
charge_oml = np.array([18, 14, 9])
print(ch_de)
print(ch_at)
print(ch_oml)



#plt.plot(Nee, npo)
plt.plot(Nee, charge_detach)
plt.plot(Nee, charge_attach)
plt.plot(Nee, charge_oml)
#plt.plot(neni, carge_attach)
#plt.plot(neni, carge_oml)
plt.xlabel('Electron Density [m-3]')
plt.ylabel('Number of Charges Per Particle')
plt.legend(['With Secondary Emission', 'Without Secondary Emission', 'OML Theory'])
fig = plt.gcf()
plt.xlim(1.5E15,6E15)
plt.ylim(2,20)

plt.grid(True)
fig.set_size_inches(8.5, 8.5)
plt.rcParams.update({'font.size': 8})
