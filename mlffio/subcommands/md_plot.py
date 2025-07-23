#!/usr/bin/env python
import matplotlib.pyplot as plt 
import numpy as np
import os 

os.system("echo '#time_fs  temp_K  pot_ene_eV  msd/6t_A^2/fs' > etmsd.txt")
os.system("grep mdstep lasp.out | awk '{print $2,$4,$3,$9}' >> etmsd.txt")
data=np.genfromtxt('etmsd.txt')
time,temp,e,d=data[:,0],data[:,1],data[:,2],data[:,3]*0.1

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.title('time v.s. temperature')
plt.scatter(time,temp)
plt.xlabel('time (fs)')
plt.ylabel('temperature (K)')
plt.grid()

plt.subplot(1,3,2)
plt.title('time v.s. potential_energy')
plt.scatter(time,e)
plt.xlabel('time (fs)')
plt.ylabel('total_energy (eV)')
plt.grid()

plt.subplot(1,3,3)
plt.title('time v.s. diffusion coefficient')
plt.scatter(time,d)
plt.xlabel('time (fs)')
plt.ylabel('diffusion coefficient (cm^2/S)')
plt.xlim((len(time)*0*10,len(time)*10*1))
minimum=min(d[int(len(time)*0.2):int(len(time))])*0.8
maximum=max(d[int(len(time)*0.2):int(len(time))])*1.2
plt.ylim((minimum,maximum))
plt.grid()
plt.tight_layout()
plt.savefig('md-results.png')