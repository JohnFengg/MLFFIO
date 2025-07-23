#!/usr/bin/env python
import numpy as np 
import matplotlib.pyplot as plt 
import warnings
import os 
from . import lasp-process

lasp-process.run_loss()

loss_data=np.genfromtxt('loss.txt')
total,e,f,s=loss_data[:,0],loss_data[:,1],loss_data[:,2],loss_data[:,3]
rmse_data=np.genfromtxt('rmse.txt')
r_e,r_f,r_s=rmse_data[:,0],rmse_data[:,1],rmse_data[:,2]
if len(total)!=len(r_e):
    warnings.warn('length mismatch is detected: the loss value and rmse are not the same',UserWarning)
steps=np.linspace(0,len(total),len(total))

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(steps,total,label='total_loss')
plt.plot(steps,e,label='energy_loss (meV/atom)')
plt.plot(steps,f,label='forces_loss (eV/A)')
plt.plot(steps,s,label='stress_loss (GPa)')
plt.ylim((-1,total[30]*1.2))
plt.legend()
plt.grid()
plt.title('loss function')

plt.subplot(2,1,2)
plt.plot(steps,r_e,label='energy (meV/atom)')
plt.plot(steps,r_f*100,label='forces (e+01 eV/A)')
plt.plot(steps,r_s,label='stress (GPa)')
plt.legend()
plt.grid()
plt.ylim((-1,max(r_s[30],r_f[30],r_s[30])*1.2))
plt.title('RMSE')
plt.tight_layout()
plt.savefig('loss_and_rmse.png')

