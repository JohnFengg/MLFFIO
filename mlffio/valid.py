#!/usr/bin/env python 
import numpy as np
import matplotlib.pyplot as plt
import os,json
from ase.io import read,trajectory,write
from ase.calculators.singlepoint import SinglePointCalculator
from ase import Atoms
from lasp_calculator import lasp_calculator 
from ase.calculators.vasp import Vasp
from mlffio import arc2atoms,atoms2arc
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,root_mean_squared_error
# from custodian.vasp.handlers import UnconvergedErrorHandler
from custodian.vasp.io import load_vasprun

class data_process():
    def __init__(self,atoms_1,atoms_2):
        self.atoms_1=atoms_1
        self.atoms_2=atoms_2

    def mse_calc(self):
        self.energy_1=np.array([atoms.get_potential_energy() for atoms in self.atoms_1])
        self.energy_2=np.array([atoms.get_potential_energy() for atoms in self.atoms_2])
        self.force_1=np.concatenate([atoms.get_forces().flatten() for atoms in self.atoms_1])
        self.force_2=np.concatenate([atoms.get_forces().flatten() for atoms in self.atoms_2])
        # self.force_1=np.array([atoms.get_forces().flatten() for atoms in self.atoms_1])
        # self.force_2=np.array([atoms.get_forces().flatten() for atoms in self.atoms_2])
        err_en=self.mse(self.energy_1,self.energy_2)
        # err_for=np.mean([self.mse(self.force_1[i],self.force_2[i]) for i in range(len(self.force_1))],axis=0)
        err_for=self.mse(self.force_1, self.force_2)
        return err_en,err_for
    
    def results_plot(self,prefix_x,prefix_y,figname):
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.title('Energies comparison')
        plt.scatter(self.energy_1,self.energy_2)
        min,max=self.minmax([self.energy_1,self.energy_2],ratio=0.02)
        plt.xlim((min,max))
        plt.ylim((min,max))
        x_lin,y_lin=np.linspace(min,max,1000),np.linspace(min,max,1000)
        plt.plot(x_lin,y_lin,'-r')
        plt.xlabel(f'Energies of {prefix_x} (eV)')
        plt.ylabel(f'Energies of {prefix_y} (eV)')
        plt.grid()

        plt.subplot(1,2,2)
        plt.title('Forces comparison')
        f_1,f_2=self.force_1.flatten(),self.force_2.flatten()
        plt.scatter(f_1,f_2)
        min,max=self.minmax([f_1,f_2],ratio=0.4)
        plt.xlim((min,max))
        plt.ylim((min,max))
        x_lin,y_lin=np.linspace(min,max,1000),np.linspace(min,max,1000)
        plt.plot(x_lin,y_lin,'-r')
        plt.xlabel(f'Forces of {prefix_x} (eV/A)')
        plt.ylabel(f'Forces of {prefix_y} (eV/A)')
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f'{figname}.png')

    @staticmethod    
    def e_and_f_parse(atoms):
        energies,forces=[],[]
        for atom in atoms:
            energies.append(atom.get_potential_energy())
            forces.extend(atom.get_forces())
        return energies,forces
    
    @staticmethod   
    def random(arc_file,lens):
        atoms_raw=arc2atoms(arc_file)
        if lens>len(atoms_raw):
            raise ValueError(f"Requested {lens} images, but only {len(atoms_raw)} available.")
        index=[i for i in range(len(atoms_raw))]
        selected_index=np.random.choice(index,lens,replace=False)
        print(f'{len(atoms_raw)} images of {arc_file} are detected, {lens} images are extracted randomly')
        return [atoms_raw[i] for i in selected_index]

    @staticmethod 
    def mse(arr_1,arr_2):
        mse=mean_squared_error(arr_1,arr_2)
        mae=mean_absolute_error(arr_1,arr_2)
        rmse=root_mean_squared_error(arr_1,arr_2)
        return [mse,rmse,mae]
    
    @staticmethod 
    def minmax(arr_list,ratio):
        lst_min,lst_max=[],[]
        for arr in arr_list:
            lst_min.append(np.min(arr))
            lst_max.append(np.max(arr))
        return min(lst_min)*(1-ratio),max(lst_max)*(1+ratio)

    @staticmethod
    def vp_check(work_paths:list):
        unconverge_path=[]
        unconverge_num=0
        for path in work_paths:
            path=os.path.join(path,'vasprun.xml')
            # check_converge=UnconvergedErrorHandler()
            # print(check_converge.check(path))
            v=load_vasprun(path)
            if not v.converged:
                print(f'Path : {path} job was unconverge')
                unconverge_path.append(path)
                unconverge_num+=1
        print(f'total number of unconverged jobs is {unconverge_num}')
        return unconverge_path,unconverge_num


class calculation():
    def __init__(self,atoms):
        self.atoms=atoms
    
    def vasp_auto_calc(self,work_dir):
        atoms_lst=[]
        for i,atoms in enumerate(self.atoms):
            atoms_fp=self.vasp_sp(atoms,work_dir=os.path.join(work_dir,i))
            atoms_lst.append(atoms_fp)
        return atoms_lst

    def lasp_calc(self,work_dir,potname):
        lc=lasp_calculator(
                            atoms=self.atoms,
                            command='mpirun -np $SLURM_NTASKS lasp',
                            work_dir=work_dir,
                            mode='nn',
                            potential=potname
                        )
        atoms_nn=lc.calculate()
        return atoms_nn

    @staticmethod 
    def vasp_sp(atoms,work_dir):
        calc = Vasp(xc='pbe',
            directory=work_dir,
            setups={'Y':'_sv','Zr':'_sv'},
            #kpts
            kspacing=0.5,
            kgamma=False,
            #basic
            prec='Normal',
            encut=450,
            nelmin=5,
            nelm=1000,
            ediff=1E-5,
            algo='Fast',
            # nsim=2,
            ismear=0,
            sigma=0.02,
            npar=2,
            kpar=6,
            #Geomopt
            nsw=0,
            ibrion=-1,
            ispin=2,
            isym=0, 
            #output
            lcharg='.FALSE.',
            lwave='.FALSE.',
            # amix=0.2
        )

        atoms.calc=calc
        try:
            atoms.get_potential_energy()
        except:
            pass

        try:
            atoms_new=read(f'{dir}/OUTCAR')
        except Exception:
            atoms_new=Atoms('H',positions=[[0,0,0]])
            atoms_new.set_calculator(SinglePointCalculator(atoms_new,energy=0.0))

        return atoms_new


if __name__ == '__main__':
    # atoms=calculation.random('all.arc',50)+calculation.random('md.arc',50)
    # print(f'Final images for validation calc is {len(atoms)}')
    # calc=calculation(atoms)
    # atoms_fp=calc.vasp_auto_calc(work_dir='tmp_fp')
    # write('fp.traj',atoms_fp)
    # atoms_nn=calc.lasp_calc(work_dir='tmp_nn',potname='YSZ.pot')
    # write('nn.traj',atoms_nn)

    # results={}
    # for i in ['3','6','9','12']:
    #     atoms_fp=read(os.path.join(i,'fp.traj'),'50:')
    #     atoms_nn=read(os.path.join(i,'nn.traj'),'50:')
    #     print(len(atoms_fp),len(atoms_nn))
    #     dp=data_process(atoms_fp,atoms_nn)
    #     err_en,err_for=dp.mse_calc()
    #     results[i]={'energy: eV':{'mse':err_en[0],'rmse':err_en[1],'mae':err_en[2]},
    #              'force: eV/A':{'mse':err_for[0],'rmse':err_for[1],'mae':err_for[2]}}
    #     dp.results_plot('DFT','NN',f'{i}_md_err.png')
    # with open('md_err.json', 'w') as f:
    #     json.dump(results,f,indent=2)

    atoms_fp=read(os.path.join('3','fp.traj'),':')+read(os.path.join('6','fp.traj'),':')+read(os.path.join('9','fp.traj'),':')+read(os.path.join('12','fp.traj'),':')
    atoms_nn=read(os.path.join('3','nn.traj'),':')+read(os.path.join('6','nn.traj'),':')+read(os.path.join('9','nn.traj'),':')+read(os.path.join('12','nn.traj'),':')
    print(len(atoms_fp),len(atoms_nn))
    dp=data_process(atoms_fp,atoms_nn)
    err_en,err_for=dp.mse_calc()
    dp.results_plot('DFT','NN','total_err.png')
    results={'energy: eV':{'mse':err_en[0],'rmse':err_en[1],'mae':err_en[2]},
             'force: eV/A':{'mse':err_for[0],'rmse':err_for[1],'mae':err_for[2]}}
    with open('total_err.json', 'w') as f:
        json.dump(results,f,indent=2)
