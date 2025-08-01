#!/usr/bin/env python
from ase.io import read,write
from ase.calculators.vasp import Vasp
import os,subprocess
from mlffio import arc2atoms,atoms2arc
from ase.io.trajectory import Trajectory

class lasp_calculator():
    def __init__(self,atoms:list,command:str,work_dir:str,mode:str,potential=None):
        """
        mode='nn','ssw','geomopt'
        """
        self.atoms=atoms
        self.command=command
        self.dir=work_dir
        if potential is not None:
            self.potential=potential
        self.mode=mode

    def write_input(self,atoms):
        if self.dir not in os.listdir():
            os.mkdir(self.dir)
        atoms2arc([atoms],arcfile=f'{self.dir}/struc.arc',has_energy=False)
        f_input=open(f'{self.dir}/lasp.in','w')
        f_input.write('#Generated by lasptools')
        f_input.write('str_input struc.arc\n')
        src_pot_path=os.path.abspath(self.potential)
        work_pot_path=os.path.abspath(self.dir)
        rel_path=os.path.relpath(src_pot_path,start=work_pot_path)
        if not os.path.lexists(os.path.join(work_pot_path,self.potential)):
            os.symlink(rel_path,os.path.join(work_pot_path,self.potential))
        symbols=list(dict.fromkeys(atoms.get_chemical_symbols()))
        f_input.write('potential NN\n')
        f_input.write('%block netinfo\n')
        for symbol in symbols:
            f_input.write(f'  {symbol} {self.potential}\n')
        f_input.write('%endblock netinfo\n')
        f_input.write('PrintChg 0\n')

        if self.mode=='nn':
            f_input.write('explore_type readarc\n')

        elif self.mode=='geomopt':
            f_input.write('explore_type ssw\n')
            f_input.write('Run_type 0\n')
            f_input.write('SSW.SSWsteps 1\n')
            f_input.write('SSW.output T\n')
            f_input.write('SSW.printevery T\n')
            f_input.write('SSW.ftol 0.03\n')

        elif self.mode=='ssw-variable':
            f_input.write('explore_type ssw\n')
            f_input.write('Run_type 15\n')
            f_input.write('SSW.quick_setting 1\n')
            f_input.write('SSW.SSWsteps 2000\n')
            f_input.write('SSW.Temp 100\n')
            f_input.write('SSW.output T\n')
            f_input.write('SSW.printevery T\n')
            f_input.write('SSW.printdelay -1\n')

        elif self.mode=='ssw-fixed':
            f_input.write('explore_type ssw\n')
            f_input.write('Run_type 5\n')
            f_input.write('SSW.quick_setting 1\n')
            f_input.write('SSW.SSWsteps 2000\n')
            f_input.write('SSW.Temp 100\n')
            f_input.write('SSW.output T\n')
            f_input.write('SSW.printevery T\n')
            f_input.write('SSW.printdelay -1\n')
        f_input.close()
        
    def read_output(self,struc_file=None,force_file=None):
        if struc_file is None:
            return arc2atoms(os.path.join(self.dir,'allstr.arc'),
                                os.path.join(self.dir,'allfor.arc'))  
        elif force_file is None:
            return arc2atoms(os.path.join(self.dir,struc_file))
        else:
            return arc2atoms(os.path.join(self.dir,struc_file),
                    os.path.join(self.dir,force_file))      


    def run(self):
        current_dir=os.getcwd()
        os.chdir(self.dir)
        try:
            results=subprocess.run(self.command,shell=True,capture_output=True,text=True,check=True)
        except subprocess.CalledProcessError as e:
            print('Calculation failed')
        os.chdir(current_dir)
    
    def calculate(self,**kwarg):
        results=[]
        for atoms in self.atoms:
            self.write_input(atoms)
            self.run()
            results.append(self.read_output(**kwarg)[-1])
        return results

if __name__=="__main__":
    # atoms=[arc2atoms('all.arc')[0]]
    # lasp_calc=lasp_calculator(
    #                           atoms=atoms,
    #                           command='mpirun -np 8 lasp',
    #                           work_dir='your_path',
    #                           mode='nn',
    #                           potential='your_filename'
    #                         )
    # atoms_nn=lasp_calc.calculte()
    # traj=Trajectory('your_filename','w')
    # traj.write(atoms_nn[0])
    
    atoms=[arc2atoms(file)[0] for file in os.listdir() if 'arc' in file]
    print('%s images are detected in the folder'%(len(atoms)))
    lasp_calc=lasp_calculator(
                              atoms=atoms,
                              command='mpirun -np 28 lasp',
                              work_dir='your_path',
                              mode='ssw-fixed',
                              potential='your_filename'
                            )
    atoms_nn=lasp_calc.calculate(struc_file='best.arc')
    write('opt.traj',atoms_nn)
    
