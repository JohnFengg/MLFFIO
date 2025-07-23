#!/usr/bin/env python
import os,json
import numpy as np
from custodian.vasp.handlers import UnconvergedErrorHandler
from custodian.vasp.io import load_vasprun
from ase.io import read,trajectory
from mlffio import atoms2arc

class sampling():
    def __init__(self,work_dir):
        self.work=work_dir
        if 'src' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'src'))
        self.src=os.path.abspath(os.path.join(self.work,'src'))
        if 'sampling' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'sampling'))
        self.samp=os.path.abspath(os.path.join(self.work,'sampling'))

    def config_init(self,config_path,filename='struc.arc'):
        """
        config_path is a list of path, like ["../init/1/","../init/2/","..."]
        """
        index=0
        job_path=[]
        for path in config_path:
            if str(index) not in os.listdir(self.samp):
                os.mkdir(os.path.join(self.samp,str(index)))
            src_path=os.path.abspath(os.path.join(path,filename))
            targ_path=os.path.abspath(os.path.join(self.samp,str(index)))
            job_path.append(targ_path)
            relpath=os.path.relpath(src_path,start=targ_path)
            if not os.path.lexists(os.path.join(targ_path,filename)):
                os.symlink(relpath,os.path.join(targ_path,filename))
            index+=1
        # print(job_path)
        with open(os.path.join(self.work,'job_path'),'w') as f:
            json.dump(job_path,f,indent=2)

    def write_input(self,pot_file,pot_name,input_file=None,slurm_file=None):
        self.pot_name=pot_name
        if input_file is not None:
            os.system(f"cp {input_file} {self.src}/lasp.in")
        else:
            f_input=open(f'{self.src}/lasp.in','w')
            f_input.write()
            f_input.close()
        os.system(f"cp {pot_file} {self.src}")
        slurm=slurm_file is not None
        if slurm:
            os.system(f"cp {slurm_file} {self.src}/sub.slurm")
        
        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)
        for path in job_path:
            os.symlink(os.path.relpath(
                                        os.path.join(self.src,'lasp.in'),
                                        start=path
                                        ),
                       os.path.join(path,'lasp.in'))
            os.symlink(os.path.relpath(
                                        os.path.join(self.src,pot_name),
                                        start=path
                                        ),
                       os.path.join(path,pot_name))
            if slurm:
                os.symlink(os.path.relpath(
                                        os.path.join(self.src,'sub.slurm'),
                                        start=path
                                        ),
                           os.path.join(path,'sub.slurm'))

    def run(self,command=None):
        if command is not None:
            self.command=command
        else:
            self.command='sbatch sub.slurm'
        #     check_list=['lasp.in',self.pot_name]
        # else:
        #     check_list=['lasp.in',self.pot_name,'sub.slurm']
        root_dir=os.getcwd()

        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)

        for path in job_path:
            os.chdir(path)
            # check=[a in os.listdir('./') for a in check_list]
            # if all(check):
            os.system(self.command)
            # else:
            #     raise ValueError('The input files in not complete, stop running job...')
            os.chdir(root_dir)

    def check():
        pass

class labelling():
    def __init__(self,work_dir):
        self.work=work_dir
        if 'src' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'src'))
        self.src=os.path.abspath(os.path.join(self.work,'src'))
        if 'labelling' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'labelling'))
        self.label=os.path.abspath(os.path.join(self.work,'labelling'))

    def traj_extract(self,ratio,samp_path,filename='allstr.arc',continued=0):
        with open(os.path.join(samp_path,'job_path'),'r') as f:
            samp_job_path=json.load(f)
        
        label_path=[]
        for i,path in enumerate(samp_job_path):
            if str(i) not in os.listdir(self.label):
                os.mkdir(os.path.join(self.label,str(i)))
            tmp_job_path=os.path.join(self.label,str(i))
            atoms=read(os.path.join(path,filename),':')
            idx=[i for i in range(len(atoms))]
            mask=list(
                np.random.choice(
                                idx,
                                int(len(idx)*ratio),
                                replace=False
                                )
                    )
            seleted_traj=[atoms[n] for n in mask]
            for j in range(continued,len(idx)*ratio,1):
                if str(j) not in os.listdir(tmp_job_path):
                    os.mkdir(os.path.join(tmp_job_path,str(j)))
                label_path.append(os.path.join(tmp_job_path,str(j)))
                seleted_traj[j].write(os.path.join(tmp_job_path,str(j),'POSCAR'))

            remenent=[i for i in idx if i not in mask]        
            remenent_traj=[atoms[r] for r in remenent]
            traj_r=trajectory.Trajectory(f'{tmp_job_path}/remenent.traj','w')
            for j in range(len(remenent_traj)):
                traj_r.write(remenent_traj[j])
        with open(os.path.join(self.work,'job_path'),'w') as f:
            json.dump(label_path,f,indent=2)

    def write_input(self,input_file,slurm_file=None):
        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)
        
        os.system(f"cp {input_file} {os.path.join(self.src,'run.py')}")
        slurm=slurm_file is not None
        if slurm:
            os.system(f"cp {slurm_file} {os.path.join(self.src,'sub.slurm')}")
        for path in job_path:
            os.symlink(os.path.relpath(
                                        os.path.join(self.src,'run.py'),
                                        start=path
                                        ),
                       os.path.join(path,'run.py'))
            if slurm:
                os.symlink(os.path.relpath(
                                            os.path.join(self.src,'sub.slurm'),
                                            start=path
                                            ),
                        os.path.join(path,'sub.slurm'))

    def run(self,command=None):
        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)

        if command is not None:
            self.command=command
        else:
            self.command='sbatch sub.slurm'

        root_dir=os.getcwd()
        for path in job_path:
            os.chdir(path)
            os.system(self.command)
            os.chdir(root_dir)
        
    def check(self):
        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)

        dict={'job_number':0,'unconverged_number':0}
        dict['job_number']=len(job_path)
        unconverge=0
        for path in job_path:
            v=load_vasprun(os.path.join(path,'vasprun.xml'))
            if not v.converged:
                unconverge+=1
            dict[path]=v.converged
        dict['unconverged_number']=unconverge
        with open('dft_check.log','w') as f:
            json.dump(dict,f,indent=2)

    def results_pkg(self):
        with open(os.path.join(self.work,'job_path'),'r') as f:
            job_path=json.load(f)
        traj=trajectory.Trajectory(f'{self.work}/dft.traj','w')
        for path in job_path:
            atoms=read(f'{os.path.join(path,'OUTCAR')}')
            traj.write(atoms)
        
class train():
    def __init__(self,work_dir):
        self.work=work_dir
        if 'src' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'src'))
        self.src=os.path.abspath(os.path.join(self.work,'src'))
        if 'train' not in os.listdir(self.work):
            os.mkdir(os.path.join(self.work,'train'))
        self.trn=os.path.abspath(os.path.join(self.work,'train'))
        
    def write_input(self,traj_file,pot_name=None,pot_file=None,loss_set=None,input_file=None,slurm_file=None):
        atoms=[]
        for traj in traj_file:
            atom=read(traj,':')
            atoms.extend(atom)
        atoms2str(atoms,'allstr.arc',self.trn,write_force=True)
        root_dir=os.getcwd()
        os.chdir(self.trn)
        os.system('shiftformat arc2train allstr.arc allfor.arc')
        os.chdir(root_dir)

        if input_file is not None:
            os.system(f"cp {input_file} {self.src}/lasp.in")
        else:
            f_input=open(f'{self.src}/lasp.in','w')
            f_input.write()
            f_input.close()

        if loss_set is not None:
            os.system(f"cp {loss_set} {self.trn}/adjust_factor")
        else:
            f_input=open(f'{self.trn}/adjust_factor','w')
            f_input.write()
            f_input.close()

        if pot_file is not None:
            os.system(f"cp {pot_file} {os.path.join(self.trn,pot_name)}")

        slurm=slurm_file is not None
        if slurm:
            os.system(f"cp {slurm_file} {self.src}/sub.slurm")
        os.symlink(os.path.relpath(
                                    os.path.join(self.src,'lasp.in'),
                                    start=self.trn
                                    ),
                    os.path.join(self.trn,'lasp.in'))
        if slurm:
            os.symlink(os.path.relpath(
                                    os.path.join(self.src,'sub.slurm'),
                                    start=self.trn
                                    ),
                        os.path.join(self.trn,'sub.slurm'))

    def run(self,command=None):
        if command is not None:
            self.command=command
        else:
            self.command='sbatch sub.slurm'

        root_dir=os.getcwd()
        os.chdir(self.trn)
        os.system(self.command)
        os.chdir(root_dir)

if __name__=="__main__":
    samp=sampling(work_dir='1-ssw')
    config_path=[
        "../source/init_struc/3mol/0",
        "../source/init_struc/3mol/1",
        "../source/init_struc/3mol/2",
        "../source/init_struc/3mol/3",
        "../source/init_struc/3mol/4",
        "../source/init_struc/6mol/0",
        "../source/init_struc/6mol/1",
        "../source/init_struc/6mol/2",
        "../source/init_struc/6mol/3",
        "../source/init_struc/6mol/4",
        "../source/init_struc/9mol/0",
        "../source/init_struc/9mol/1",
        "../source/init_struc/9mol/2",
        "../source/init_struc/9mol/3",
        "../source/init_struc/9mol/4",
        "../source/init_struc/12.5mol/0",
        "../source/init_struc/12.5mol/1",
        "../source/init_struc/12.5mol/2",
        "../source/init_struc/12.5mol/3",
        "../source/init_struc/12.5mol/4"
    ]
    samp.config_init(config_path=config_path)
    samp.write_input(pot_file='../c003/3-train/run1/train/YSZ.pot',
                     pot_name='YSZ.pot',
                     input_file='./lasp-ssw.in',
                     slurm_file='./lasp-ssw.slurm')
    samp.run()

    label=labelling(work_dir='2-dft-recalc')
    label.traj_extract(ratio=1,
                           samp_path='./1-ssw/',
                           filename='Badstr.arc',
                           continued=0)
    label.write_input(input_file='fp.py',
                          slurm_file='fp.slurm')
    label.run()
    label.check()
    label.results_pkg()

    trn=train('3-train/run1')
    trn.write_input(traj_file=['../c002/2-dft-recalc/dft.traj','../c001/2-dft-recalc/all.traj'],
                      loss_set='adjust_factor',
                    #   pot_file='3-train/run1/train/SavePot/pot_in_step-500',
                    #   pot_name='YSZ.input',
                      input_file='lasp-train.in',
                      slurm_file='sub-train.slurm')
    trn.run()