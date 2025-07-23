#!/usr/bin/env python
from ase.io import read,trajectory,write
import numpy as np 
from mlffio import arc2atoms,atoms2arc
import os,sys
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
import spglib
from ase.build import surface

class lattice():
    def __init__(self):
        pass
    
    @staticmethod
    def lattice_judge(atoms):
        results=[]
        for atom in atoms:
            structure=AseAtomsAdaptor.get_structure(atom)
            sga=SpacegroupAnalyzer(structure)
            results.append(sga.get_crystal_system())
        return results
    
    @staticmethod    
    def get_crystal_system(atoms, tol=2):
        results=[]
        for atom in atoms:
            n = 'triclinic' 
            cell=atom.get_cell().array
            lengths = np.linalg.norm(cell, axis=1)
            a, b, c = lengths
            alpha = np.degrees(np.arccos(np.dot(cell[1], cell[2]) / (np.linalg.norm(cell[1]) * np.linalg.norm(cell[2]))))
            beta  = np.degrees(np.arccos(np.dot(cell[0], cell[2]) / (np.linalg.norm(cell[0]) * np.linalg.norm(cell[2]))))
            gamma = np.degrees(np.arccos(np.dot(cell[0], cell[1]) / (np.linalg.norm(cell[0]) * np.linalg.norm(cell[1]))))
            print(alpha,beta,gamma)
            def close(x, y): return abs(x - y) < tol

            if close(a, b) and close(b, c):
                if close(alpha, 90) and close(beta, 90) and close(gamma, 90):
                    n='cubic'
                elif close(alpha, beta) and close(beta, gamma):
                    n='trigonal'
            elif close(a, b) and close(alpha, 90) and close(beta, 90) and close(gamma, 120):
                n='hexagonal'
            elif close(a, b) and close(alpha, 90) and close(beta, 90) and close(gamma, 90):
                n='tetragonal'
            elif close(alpha, 90) and close(beta, 90) and close(gamma, 90):
                n='orthorhombic'
            elif close(alpha, gamma) and close(beta, 90):
                n='monoclinic'
            else:
                n='triclinic'

            results.append(n)
        
        return results
    
    @staticmethod
    def get_space_group(atoms,tol=1e-3):
        sgs=[]
        for atom in atoms:
            lattice=atom.get_cell()
            positions=atom.get_scaled_positions()
            numbers=atom.get_atomic_numbers()
            cell=(lattice,positions,numbers)
            sg=spglib.get_spacegroup(cell,symprec=tol)
            sgs.append(sg)
        return sgs
    
    @staticmethod
    def slab_gen(atoms,indice,layers,vacuum=15.0):
        slab=surface(lattice=atoms,indices=indice,layers=layers,vacuum=vacuum)
        return slab

class compostion():
    def __init__(self,atoms):
        self.atoms=atoms

    def oxide_doping(self,substituted_ele,doping_ele,doping_num,vo_num,supercell):
        atoms_super=self.atoms.repeat(supercell)
        o_idx=[atom.index for atom in atoms_super if atom.symbol=='O']
        vo_selected=np.random.choice(o_idx,vo_num,replace=False)
        atoms_vo=atoms_super[[i for i in range(len(atoms_super)) if i not in set(vo_selected)]]
        zr_idx=[atom.index for atom in atoms_vo if atom.symbol==substituted_ele]
        zr_seleted=np.random.choice(zr_idx,doping_num,replace=False)
        atoms_new=atoms_vo.copy()
        for zr in zr_seleted:
            atoms_new[zr].symbol=doping_ele
        return atoms_new




if __name__=='__main__':
    # files=[i for i in os.listdir() if 'ZrO2' in i]
    # atoms=[read(file,format='vasp') for file in files]
    # # print(get_crystal_system(atoms,tol=1e-5))
    # print(lattice.get_space_group(atoms))
    # print(lattice.lattice_judge(atoms))
    # for i in [3,6,9,12]:
    #     atoms=read(f'../{i}/all.arc@0')
    #     print(i,atoms.get_chemical_formula())
    # doped_dict={
    #             '3':{'Y':2,'vo':1},
    #             '6':{'Y':4,'vo':2},
    #             '9':{'Y':6,'vo':3},
    #             '12':{'Y':8,'vo':4},
    #             }
    
    # atoms=read('ZrO2.ortho-pnma.poscar')
    # try:
    #     os.mkdir('ortho-pnma')
    # except:
    #     pass
    # for k,v in doped_dict.items():
    #     cp=compostion(atoms)
    #     atoms_new=cp.oxide_doping('Zr','Y',v['Y'],v['vo'],(2,2,2))
    #     atoms2arc([atoms_new],'ortho-pnma/struc%s.arc'%k,has_energy=False)
    #     print(atoms_new.get_chemical_formula())
    #     # write('tetra/POSCAR.tetra.%s'%k,atoms_new)

    # for i in [3,6,9,12]:
    #     atoms=read('../%s/best.traj@-1'%i)
    #     print(atoms.get_chemical_formula())
    #     atoms2arc([atoms],'tril/struc%s.arc'%i,has_energy=False)
    from ase.visualize import view
    # atoms=read('test.traj')
    atoms=arc2atoms('struc.arc')[0]
    slab=lattice.slab_gen(atoms,(1,1,1),2)
    view(slab)
