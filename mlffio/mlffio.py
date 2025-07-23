#!/usr/bin/env python
import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import cellpar_to_cell
from ase.data import chemical_symbols
import math as m
from functools import reduce
import sys
from ase.io import read 

Eledict  = { 'H':1,     'He':2,   'Li':3,    'Be':4,   'B':5,     'C':6,     'N':7,     'O':8,
             'F':9,     'Ne':10,  'Na':11,   'Mg':12,  'Al':13,   'Si':14,   'P':15,    'S':16,
             'Cl':17,   'Ar':18,  'K':19,    'Ca':20,  'Sc':21,   'Ti':22,   'V':23,    'Cr':24,
             'Mn':25,   'Fe':26,  'Co':27,   'Ni':28,  'Cu':29,   'Zn':30,   'Ga':31,   'Ge':32,
             'As':33,   'Se':34,  'Br':35,   'Kr':36,  'Rb':37,   'Sr':38,   'Y':39,    'Zr':40,
             'Nb':41,   'Mo':42,  'Tc':43,   'Ru':44,  'Rh':45,   'Pd':46,   'Ag':47,   'Cd':48,
             'In':49,   'Sn':50,  'Sb':51,   'Te':52,  'I':53,    'Xe':54,   'Cs':55,   'Ba':56,
             'La':57,   'Ce':58,  'Pr':59,   'Nd':60,  'Pm':61,   'Sm':62,   'Eu':63,   'Gd':64, 
             'Tb':65,   'Dy':66,  'Ho':67,   'Er':68,  'Tm':69,   'Yb':70,   'Lu':71,   'Hf':72, 
             'Ta':73,   'W':74,   'Re':75,   'Os':76,  'Ir':77,   'Pt':78,   'Au':79,   'Hg':80, 
             'Tl':81,   'Pb':82,  'Bi':83,   'Po':84,  'At':85,   'Rn':86,   'Fr':87,   'Ra':88, 
             'Ac':89,   'Th':90,  'Pa':91,   'U':92,   'Np':93,   'Pu':94,   'Am':95,   'Cm':96, 
             'Bk':97,   'Cf':98,  'Es':99,   'Fm':100, 'Md':101,  'No':102,  'Lr':103,  'Rf':104, 
             'Db':105,  'Sg':106, 'Bh':107,  'Hs':108, 'Mt':109,  'Ds':110,  'Rg':111,  'Cn':112, 
             'Uut':113, 'Fl':114, 'Uup':115, 'Lv':116, 'Uus':117, 'UUo':118} 

Eletable = [ 'H'  ,   'He' ,  'Li' ,  'Be' ,  'B'  ,  'C'  ,  'N' ,  'O' , 
             'F'  ,   'Ne' ,  'Na' ,  'Mg' ,  'Al' ,  'Si' ,  'P' ,  'S' ,
             'Cl' ,   'Ar' ,  'K'  ,  'Ca' ,  'Sc' ,  'Ti' ,  'V' ,  'Cr',
             'Mn' ,   'Fe' ,  'Co' ,  'Ni' ,  'Cu' ,  'Zn' ,  'Ga',  'Ge',
             'As' ,   'Se' ,  'Br' ,  'Kr' ,  'Rb' ,  'Sr' ,  'Y' ,  'Zr',
             'Nb' ,   'Mo' ,  'Tc' ,  'Ru' ,  'Rh' ,  'Pd' ,  'Ag',  'Cd',
             'In' ,   'Sn' ,  'Sb' ,  'Te' ,  'I'  ,  'Xe' ,  'Cs',  'Ba',
             'La' ,   'Ce' ,  'Pr' ,  'Nd' ,  'Pm' ,  'Sm' ,  'Eu',  'Gd', 
             'Tb' ,   'Dy' ,  'Ho' ,  'Er' ,  'Tm' ,  'Yb' ,  'Lu',  'Hf', 
             'Ta' ,   'W'  ,  'Re' ,  'Os' ,  'Ir' ,  'Pt' ,  'Au',  'Hg', 
             'Tl' ,   'Pb' ,  'Bi' ,  'Po' ,  'At' ,  'Rn' ,  'Fr',  'Ra', 
             'Ac' ,   'Th' ,  'Pa' ,  'U'  ,  'Np' ,  'Pu' ,  'Am',  'Cm', 
             'Bk' ,   'Cf' ,  'Es' ,  'Fm' ,  'Md' ,  'No' ,  'Lr',  'Rf', 
             'Db' ,   'Sg' ,  'Bh' ,  'Hs' ,  'Mt' ,  'Ds' ,  'Rg',  'Cn', 
             'Uut',   'Fl' ,  'Uup',  'Lv' ,  'Uus',  'UUo', ] 

def reduceList( l ):
    return reduce( lambda a,b:a+b, l)

class Atom(object):
    def __init__(self, coord, ele, charge):
        self.cdnt = [x for x in coord]  # save coordination
        self.ele  = ele
        self.no   = Eledict[ele] 
        self.charge = charge

class Structure(object):
    def __init__(self, e): 
        self.atom = []
        self.e    = float(e)
        self.e_weight = 1.0
        self.f_weight = 1.0

    def addAtom(self, line, type=1): # type 1 => for arc file; 2 => Data file
        if type == 1:
            coord = [float(xa) for xa in line.split()[1:4]]
            ele   = line.split()[0]
            charge = float(line.split()[8])
        elif type == 2:
            coord = [float(xa) for xa in line.split()[2:5]]
            ele   = Eletable[int(line.split()[1])-1]
            charge = float(line.split()[5])
        if type == 3:
            coord = [float(xa) for xa in line.split()[1:4]]
            ele   = line.split()[0]
            charge = 0.0

        self.atom.append(Atom(coord, ele, charge))

    def addStress(self, line, type=1): # type 1 => for arc file; 2 => Data file
        if   type == 1: self.stress = [float(x) for x in line.split()]
        elif type == 2: self.stress = [float(x) for x in line.split()[1:7]]

    def addForce(self, line, iatom, type=1): # type 1 => for arc file; 2 => Data file
        if   type == 1: self.atom[iatom].force = [float(x) for x in line.split()]
        elif type == 2: self.atom[iatom].force = [float(x) for x in line.split()[2:5]]

    def addLat(self, line):
        self.abc = [float(l) for l in line.split()[1:7]]
        a, b, c  = self.abc[0:3]
        alpha, beta, gamma = [x*np.pi/180.0 for x in self.abc[3:]]

        bc2 = b**2 + c**2 - 2*b*c*m.cos(alpha)
        h1 = a
        h2 = b * m.cos(gamma)
        h3 = b * m.sin(gamma)
        h4 = c * m.cos(beta)
        h5 = ((h2 - h4)**2 + h3**2 + c**2 - h4**2 - bc2)/(2 * h3)
        h6 = m.sqrt(c**2 - h4**2 - h5**2)
        self.lat = [[h1, 0., 0.], [h2, h3, 0.], [h4, h5, h6]]
        self.nlat = np.array(self.lat)

    def calAtomNum(self, ):
        self.natm    = len(self.atom)
        self.eleNum, self.ntpatom = list(np.unique(
                 [atom.no for atom in self.atom], return_counts=True))
        self.eleList  = [Eletable[ele-1] for ele in self.eleNum]
        self.countEle = dict( (Eletable[ele-1],no) for ele,no in zip(self.eleNum, self.ntpatom) )

    def sortAtom(self, ):
        self.atom.sort(key=lambda atom: atom.no)

    def printStr(self, fout, istr=0, sort_ele=True):
        fout.write("\t\t\t\tEnergy\t%8d        0.0000 %17.6f\n"%(istr, self.e))
        fout.write("!DATE\n")
        fout.write("PBC%15.9f%15.9f%15.9f%15.9f%15.9f%15.9f\n"%(
                    self.abc[0], self.abc[1], self.abc[2],
                    self.abc[3], self.abc[4], self.abc[5]))
        if sort_ele: self.sortAtom()
        for iatom,atom in enumerate(self.atom):
            fout.write("%-2s%18.9f%15.9f%15.9f CORE %4d %-2s %-2s %8.4f %4d\n"%(
                        atom.ele, atom.cdnt[0], atom.cdnt[1], atom.cdnt[2],
                        iatom+1, atom.ele, atom.ele, atom.charge, iatom+1))
        fout.write("end\nend\n")

class Set(list):
    def __init__(self, fname, forname=False, noE=False, newform=False):
        list.__init__(self)
        if newform: return
        self.fname = fname

        currentStr = -1; numpath = 0; self.pathway = []
        for line in open(fname):
            # add structure
            if not noE:
                if 'Energy' in line or 'IS' in line or 'FS' in line or ('SSW' in line and 'Str' not in line):
                    currentStr += 1
                    if '****' not in line: 
                        self.append(Structure(line.split()[3]))
                        self[currentStr].conv = float(line.split()[2])
                        try :
                            self[currentStr].pathid = int(line.split()[1])
                        except:
                            self[currentStr].pathid = 0
                    else:                  
                        self.append(Structure(0.0))
                        self[currentStr].conv = 9999.0
                    self[currentStr].id = currentStr
                elif 'Str' in line:
                    if '****' not in line: self.append(Structure(line.split()[4]))
                    else:                  self.append(Structure(0.0))
                    currentStr += 1
                    self[currentStr].nminimum = int(line.split()[1])
                    self[currentStr].numstr   = int(line.split()[2]) 
                elif 'React' in line or 'TS' in line:
                    currentStr += 1
                    self.append(Structure(line.split()[4]))
                    self[currentStr].numstr = int(line.split()[2])
                    self[currentStr].nopathway = int(line.split()[1])
                    if 'React' in line: self[currentStr].ts = False
                    else:               self[currentStr].ts = True
                elif 'CAR File' in line:
                    self.append(Structure(0.0))
                    currentStr += 1
            else:
                if 'DATE' in line:
                    self.append(Structure(0.0))
                    currentStr += 1

            # complete infomation in single structure
            if 'PBC' in line and 'ON' not in line and 'OFF' not in line:
                self[currentStr].addLat(line)
            elif 'CORE' in line:
                self[currentStr].addAtom(line)

        # read force information
        if forname:
            currentStr = -1
            for line in open(forname):
                if "For" in line:
                    currentStr += 1
                    iatom = 0
                    for atom in self[currentStr].atom: atom.force = [0.0, 0.0, 0.0]
                elif len(line.split()) == 6:
                    self[currentStr].addStress(line)
                elif len(line.split()) == 3:
                    if "****" not in line: self[currentStr].addForce(line, iatom)
                    else:                  self[currentStr].addForce('0.0 0.0 0.0', iatom)
                    iatom += 1

        self.numstr = currentStr + 1
        for istr in range(self.numstr): 
            self[istr].calAtomNum()
            self[istr].id = istr

    def genDataStr(self, set, fname):
        """ generate new version of Data_train file"""

        with open(fname,'w') as fout:
            for istr in set:
                astr = self[istr]
                astr.sortAtom()
                fout.write(" Start one structure\n")
                fout.write("Energy is %12.6f eV\n"%(astr.e))
                fout.write("total number of element is %5d\n"%astr.natm)
                fout.write("element in structure:\n")
                fout.write("symbol %s\n"%reduceList(["%4s"%s   for s   in astr.eleList]))
                fout.write("No.    %s\n"%reduceList(["%4d"%num for num in astr.eleNum]))
                fout.write("number %s\n"%reduceList(["%4d"%num for num in astr.ntpatom]))
                fout.write("weight %9.3f  %9.3f\n"%(astr.e_weight, astr.f_weight))
                for lat in astr.lat:
                    fout.write("lat %15.8f  %15.8f  %15.8f\n"%(lat[0], lat[1], lat[2]))
                for atom in astr.atom:
                    fout.write("ele %4s %15.8f  %15.8f  %15.8f  %15.8f\n"%(
                           atom.no, atom.cdnt[0], atom.cdnt[1], atom.cdnt[2], atom.charge))
                fout.write(" End one structure\n\n")

    def genDataForce(self, set, fname):
        """ generate new version of Data_force file, must write str.txt first"""
        with open(fname,'w') as fout: 
            for istr in set:
                astr = self[istr]
                fout.write(" Start one structure\n") 
                fout.write("stress %15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n"%(
                    astr.stress[0], astr.stress[1], astr.stress[2], 
                    astr.stress[3], astr.stress[4], astr.stress[5]))
                for atom in astr.atom:
                    fout.write("force %4d %15.8f %15.8f %15.8f\n"%(
                        atom.no, atom.force[0], atom.force[1], atom.force[2]))
                fout.write(" End one structure\n\n")

    def genArc(self, set, fname, press=False, sort_ele=True):
        fout = open(fname, 'w')
        fout.write("!BIOSYM archive 2\nPBC=ON\n")
        for istr in set:
            self[istr].printStr(fout, istr=istr, sort_ele=sort_ele)
        fout.close()

    def genAllfor(self, set, fname, flag=False):
        fout = open(fname, "w")
        for istr in set:
            str = self[istr]
            if flag:
                fout.write(" For   %d  %d  SS-fixlat   %12.6f\n"%(str.nminimum, str.numstr, str.e))
            else:
                fout.write(" For   0  0  SS-fixlat   %12.6f\n"%(str.e))

            fout.write("%15.8f %15.8f %15.8f %15.8f %15.8f %15.8f\n"%( 
                str.stress[0], str.stress[1], str.stress[2], 
                str.stress[3], str.stress[4], str.stress[5]))
            for atom in str.atom:
                fout.write("%15.8f %15.8f %15.8f\n"%(atom.force[0], atom.force[1], atom.force[2]))
            fout.write("\n ")
        fout.close()

class trainSet(Set, list):
    def __init__(self, strname, forname=False):
        list.__init__(self)

        # process coordination
        currentStr = -1
        for line in open(strname):
            # add structure
            if 'Start' in line:
                currentStr += 1
            elif 'Energy' in line:
                self.append(Structure(line.split()[2]))
                self[currentStr].lat = []
                self[currentStr].nminimum = 1
                self[currentStr].numstr   = currentStr
            elif 'lat' in line:
                self[currentStr].lat.append([float(item) for item in line.split()[1:4]])
            elif 'ele' in line and 'element' not in line:
                self[currentStr].addAtom(line, type=2)
        
        self.numstr = currentStr + 1
        for istr in range(self.numstr): 
            self[istr].calAtomNum()
            self[istr].nlat=np.array(self[istr].lat)
            self.lat2abc(istr)
            self[istr].id = istr

        # process force
        if forname:
            currentStr = -1
            for line in open(forname):
                if 'Start' in line:
                    currentStr += 1
                    iatom = 0
                elif 'stress' in line:
                    self[currentStr].addStress(line, 2)
                elif 'force'  in line:
                    self[currentStr].addForce(line, iatom, 2)
                    iatom += 1

    def lat2abc(self, i):
        a = np.linalg.norm(self[i].lat[0])
        b = np.linalg.norm(self[i].lat[1])
        c = np.linalg.norm(self[i].lat[2])
        alpha = m.acos(np.dot(self[i].lat[1],self[i].lat[2]) / (b*c))*180.0/np.pi
        beta  = m.acos(np.dot(self[i].lat[0],self[i].lat[2]) / (a*c))*180.0/np.pi
        gamma = m.acos(np.dot(self[i].lat[0],self[i].lat[1]) / (a*b))*180.0/np.pi
        self[i].abc = [a,b,c,alpha,beta,gamma]

def arc2trainset(structure_file='allstr.arc',force_file='allfor.arc'):
    arcfile=Set(structure_file,force_file)
    arcfile.genDataStr(range(arcfile.numstr),'TrainStr.txt')
    arcfile.genDataForce(range(arcfile.numstr),'TrainFor.txt')

def trainset2arc(train_arc='TrainStr.txt',train_for='TrainFor.txt'):
    arcfile=trainSet(train_arc,train_for)
    arcfile.genArc(range(arcfile.numstr),'allstr.arc')
    arcfile.genAllfor(range(arcfile.numstr),'allfor.arc')


def arc2atoms(structure_file, force_file=None):
    with open(structure_file, 'r') as f:
        structures,energies = parse_dmol_arc_lines(f.readlines())

    if force_file is not None:
        try:
            with open(force_file, 'r') as f_for:
                force_data = parse_force_file(f_for.readlines())
            
            for idx, atoms in enumerate(structures):
                if idx < len(force_data):
                    fd = force_data[idx]
                    if len(fd['forces']) == len(atoms): 
                        atoms.calc = SinglePointCalculator(
                                                            atoms,
                                                            energy=fd['energy'],
                                                            forces=fd['forces'],
                                                            stress=fd['stress']
                                                        )
                    else:
                        print(f"the atoms number ({len(atoms)}) in structure {idx} not equal to force ({len(fd['forces'])})")

        except Exception as e:
            print(e)
    else:
        for idx,atoms in enumerate(structures):
            atoms.calc = SinglePointCalculator(
                                                atoms,
                                                energy=energies[idx]
                                            )

    return structures

def atoms2arc(atoms,arcfile,has_energy=True,write_force=False):
    if has_energy:
        energy=[atom.get_potential_energy() for atom in atoms]
    else:
        energy=[0.0]

    fout = open(arcfile, 'w')
    fout.write("!BIOSYM archive 2\nPBC=ON\n")
    index=0
    for atom_i in atoms:
        abc = lat2abc(atom_i.cell[:])
        ele = atom_i.get_chemical_symbols()
        pos = atom_i.get_positions()
        fout.write("\t\t\t\tEnergy\t%8d        0.0000 %17.6f\n" % (0, energy[index]))
        fout.write("!DATE\n")
        fout.write("PBC%15.9f%15.9f%15.9f%15.9f%15.9f%15.9f\n" % (
                abc[0], abc[1], abc[2],abc[3], abc[4], abc[5]))
        for iatom, atom in enumerate(ele):
            fout.write("%-2s%18.9f%15.9f%15.9f CORE %4d %-2s %-2s %8.4f %4d\n" % (
                    atom, pos[iatom][0], pos[iatom][1],pos[iatom][2],
                    iatom + 1, atom, atom, 0, iatom + 1))
        fout.write("end\nend\n")
        index+=1
    fout.close()
    if write_force:
        atoms2for(atoms)

def parse_dmol_arc_lines(lines):
    if lines[1].startswith('PBC=ON'):
        PBC=True
    else:
        PBC=False
    images = []
    energies = []
    i = 2
    while i < len(lines):
        if lines[i].split()[0]=='Energy':
            try:
                energy = float(lines[i].split()[-1])
            except:
                energy = float(lines[i].split()[-2])
            energies.append(energy)

            # cell information
            if i+2 < len(lines) and lines[i+2].startswith('PBC'):
                cell_params = list(map(float, lines[i+2].split()[1:7]))
                cell = cellpar_to_cell(cell_params)
                i += 3
            # atoms
            symbols, positions = [], []
            while i < len(lines) and not lines[i].startswith('end'):
                parts = lines[i].split()
                if len(parts) >= 4:
                    try:
                        symbol = parts[0].capitalize()
                        if symbol in chemical_symbols:
                            pos = list(map(float, parts[1:4]))
                            symbols.append(symbol)
                            positions.append(pos)
                    except Exception:
                        raise ValueError("Parse for atoms informations failed")
                i += 1
            # store
            images.append(Atoms(
                symbols=symbols,
                positions=positions,
                cell=cell,
                pbc=PBC
            ))
        i += 1
    return images,energies

def parse_force_file(lines):
    data = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # match "For           0  0           -211.2902374" lines
        if line.startswith('For') and len(line.split()) >= 3:
            try:
                energy = float(line.split()[-1])
                # match stress "-0.4040285E-01 -0.3352130E-05 ..." lines
                stress_line = lines[i+1].strip()
                s = list(map(float, stress_line.split()[:6]))
                stress = np.zeros((3, 3))
                stress[0,0] = s[0]; stress[1,1] = s[1]; stress[2,2] = s[2]  # 对角
                stress[1,2] = stress[2,1] = s[3]  # yz
                stress[0,2] = stress[2,0] = s[4]  # xz
                stress[0,1] = stress[1,0] = s[5]  # xy
                # stress *= -0.1  # Hartree/Bohr³ → GPa

                i += 2
                forces = []
                while i < len(lines):
                    if lines[i].strip().startswith("For"):  
                        break
                    f_line = lines[i].strip()
                    if f_line:
                        try:
                            forces.append(list(map(float, f_line.split()[:3])))
                        except ValueError:
                            print(f"忽略非法力数据行: {lines[i]}")
                    i += 1

                data.append({
                    'energy': energy,
                    'stress': stress,
                    'forces': np.array(forces)
                })
            except:
                raise ValueError("Parse for force informations failed")
            
        else:
            i += 1
    return data

def lat2abc(lat):
    nlat = np.array(lat)
    a = np.linalg.norm(lat[0])
    b = np.linalg.norm(lat[1])
    c = np.linalg.norm(lat[2])
    alpha = m.acos(np.dot(lat[1], lat[2]) / (b * c)) * 180.0 / np.pi
    beta = m.acos(np.dot(lat[0], lat[2]) / (a * c)) * 180.0 / np.pi
    gamma = m.acos(np.dot(lat[0], lat[1]) / (a * b)) * 180.0 / np.pi
    return [a, b, c, alpha, beta, gamma]

def atoms2for(atoms,forfile="allfor.arc"):
    fout=open(forfile,'w')
    for atom_i in atoms:
        stress=-(atom_i.get_stress())
        energy=atom_i.get_potential_energy()
        fout.write("\tFor\t%8d        0.0000 %17.7f\n" % (0, energy))
        fout.write("%16.7E%16.7E%16.7E%16.7E%16.7E%16.7E\n" % (
                stress[0],stress[1],stress[2],stress[3],stress[4],stress[5]))
        for force in atom_i.get_forces():
            fout.write("\t\t\t%16.10f%16.10f%16.10f\n" %(force[0],force[1],force[2]))
        fout.write('\n')


if __name__ == "__main__":
    # atoms = arc2atoms("allstr.arc",force_file="allfor.arc"  )
    # atoms=read('all.traj',':')
    # atoms2arc(atoms,"allstr.arc",write_force=True)
    trainset2arc()
    arc2trainset()