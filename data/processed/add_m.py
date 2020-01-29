from rdkit import Chem
from rdkit.Chem import AllChem
from deepsmiles import Converter
f = open('ms-nmr.txt')
f1 = open('ms-m-nmr.txt','w')

for line in f:
    if '[' in line:
        smi = Converter(rings=True,branches=True).decode(line.split('],')[1])
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
        atoms = mol.GetAtoms()
        l = [a.GetSymbol() for a in atoms]
        cs = l.count('C')
        hs = l.count('H')
        os = l.count('O')
    #print(line,cs,hs,os)
        l1 = str(line.split(']')[0])+', '+str([cs,hs,os]).split('[')[1]+line.split(']')[1]
    else:
        l1 = line
    f1.write(l1)
