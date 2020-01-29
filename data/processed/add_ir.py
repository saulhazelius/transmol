from rdkit import Chem
from rdkit.Chem import AllChem
from deepsmiles import Converter
f = open('ms-nmr.txt')
f1 = open('ms-m-ir-nmr.txt','w')
# alcohol, ether, carbonyl
for line in f:
    if '[' in line:
        smi = Converter(rings=True,branches=True).decode(line.split('],')[1])
        mol = Chem.MolFromSmiles(smi)
        mol = AllChem.AddHs(mol)
       # print(smi,mol.HasSubstructMatch(Chem.MolFromSmarts('[$(C[OX2H1]);!$([CX3](=O)[OX2H1])]')))
        print(smi,mol.HasSubstructMatch(Chem.MolFromSmarts('[$(c[OX2H1])]')))
       # print(smi,mol.HasSubstructMatch(Chem.MolFromSmarts('[$(C[OX2H1])&(c[OX2H1])]')))
        o = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[$(C[OX2H1])]')) == True or mol.HasSubstructMatch(Chem.MolFromSmarts('[$(c[OX2H1])]')) == True else 0
        e = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[o]([c])[c]')) == True or mol.HasSubstructMatch(Chem.MolFromSmarts('[O]([C])[C]')) == True else 0
        carb = 1 if mol.HasSubstructMatch(Chem.MolFromSmarts('[$([CX3]=[OX1]),$([CX3+]-[OX1-])]')) == True else 0
        atoms = mol.GetAtoms()
        l = [a.GetSymbol() for a in atoms]
        cs = l.count('C')
        hs = l.count('H')
        os = l.count('O')
    #print(line,cs,hs,os)
        l1 = str(line.split(']')[0])+', '+str([cs,hs,os,o,e,carb]).split('[')[1]+line.split(']')[1]
    else:
        l1 = line
    f1.write(l1)
