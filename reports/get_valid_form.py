from rdkit import Chem
from rdkit.Chem import AllChem
import re
import deepsmiles
f = open('allspec_90')

ccc =0 
for line in f:
    if re.search('pred',line):
        try:
            smi = deepsmiles.Converter(rings=True,branches=True).decode(line.split()[1].split('<')[0].strip())
        except:
            smi = None
        if smi:
            try:
                csmi = Chem.MolToSmiles(Chem.MolFromSmiles(smi))
            except:
                pass
            if csmi:
                l = []
                mm = Chem.MolFromSmiles(csmi)
                mm = AllChem.AddHs(mm)
                for at in mm.GetAtoms():
                    l.append(at.GetSymbol())
                    cs = l.count('C')
                    hs = l.count('H')
                    os = l.count('O')
                print('pred',csmi,cs,hs,os)
    if re.search('real',line):
        try:
            smi2 = deepsmiles.Converter(rings=True,branches=True).decode(line.split()[1].split('<')[0].strip())
        except:
            smi2 = None
        if smi2:
                    try:
                        csmi2 = Chem.MolToSmiles(Chem.MolFromSmiles(smi2))
                    except:
                        pass
                    if csmi2:
                        l2 = []
                        mm2 = Chem.MolFromSmiles(csmi2)
                        mm2 = AllChem.AddHs(mm2)
                        for at2 in mm2.GetAtoms():
                            l2.append(at2.GetSymbol())
                            cs2 = l2.count('C')
                            hs2 = l2.count('H')
                            os2 = l2.count('O')
                        print('real',csmi2, cs2, hs2, os2)

                    if csmi == csmi2:
                        ccc +=1
                    else:
                        print('fails ',csmi,csmi2)

print('Correct: ',ccc)

