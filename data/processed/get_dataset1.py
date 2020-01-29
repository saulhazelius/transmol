f1 = open('../interim/nmr-smi.txt')
f2 = open('../interim/CO.csv')
f3 = open('ms-nmr.txt','w')
uno =[]
dos = []
for line in f1:
    
    uno.append(line.strip())

for line2 in f2:
#    if len(line2.split()) == 1:
     dos.append(line2.strip())
s = 0
for k in uno:
    for j in dos:
        pat = str(k.split('],')[1])
        if len(j.split()) == 1:
            if pat ==j:
         #       print(pat)
          #      print(j)
                s = 1
                f3.write(k+'\n')
            else:
                s = 0
        if s == 1:
            f3.write(j+'\n')

