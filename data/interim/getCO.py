import re
f = open('ms-dsmi.csv')
ff = open('CO.csv','w')
c = 0
for line in f:
	if line[0].isdigit() == False:
		if not re.search('\[',line) and not re.search('I',line) and not re.search('l',line) and not re.search('B',line) and not re.search('S', line) and not re.search('P',line) and not re.search('N',line) and not re.search('F',line) and not re.search('n',line) and not re.search('p',line) and not re.search('s',line):
			#print(line)
			c = 1
		else:
			c = 0
	if c == 1:
		ff.write(line)
	
