# -*- coding: utf-8 -*-
import torch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def new_data(file):
  ml = 0
  d = {}
  ind2w = {0: "SOS", 1: "EOS"}
  lets = 2
  f  = file
  mass = []
  mspec = []
  smi = []
  nmr = []
  atom = []
  ir = []
  c = 0
  for line in f:
    if line[0].isalpha() == True:
        c += 1
        smi.append(line.strip())
        for letter in line.strip():
          if letter not in d:
            d[letter] = int(lets)
            ind2w[lets] = letter
            lets += 1

        if len(line.strip()) > ml:
          ml = len(line.strip())
        else: 
          ml = ml
        if c > 1:
          mass.append(torch.tensor(mspec).cuda())
        mspec = []
    else:
       if len(line.split()) == 2 and not '[' in line:
        mspec.append((float(line.split()[0]),float(line.split()[1])))


       else:
         nmr.append((line.split("],")[0]).split("[")[1].split(",")[:-6]) 
         atom.append((line.split("],")[0]).split("[")[1].split(",")[-6:-3])
         ir.append((line.split("],")[0]).split("[")[1].split(",")[-3:])
  nmrs = [torch.tensor([float(h) for h in i]).unsqueeze(dim=0).cuda() for i in nmr]  
  atoms = [torch.tensor([float(k) for k in ii]).unsqueeze(dim=0).cuda() for ii in atom]
  irs = [torch.tensor([float(j) for j in iii]).unsqueeze(dim=0).cuda() for iii in ir]
  return mass, nmrs, atoms, irs, smi, ml, d, ind2w



f = open('../../data/processed/ms-m-ir-nmr.txt')

x,x2,x3,x4,y,ml,dic,ind2w = new_data(f)
#parche temporal por que no lee ultimo spec de mass
del x2[-1]
del x3[-1]
del x4[-1]
del y[-1]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules import *
import numpy as np

#Sets and Transformer classes taken from Juho Lee et al. except the DecoderRNN
class SmallDeepSet(nn.Module):
    def __init__(self, pool="max"):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=64),
        )
        self.dec = nn.Sequential(
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=1),
        )
        self.pool = pool

    def forward(self, x):
        x = self.enc(x)
        if self.pool == "max":
            x = x.max(dim=1)[0]
        elif self.pool == "mean":
            x = x.mean(dim=1)
        elif self.pool == "sum":
            x = x.sum(dim=1)
        x = self.dec(x)
        return x


class SmallSetTransformer(nn.Module):
    def __init__(self,):
        super().__init__()
        self.enc = nn.Sequential(
            SAB(dim_in=1, dim_out=64, num_heads=4),
            SAB(dim_in=64, dim_out=64, num_heads=4),
        )
        self.dec = nn.Sequential(
            PMA(dim=64, num_heads=4, num_seeds=1),
            nn.Linear(in_features=64, out_features=64),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=64, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))


class DecoderRNN(nn.Module):
        def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_length=ml):
                super(DecoderRNN, self).__init__()
                self.embed = nn.Embedding(vocab_size, embed_size)
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
                self.out = nn.Linear(hidden_size, vocab_size)
                self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, in_smiles ,hidden,cell):#
                embeddings = self.embed(in_smiles).view(1,1,-1)
                embeddings = F.relu(embeddings)
                output,(hidden, cell) = self.lstm(embeddings,(hidden,cell))
                outputs = self.softmax(self.out(output[0]))
                return outputs, hidden, cell



EMB=512
HID=512
LAY=4

VOC = len(ind2w)

def tensor_from_smiles(smiles_b):
        indexes = [dic[let] for let in smiles_b]
        indexes.append(int(1)) # 0 : EOS token
        in_smiles = torch.cuda.LongTensor(indexes)
        

        return in_smiles

def evaluate(encoder1, encoder2, encoder3, encoder4, decoder,x,x2,x3,x4,smi):

	loss2 = 0
	hidden1 = encoder1(x)
	temp = torch.randn(LAY,1,int(HID/4)).cuda()
	temp[0] = hidden1
	temp[1] = hidden1
	temp[2] = hidden1
	temp[3] = hidden1
	hidden1 = temp
	hidden2 = encoder2(x2)
	temp2 = torch.randn(LAY,1,int(HID/4)).cuda()
	temp2[0] = hidden2
	temp2[1] = hidden2
	temp2[2] = hidden2
	temp2[3] = hidden2
	hidden2 = temp2
	hidden3 = encoder3(x3)
	temp = torch.randn(LAY,1,int(HID/4)).cuda()
	temp3[0] = hidden3
	temp3[1] = hidden3
	temp3[2] = hidden3
	temp3[3] = hidden3
	hidden3 = temp3
	hidden4 = encoder4(x4)
	temp4 = torch.randn(LAY,1,int(HID/4)).cuda()
	temp4[0] = hidden4
	temp4[1] = hidden4
	temp4[2] = hidden4
	temp4[3] = hidden4
	hidden4 = temp4
 
	hidden = torch.cat((hidden1,hidden2,hidden3,hidden4),dim=-1) #
	cell = hidden
	pretarget = tensor_from_smiles(smi)
	target = torch.cuda.LongTensor([[0]])
	for di in range(len(pretarget)):
		output, hidden, cell = decoder(target,hidden,cell)			
		target = pretarget[di].unsqueeze(dim=0)
		loss2 += criterion(output, target)
	voss = loss2.item()/len(pretarget)
	return  voss

def evaval(encoder1, encoder2,encoder3,encoder4,decoder,val_pairs):
	voss=0
	for i in range(len(val_pairs[0])):
		X = torch.tensor(val_pairs[0][i]).cuda().unsqueeze(dim=0)
		X2 = torch.tensor(val_pairs[1][i]).cuda().unsqueeze(dim=-1)
		X3 = torch.tensor(val_pairs[2][i]).cuda().unsqueeze(dim=-1)
		X4 = torch.tensor(val_pairs[3][i]).cuda().unsqueeze(dim=-1)
		smi = val_pairs[4][i]
		voss += evaluate(encoder1,encoder2,encoder3,encoder4,decoder,X,X2,X3,X4,smi)
	voss_prom = voss/(len(val_pairs[0]))
	return voss_prom
def evaluateR(encoder1, encoder2,encoder3,encoder4,decoder,x,x2,x3,x4,max_length=ml): # puede cambiarse a numero muy grande
         hidden1 = encoder1(x)
         temp = torch.randn(LAY,1,int(HID/4)).cuda()
         temp[0] = hidden1
         temp[1] = hidden1
         temp[2] = hidden1
         temp[3] = hidden1
         hidden1 = temp
         hidden2 = encoder2(x2)
         temp2 = torch.randn(LAY,1,int(HID/4)).cuda()
         temp2[0] = hidden2
         temp2[1] = hidden2
         temp2[2] = hidden2
         temp2[3] = hidden2
         hidden2 = temp2
         hidden3 = encoder3(x3)
         temp3 = torch.randn(LAY,1,int(HID/4)).cuda()
         temp3[0] = hidden3
         temp3[1] = hidden3
         temp3[2] = hidden3
         temp3[3] = hidden3
         hidden3 = temp3
         hidden4 = encoder4(x4)
         temp4 = torch.randn(LAY,1,int(HID/4)).cuda()
         temp4[0] = hidden4
         temp4[1] = hidden4
         temp4[2] = hidden4
         temp4[3] = hidden4
         hidden4 = temp4


         hidden = torch.cat((hidden1,hidden2,hidden3,hidden4),dim=-1)
         cell = hidden
         target =torch.cuda.LongTensor([0])#0: SOS
         decoded_words = []
         for di in range(max_length):
                 output, hidden, cell = decoder(target,hidden,cell)
                 topv,topi = output.data.topk(1)
                 if int(topi[0][0]) == int(1): # EOS 
                         decoded_words.append('<EOS>')
                         break
                 else:
                         decoded_words.append(ind2w[int(topi[0][0])])
                 target = topi.squeeze().unsqueeze(dim=0)
         return decoded_words

def evaluateRandomly(encoder1, encoder2, encoder3,encoder4,decoder,ppair,n=50):
        for i in range(n):
                choice = random.randint(0,len(ppair)-1)
                xt = torch.tensor(ppair[0][choice]).cuda().unsqueeze(dim=0)
                x2t = torch.tensor(ppair[1][choice]).cuda().unsqueeze(dim=-1)
                x3t = torch.tensor(ppair[2][choice]).cuda().unsqueeze(dim=-1)
                x4t = torch.tensor(ppair[3][choice]).cuda().unsqueeze(dim=-1)

                smi = ppair[4][choice]
		
                output_words = evaluateR(encoder1,encoder2,encoder3,encoder4,decoder,xt,x2t,x3t,x4t)
                output_s = ''.join(output_words)
                print("pred:",output_s)
                print("real:",smi)
def evaluateTodo(encoder1,encoder2,encoder3,encoder4,decoder,ppair):
        for i in range(len(ppair[0])):
                X = torch.tensor(ppair[0][i]).cuda().unsqueeze(dim=0)
                X2 = torch.tensor(ppair[1][i]).cuda().unsqueeze(dim=-1)
                X3 = torch.tensor(ppair[2][i]).cuda().unsqueeze(dim=-1)
                X4 = torch.tensor(ppair[3][i]).cuda().unsqueeze(dim=-1)

                smi = ppair[4][i]
                output_words = evaluateR(encoder1,encoder2,encoder3,encoder4,decoder,X,X2,X3,X4)
                output_s = ''.join(output_words)
                print("pred:",output_s)
                print("real:",smi)

criterion = torch.nn.NLLLoss()

epochs = 160
X_train1, X_test1,X_train2,X_test2,X_train3,X_test3,X_train4,X_test4,y_train,y_test = train_test_split(x,x2,x3,x4,y,test_size=0.1,random_state=42,shuffle=True) # 
l_r = 0.00003
decoder = DecoderRNN(EMB,HID,VOC,LAY).cuda()
encoder1 = SetTransformer(2,1,int(HID/4)).cuda()
encoder2 = SetTransformer(1,1,int(HID/4)).cuda()
encoder3 = SetTransformer(1,1,int(HID/4)).cuda()
encoder4 = SetTransformer(1,1,int(HID/4)).cuda()
	
encoder_optimizer1 = optim.RMSprop(encoder1.parameters(),lr = l_r)
encoder_optimizer2 = optim.RMSprop(encoder2.parameters(),lr = l_r) 
encoder_optimizer3 = optim.RMSprop(encoder3.parameters(),lr = l_r)
encoder_optimizer4 = optim.RMSprop(encoder4.parameters(),lr = l_r)
decoder_optimizer = optim.RMSprop(decoder.parameters(),lr = l_r,momentum=0.2)
px = [] 
ppx = [] 
py=[]
ppy=[]
j = 0# iterations
jj = 0# iterations

x_tpair = [X_train1, X_train2,X_train3,X_train4,y_train]
val_pairs = [X_test1,X_test2,X_test3,X_test4,y_test]
b_size = len(X_train1)
for i in range(epochs):
	p = 0 # for plotting
	for b in range(b_size):   # batch
		j += 1 
		p += 1
		decoder_optimizer.zero_grad()
		encoder_optimizer1.zero_grad()
		encoder_optimizer2.zero_grad()
		encoder_optimizer3.zero_grad()
		encoder_optimizer4.zero_grad()
		loss = 0
		smis = y_train[b]
		X_train11 = X_train1[b].unsqueeze(dim=0)
		X_train22 = X_train2[b].unsqueeze(dim=-1)
		X_train33 = X_train3[b].unsqueeze(dim=-1)
		X_train44 = X_train4[b].unsqueeze(dim=-1)
		pretarget = tensor_from_smiles(smis)
		hidden1=encoder1(X_train11) # inithidden , features from encoder output
		temp = torch.randn(LAY,1,int(HID/4)).cuda()
		temp[0] = hidden1
		temp[1] = hidden1
		temp[2] = hidden1
		temp[3] = hidden1
		hidden1 = temp
	 

		hidden2=encoder2(X_train22) # inithidden , features from encoder output
		temp2 = torch.randn(LAY,1,int(HID/4)).cuda()
		temp2[0] = hidden2
		temp2[1] = hidden2
		temp2[2] = hidden2
		temp2[3] = hidden2
		hidden2 = temp2
	
		hidden3=encoder3(X_train33) # inithidden , features from encoder output
		temp3 = torch.randn(LAY,1,int(HID/4)).cuda()
		temp3[0] = hidden3
		temp3[1] = hidden3
		temp3[2] = hidden3
		temp3[3] = hidden3
		hidden3 = temp3

		hidden4=encoder4(X_train44) # inithidden , features from encoder output
		temp4 = torch.randn(LAY,1,int(HID/4)).cuda()
		temp4[0] = hidden4
		temp4[1] = hidden4
		temp4[2] = hidden4
		temp4[3] = hidden4
		hidden4 = temp4

		hidden = torch.cat((hidden1,hidden2,hidden3,hidden4),dim=-1) ###DIIIMMMMMMM?????
		cell = hidden
		target = torch.cuda.LongTensor([0])#0: SOS
		for s in range(len(pretarget)):
			output,hidden,cell  = decoder(target,hidden,cell)
			target = pretarget[s].unsqueeze(dim=0)
			loss += criterion(output, target)
		loss.backward()
		encoder_optimizer1.step()
		encoder_optimizer2.step()
		encoder_optimizer3.step()
		encoder_optimizer4.step()
  
		decoder_optimizer.step() 
		if p % 200 == 0: # plot every
			px.append(j)
			py.append(loss.item()/len(pretarget))

	print("validation set evaluation, epoch:",i+1)
	evaluateTodo(encoder1,encoder2,encoder3,encoder4,decoder,val_pairs)
	voss = evaval(encoder1,encoder2,encoder3,encoder4,decoder,val_pairs)
	print("val",i+1,j,voss)
	ppx.append(j)
	ppy.append(voss)
	plt.plot(px,py)
	plt.plot(ppx,ppy,marker='P')
	plt.xlabel('iterations')
	plt.ylabel('loss')
	plt.show()
	print("train set eval")
	print('epoch: ',i+1)
	evaluateRandomly(encoder1,encoder2,encoder3,encoder4,decoder,x_tpair,n=50)
#torch.save(encoder1.state_dict(),'enc1_w')
#torch.save(encoder2.state_dict(),'enc2_w')
#torch.save(encoder3.state_dict(),'enc3_w')
#torch.save(encoder4.state_dict(),'enc4_w')
#torch.save(decoder.state_dict(),'dec_w')

