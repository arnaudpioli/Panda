# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:36:46 2017

@author: Samorg
"""
from math import *
from Pearsoncoef import *
import numpy as np
import matplotlib.pyplot as plt


def noterMax(index,ref,exercise,filtreflex) :
    if ref==0:
        ref=0.0001
    #note=abs(float(filtreflex[index])-float(ref))/float(ref)
    note=1-sqrt(pow(filtreflex[index]-ref,2))/100
    if note>1 or note<0:       
        note=0
                                                                               
    return note              


def Corr_flex_rot(x,y,z,exercise,filtrerot,filtreflex):
    note=0
    note1=0
    for i in range(y-x):
        if filtrerot[x+i+1]>=filtrerot[x+i] and filtreflex[x+i+1]>=filtreflex[x+i]:
            note1=note1+1    
    for i in range(z-y):
        if filtrerot[y+i]<=filtrerot[y+i+1] and filtreflex[y+i]<=filtreflex[y+i+1]:
            note=note+1            
    return note+note1
    
def Symetrieflex(x,y,z,filtreflex):
    global symetrie
    diff=0
    flex=filtreflex
    if z-y>y-x:
        nb_point=y-x
    else:
        nb_point=z-y           
    partie1=list(reversed(filtreflex[y:y+nb_point]))
    partie2=filtreflex[y-nb_point:y]
    for i in range(len(partie1)):
        diff=abs(diff+partie1[i]-partie2[i])
    #symetrie=sqrt(pow(diff,2)/(sum(partie1)))
    #symetrie=diff*(sum(partie1)/sum(partie2))
    symetrie=pearson(partie1,partie2)
    return symetrie


def SymetrieRot(x,y,z,filtrerot):
    diff=0
    if z-y>y-x:
        nb_point=y-x
    else:
        nb_point=z-y
        
    global symetrie    
    partie1=list(reversed(filtrerot[y:y+nb_point]))
    partie2=filtrerot[y-nb_point:y]
    for i in range(len(partie1)):
        diff=diff+abs(partie1[i]-partie2[i])
    symetrie=diff/sum(partie1)
    symetrie=pearson(partie1,partie2)
    return symetrie


def variationsignal(x,y,z,dataS,data,mvt):
#    if dataS=='flex':
#        data=filtreflex
#    elif dataS=='rot':
#        data=filtrerot
    m=0
    s=0
    compt1=0
    compt2=0
    if mvt=="Squat" :
        for i in range(z-x-1):
            if data[i+x]>=data[i+x+1]:
                compt1=compt1+1
            #print(compt1)
        for i in range(y-z-1):
            if data[i+z]<=data[i+z+1]:
                compt2=compt2+1
        compt=compt1+compt2           
    elif mvt=="extention":
        for i in range(z-x-1):
            if data[i+x]<=data[i+x+1]:
                compt1=compt1+1
            #print(compt1)
        for i in range(y-z-1):
            if data[i+z]>=data[i+z+1]:
 
               compt2=compt2+1     

        compt=compt1+compt2           

    elif mvt=="prop":
        compt=0
        if data[x]-data[x+1]>0:
            m=1
        else:
            m=0
        for i in range(1,y-x-1):
            if i !=1:
                m=s
            
            if data[i+x]-data[i+x+1]>0:
                s=1
            else:
                s=0
                
                
            if s!=m:
                compt=compt+1            
       
       
       
       
    del compt1
    del compt2
    
    return compt


def splitbits(x):
    split=[bin] *4
    split[0]=(int(x)>> 6)&0x03
    split[1]=(int(x)>> 4)&0x03  
    split[2]=(int(x)>> 2) &0x03 
    split[3]=int(x)&0x03 
    return split
    


def searchmin_Avant(x,mvt,minnn,filtreflex):
    global solution
    global cond
    cond=0
    a=minnn
    for i in reversed(range(len(minnn))):
        cond=0
        if  x>a[i] and filtreflex[x]-filtreflex[a[i]]>15:
            solution=a[i]
            cond=0
            break            
        else :
            cond=1
        if i==0 and cond==1:
            #a.append(0)
            solution=0
            #print(a)
            cond=0
            break      
    del a       
    return solution   
    



def searchmin_Apres(x,minnn,filtreflex):  
    
    global sol
    for i in range(len(minnn)):        
        if minnn[i]>x and filtreflex[x]-filtreflex[minnn[i]]>15:
            sol=minnn[i]
            break
    return sol  
    



def trim_table(x):
    table=[]
    for i in range(len(x)-1):
        if x[i][4]==x[i+1][4] :
            table.append(i)
        if x[i][2]>x[i][4]:
            table.append(i)
    return table
    

def fourierTransform(x1,x2,x3,m,filtreflex,filtrerot):
    data=[]
    if x3=='flex':
        data=filtreflex
    elif x3=='rot':
        data=filtrerot
    Fs = 1000.0;  # sampling rate
    y=data[int(x1):int(x2)]-sum(data[int(x1):int(x2)])/len(data[int(x1):int(x2)])
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    #print(frq)
    frq = frq[range(n/2)]# one side frequency range
    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(n/2)]
    #print(len(Y))
    cc=abs(Y)
    cc=abs(Y)/max(cc)
 #   plt.plot(frq)
    plt.plot(cc)

    taux_bruit=0
    refFreq=10
    for i in range(len(frq)):
        if frq[i]>refFreq:
            taux_bruit=sum(cc[i:len(cc)])
            ref=1000000        
            break
    taux_bruit=taux_bruit/sum(cc)
    return taux_bruit
    
    
def searchMaxProp(x,mvt,minnn):
    global solution
    global cond
    cond=0
    a=minnn
    for i in reversed(range(len(minnn))):
        cond=0
        if  x>a[i] :
            solution=a[i]
            cond=0
            break            
        else :
            cond=1
        if i==0 and cond==1:
            #a.append(0)
            solution=0
            #print(a)
            cond=0
            break      
    del a       
    return solution       
    
    
        
#def getStabMvt(p1,p2,flex):
#    compt=0
#    tmp=p2-p1
#    v1=[]
#    v1.append(flex[p1])
#    vp=v1[0]
#    
#    for i in range(1,tmp):
#
#        va=flex[p1+i]
#        print("vp:   ",vp)
#        print("va:   ",va)
#
#        if va-vp>2 or va-vp<-2:
#            compt=compt+1
#        vp=va  #ligne inutile non?
#        v1.append(va)
#        vp=mean(v1)
#        #Ca c'est louche! C'est pas ce qu'on avait dit, tu devais limiter les valeurs
#        #à 50, pas rééditer le tableau arrivé à 50.
#        if i==50 or i==100 or i==150 or i==200 or i==250 or i==300:
#            del v1
#            v1=[]
#            v1.append(flex[p1+i])
#            vp=v1[0]
#            
#    plt.plot(v1)
#    return compt


#Par Arnaud################

def getStabMvt(p1,p2,flex):
    compt=0
    tmp=p2-p1
    v1=[]
    v1.append(flex[p1])
    vp=v1[0]
    

    i = 0
    while i < tmp:
        k = i % 50
        va = flex[p1+i]
        if va - vp >2 or va - vp < 2:
            compt = compt + 1
            
        v1[k] = va
        vp = mean(v1)
        i = i+1
        

    
    plt.plot(v1)
    return compt


###########################



    
    
#Memes comm que plus bas.    
def getFluidMvtprop(p1,p2,p3,p4,flex):
    tmp1=p2-p1
    tmp2=p4-p3
    compt1=0
    compt2=0
    s1=0
    s2=0
    if flex[p1]-flex[p1+1]>0:
        m1=1
    else:
        m1=0
    for i in range(1,tmp1):
        if i !=1:
            m1=s1
        
        if flex[i+p1]-flex[i+p1+1]>0:
            s1=1
        else:
            s1=0
            
            
        if s1!=m1:
            compt1=compt1+1
                


    if flex[p3]-flex[p3+1]>0:
        m2=1
    else:
        m2=0
    for i in range(1,tmp2):
        if i !=1:
            m2=s2
        
        if flex[i+p3]-flex[i+p3+1]>0:
            s2=1
        else:
            s2=0
            
            
        if s2!=m2:
            compt2=compt2+1
                
    ind=compt2+compt1
    return [compt2,compt1]

#Pourquoi tu divises le mouvement en plusieurs parties?    
def getFluidMvt(p1,p2,p3,flex):
    tmp1=p2-p1
    tmp2=p3-p2
    compt1=0
    compt2=0
    s1=0
    s2=0
    if flex[p1]-flex[p1+1]>0:
        m1=1
    else:
        m1=0
    for i in range(1,tmp1):
        if i !=1:
            m1=s1
        
        if flex[i+p1]-flex[i+p1+1]>0:
            s1=1
        else:
            s1=0
            
            
        if s1!=m1:
            compt1=compt1+1
                

    if flex[p2]-flex[p2+1]>0:
        m2=1
    else:
        m2=0
    for i in range(1,tmp2):
        if i !=1:
            m2=s2
        
        if flex[i+p2]-flex[i+p2+1]>0:
            s2=1
        else:
            s2=0
            
            
        if s2!=m2:
            compt2=compt2+1
                
#Ind n'intervient nulle part?
    return ind
        
#Par Arnaud#############
#A utiliser pour rot aussi dans l'indicateur!
def getFluidity(p1, p2, flex):
    
    tmp1=p2-p1
    compt1=0
    s1=0

    if flex[p1]-flex[p1+1]>0:
        m1=1
    else:
        m1=0

    for i in range(1,tmp1):
        if i !=1:
            m1=s1
        
        if flex[i+p1]-flex[i+p1+1]>0:
            s1=1
        else:
            s1=0
            
            
        if s1!=m1:
            compt1=compt1+1
        
    return compt1
########################
    
def getCoordination(p1,p2,flex,rot):
    tmp=p2-p1
    ind=0 
    for i in range(tmp-1):
        if (flex[p1+i+1]-flex[p1+i])<=0 and (rot[p1+i+1]-rot[p1+i])<=0:
            ind=ind+1
        elif (flex[p1+i+1]-flex[p1+i])>=0 and (rot[p1+i+1]-rot[p1+i])>=0:
            ind=ind+1
    return ind

#inutile. Ici tu fais juste (flex[fin]-flex[debut])/(fin-debbut)
#def getPuissance(p1,p2,flex):
#    tmp=p2-p1;
#    acc=[]
#    for i in range(tmp-1):
#        acc.append(flex[p1+i+1]-flex[p1+i])
#        
#    ind=mean(acc)
#    return ind    
    
#Par Arnaud#########################
def getPuissance(p4, p5, p6, p7, flex):

    ind1 = abs((flex[p5]-flex[p4])/(p5-p4))    
    ind2 = abs((flex[p7]-flex[p6])/(p7-p6))    
    ind = (ind1 + ind2)/2    
    
    return ind
####################################    
    
    
def getEnduranceSquat(indicators): #4 colonnes: fluid, coor, puiss, stab
    #for i in range(len(maximus):        
    ind = 0
    MF = indicators[0][0]
    MC = indicators[0][1]
    MP = indicators[0][2]        

    for i in range(0, len(indicators)-1):
        if indicators[i+1][0] - MF >0:   #fluidité
            ind = ind + 1
        if indicators[i+1][1] - MC >0:   #coordination
            ind = ind + 1
        if indicators[i+1][2] - MP <0:   #puissance
            ind = ind + 1
        
        MF = mean(indicators[0:i][0])
        MC = mean(indicators[0:i][1])
        MP = mean(indicators[0:i][2])
                

    ind = ind / len(indicators)

    return ind    

def getEnduranceProprio(indicators): #4 colonnes: fluid, coor, puiss, stab
    #for i in range(len(maximus):        
    ind = 0
    MS = indicators[0][3]   

    for i in range(0, len(indicators)-1):
        if indicators[i+1][3] - MS >0:   #stabilité
            ind = ind + 1
        
        MS = mean(indicators[0:i][3])

    ind = ind / len(indicators)

    return ind    



'''
ARNAUD
Pour moi, reste à voir:
//- Adapter algo Alfred à Faq. Ok mais il y a un autre pb.
//- Ajouter un indicateur d'erreur sur le PDF et en JS.
//- Calcul de p4, p5, p6, p7.
//- Endurance.
- Gérer la proprioception quand il n'y a aucune répétition [A AMELIORER]
- Passer à des indicateurs exo par exo
- Passer à des indicateurs séance par séance
- Test sur mes données de bilan.
'''






'''   
Stabilité fluidité 


stab dans la zone d'arret pour la proprio 
fulid la zone du debut et fin 





Cordination 
sense de derivé entre le flexion et rotation 



Puissance 
temps de monté


Endurance 



(S1,C1,P1,F1,A1,T1)----------------->(S5,C5,P5,F5,A5,T5)

S5-mean(S4-S1) 1 par 1
''' 
    
    
    