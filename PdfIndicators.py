# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 16:59:56 2017
Git
@author: Samorg
"""
#################################################################################################
#////////////////////////////////////////////Load Modules\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
import time
import copy
tmps1=time.clock()
import sys,json
#sys.path.append('C:\Users\Arnaud\Documents\SmartR\Algos server')
from bson import ObjectId
from functions import *
import pymongo
import pprint
from detect_peaks import detect_peaks
from Pearsoncoef import *
from math import *
import pandas as pd
import textwrap
import csv
import numpy as np
from scipy.signal import savgol_filter
from math import *
import matplotlib.pyplot as plt
import cStringIO
import smtplib
from email.MIMEMultipart import MIMEMultipart
from email.MIMEText import MIMEText
from email.MIMEBase import MIMEBase
from email import encoders
##PDf Libraries 
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader
import scipy.signal as scs

def Alfred(file):
    global MSV, CSV
    global Tremble
    global PHI
    def redimA(a):
        return a
    def redimG(w):
        return w*np.pi/180# pour passage en rad/s
    MA=[[redimA(file[i][0]),redimA(file[i][1]),redimA(file[i][2])] for i in range(np.shape(file)[0])]
    MG=[[redimG(file[i][3]),redimG(file[i][4]),redimG(file[i][5])] for i in range(np.shape(file)[0])]
    CA=[[redimA(file[i][6]),redimA(file[i][7]),redimA(file[i][8])] for i in range(np.shape(file)[0])]
    CG=[[redimG(file[i][9]),redimG(file[i][10]),redimG(file[i][11])] for i in range(np.shape(file)[0])]
    def biaismoyen(D,limite):
        sommeD=0.
        for i in range(limite):
            sommeD+=D[i]
        return sommeD/float(limite)    
    lim=100
    BiaisMGx,BiaisMGy,BiaisMGz=biaismoyen([MG[i][0] for i in range(len(MG))],lim),biaismoyen([MG[i][1] for i in range(len(MG))],lim),biaismoyen([MG[i][2] for i in range(len(MG))],lim)
    MG=[[i[0]-BiaisMGx,i[1]-BiaisMGy,i[2]-BiaisMGz] for i in MG]
    BiaisCGx,BiaisCGy,BiaisCGz=biaismoyen([CG[i][0] for i in range(len(CG))],lim),biaismoyen([CG[i][1] for i in range(len(CG))],lim),biaismoyen([CG[i][2] for i in range(len(CG))],lim)
    CG=[[i[0]-BiaisCGx,i[1]-BiaisCGy,i[2]-BiaisCGz] for i in CG]
    #Sauvegarde pour différentiation mouvement/tremblement
    MABrute=MA
    CABrute=CA
    #filtre de Savitzky-Golay, implémenté avec scipy.signal
    paslissage=31 #impair, au dessus de 31 (ordre2) atténue les amplitudes d'oscillation
    ordrelissage=2
    MAx,MAy,MAz=[MA[i][0] for i in range(len(MA))],[MA[i][1] for i in range(len(MA))],[MA[i][2] for i in range(len(MA))]
    CAx,CAy,CAz=[CA[i][0] for i in range(len(CA))],[CA[i][1] for i in range(len(CA))],[CA[i][2] for i in range(len(CA))]
    MAx,MAy,MAz=scs.savgol_filter(MAx,paslissage,ordrelissage),scs.savgol_filter(MAy,paslissage,ordrelissage),scs.savgol_filter(MAz,paslissage,ordrelissage)
    CAx,CAy,CAz=scs.savgol_filter(CAx,paslissage,ordrelissage),scs.savgol_filter(CAy,paslissage,ordrelissage),scs.savgol_filter(CAz,paslissage,ordrelissage)
    MA=[[MAx[i],MAy[i],MAz[i]] for i in range(len(MA))]
    CA=[[CAx[i],CAy[i],CAz[i]] for i in range(len(CA))]
    
    """Étape 2-Choix des données pour le calcul, pour les deux modules"""
    deltat=0.017
    seuil=0.1
    def theta(gx,gz,seuil):
        if (np.abs(gz)<seuil):
            return 0
        else:
            return np.arctan(gx/gz)
    Mtheta=[theta(i[0],i[2],seuil) for i in MA]
    MTheta=np.mean(Mtheta[:lim]) #peu dépendant des variations locales de l'angle.
    MGomega=[-MG[i][0]*np.cos(MTheta)+MG[i][2]*np.sin(MTheta) for i in range(len(MG))]
    MGpoint=[0]+[(MGomega[i+1]-MGomega[i])/deltat for i in range(len(MGomega)-1)]
    Ctheta=[theta(i[0],i[2],seuil) for i in CA]
    CTheta=np.mean(Ctheta[:lim])
    CGomega=[-CG[i][0]*np.cos(CTheta)+CG[i][2]*np.sin(CTheta) for i in range(len(CG))]
    CGpoint=[0]+[(CGomega[i+1]-CGomega[i])/deltat for i in range(len(CGomega)-1)]
    MS=[[MA[i][2]*np.cos(MTheta)+MA[i][0]*np.sin(MTheta),-MA[i][1],MGomega[i],MGpoint[i]] for i in range(len(MA))]
    CS=[[CA[i][2]*np.cos(CTheta)+CA[i][0]*np.sin(CTheta),-CA[i][1],CGomega[i],CGpoint[i]] for i in range(len(CA))]
    
    """Étape 3-Calcul ou attribution des données anthropométriques"""
    #Déterminé pour la genouillère blanche des données velopanda2 sur moi
    Mr=0.055#mètres
    Malpha=40.*np.pi/180 #radians
    Mbeta=40.*np.pi/180 #radians
    Cr=0.105 #mètres
    Calpha=25.*np.pi/180 #radians
    Cbeta=30.*np.pi/180 #radians
    
    """Étape 4-Calcul des valeurs virtuelles pour les deux modules"""
    def rot(theta,vect):
        return [np.cos(theta)*vect[0]-np.sin(theta)*vect[1],np.sin(theta)*vect[0]+np.cos(theta)*vect[1]]
    def somme(vect1,vect2):
        return [vect1[0]+vect2[0],vect1[1]+vect2[1]]
    MSV=[rot(-Mbeta,somme(rot(Malpha,i[:2]),[-Mr*(i[2]**2),Mr*i[3]])) for i in MS]
    CSV=[rot(-Cbeta,somme(rot(Calpha,i[:2]),[-Cr*(i[2]**2),Cr*i[3]])) for i in CS]
    
    """Étape 5-Calcul de l'angle de flexion après passage en représentation polaire et affichage"""
    def carttopol(vect):
        norme=np.linalg.norm(vect)
        cos=vect[0]/norme
        sin=vect[1]/norme
        if(cos>=0):
            return [norme,np.arcsin(sin)*180/np.pi]
        elif(sin>=0):
            return [norme,np.arccos(cos)*180/np.pi]
        else:
            return [norme,(np.arcsin(-sin)+np.pi)*180/np.pi]
    def flexion(MSV,CSV):
        a=360+carttopol(MSV)[1]-carttopol(CSV)[1]
        if (a<200):
            return a
        else:
            a=a-360
            if (a<200):
                return a
            else:
                return a-360
            
    PHI=[flexion(MSV[i],CSV[i]) for i in range(len(MSV))]
    
    """Étape 7-Calcul des "mouvement" et "tremblements" """
    paslissage=11 #impair, au dessus de 31 (ordre2) atténue les amplitudes d'oscillation
    ordrelissage=2
    MABrutex,MABrutey,MABrutez=[MABrute[i][0] for i in range(len(MABrute))],[MABrute[i][1] for i in range(len(MABrute))],[MABrute[i][2] for i in range(len(MABrute))]
    CABrutex,CABrutey,CABrutez=[CABrute[i][0] for i in range(len(CABrute))],[CABrute[i][1] for i in range(len(CABrute))],[CABrute[i][2] for i in range(len(CABrute))]
    MABrutex,MABrutey,MABrutez=scs.savgol_filter(MABrutex,paslissage,ordrelissage),scs.savgol_filter(MABrutey,paslissage,ordrelissage),scs.savgol_filter(MABrutez,paslissage,ordrelissage)
    CABrutex,CABrutey,CABrutez=scs.savgol_filter(CABrutex,paslissage,ordrelissage),scs.savgol_filter(CABrutey,paslissage,ordrelissage),scs.savgol_filter(CABrutez,paslissage,ordrelissage)
    MABrute=[[MABrutex[i],MABrutey[i],MABrutez[i]] for i in range(len(MABrute))]
    CABrute=[[CABrutex[i],CABrutey[i],CABrutez[i]] for i in range(len(CABrute))]
    MSBrute=[[MABrute[i][2]*np.cos(MTheta)+MABrute[i][0]*np.sin(MTheta),-MABrute[i][1],MGomega[i],MGpoint[i]] for i in range(len(MABrute))]
    CSBrute=[[CABrute[i][2]*np.cos(CTheta)+CABrute[i][0]*np.sin(CTheta),-CABrute[i][1],CGomega[i],CGpoint[i]] for i in range(len(CABrute))]
    MSVBrute=[rot(-Mbeta,somme(rot(Malpha,i[:2]),[-Mr*(i[2]**2),Mr*i[3]])) for i in MSBrute]
    CSVBrute=[rot(-Cbeta,somme(rot(Calpha,i[:2]),[-Cr*(i[2]**2),Cr*i[3]])) for i in CSBrute]
    PHIBrute=[flexion(MSVBrute[i],CSVBrute[i]) for i in range(len(MSVBrute))]
    def diff(v1,v2):
        return[v1[0]-v2[0],v1[1]-v2[1],v1[2]-v2[2]]
    TREMBLACC=[np.linalg.norm(diff(MA[i],MABrute[i]))+np.linalg.norm(diff(CA[i],CABrute[i])) for i in range(len(MA))]
    return PHI,TREMBLACC

def getTime():
    DataTime=np.zeros([len(hexdata),1])
    DataTime[0]=0
    DeltaT=(hexdata[10][16]-hexdata[0][16])/11
    
    for i in range(1,11):
        DataTime[i]=DataTime[i-1]+DeltaT
    for i in range(11,len(hexdata)):
        DeltaT=(hexdata[i][16]-hexdata[i-10][16])/11
        DataTime[i]=DataTime[i-1]+DeltaT

    
    
    
    return DataTime


def getAleatoireSeries():
    global filtreflex
    if len(filtreflex) < 500 :
        Series=np.zeros([1,2])
        Series[0][0]=0
        Series[0][1]=len(filtreflex)   
             
    elif len(filtreflex) > 0 and  len(filtreflex) <1000 and  len(filtreflex) >=750 :
        Series=np.zeros([1,2])
        Series[0][0]=250
        Series[0][1]=750   
          
    elif len(filtreflex) <750 and len(filtreflex) >500 :
        Series=np.zeros([1,2])
        Series[0][0]=0
        Series[0][1]=500
        
    elif len(filtreflex) > 1000 and  len(filtreflex) <2000 and  len(filtreflex) >1500 :
        Series=np.zeros([1,2])
        Series[0][0]=750
        Series[0][1]=1250

    elif len(filtreflex) > 1000 and  len(filtreflex) <2000 and  len(filtreflex) <1500 :
        Series=np.zeros([1,2])
        Series[0][0]=1000-250
        Series[0][1]=len(filtreflex)        
                
    elif len(filtreflex) >= 2000 and len(filtreflex) <3000 :
        Series=np.zeros([2,2])
        Series[0][0]=250
        Series[0][1]=750
        Series[1][0]=1500
        Series[1][1]=2000            
    
    elif len(filtreflex) >= 3000 and  len(filtreflex) <4000 :
        Series=np.zeros([2,2])
        Series[0][0]=250
        Series[0][1]=750
        Series[1][0]=2500
        Series[1][1]=3000    
    elif len(filtreflex) >= 4000 and  len(filtreflex) <5000 :
        Series=np.zeros([3,2])
        Series[0][0]=250
        Series[0][1]=750
        Series[1][0]=1250
        Series[1][1]=1750                    
        Series[2][0]=3500
        Series[2][1]=4000                   
    elif len(filtreflex) >5000 :
        Series=np.zeros([4,2])
        Series[0][0]=250
        Series[0][1]=750
        Series[1][0]=1250
        Series[1][1]=1750                    
        Series[2][0]=3500
        Series[2][1]=4000  
        Series[3][0]=4200
        Series[3][1]=5000 
    return Series 

def extentionAnalyse():
    exercise="extention"         
    maxx=detect_peaks(filtreflex,mph=-50,mpd=20,edge='rising',valley='true')
    minn=detect_peaks(filtreflex,mph=40,mpd=30,edge='rising')
    k=0
    maxxx=[]
    minnn=[]
#################################################################################################
#////////////////////////////Filtrage minimal des mins et MAx\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################    
    for i in range(0,len(maxx-k)):
        if flex[maxx[i]]<70:
            maxxx.append(maxx[i])
    for i in range(0,len(minn)):
        if flex[minn[i]]>-20:
           minnn.append(minn[i])            
    minlen=len(minnn)
    mouvements=[]
    size=0
    maximus=np.zeros([len(maxxx),11])     
    maximus=getMaximus()      
    maximus=maximus[maximus[:, 6] <150]
    maximus=maximus[maximus[:, 6] >10]
    maximus=maximus[maximus[:, 3]-maximus[:, 0] > 15]
    maximus=maximus[maximus[:, 3]-maximus[:, 0] > 15]
    maximus=maximus[maximus[:, 3]-maximus[:, 0] < 150]
    notes=getNotes()
    Notes=pd.DataFrame(notes,columns=names)
           

           
    return


#################################################################################################
#//////////////////////////////         Mail Service      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

#################################################################################################

def SendMail():
    mail="xxxx@xxxx.com"
    password="xxxxx"
    
    recipients = ['xxxx@xxxx.fr','xxxx@xxxx.com']
    fromaddr=mail     
    msg = MIMEMultipart()
     
    msg['From'] = fromaddr
    msg['To'] = ", ".join(recipients)
    msg['Subject'] = "Rapport"     
    body = "Saut Eric,\n"+" voila le rapport de %s"%x2[pp]["user"]["firstname"]+"qui travaille avec le kine %s"%x2[pp]["user"]["physiotherapist"]+"il est atteint de %s"%x2[pp]["user"]["pathology"] +"\n"+"Renvoie le rapport a Arnaud des que tu as un moment s'il te plait \n"+"Bises et vive les Pandas ! "
   
    
   
   
   
    msg.attach(MIMEText(body, 'plain'))
    fichier="C:/datatest/rapports/Rapport_"+"_%s.pdf"%idd 
    filename = "Rapport_%s"%x2[pp]["user"]["firstname"]+"_%s.pdf"%idd
    attachment = open(fichier, "rb")
     
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)
     
    msg.attach(part)
     
    server = smtplib.SMTP('smtp.xxxxx.com', 587)
    server.starttls()
    server.login(fromaddr, password)
    text = msg.as_string()
    server.sendmail(fromaddr, recipients, text)
    server.quit()
    attachment.close()
    try:
        os.remove(str(fichier))
    except OSError:
          pass
#################################################################################################
#//////////////////////////////         Mail Service      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
def squatAnalyse():
    global maxxx
    global minnn
    global MAXIMUS
    global error
    exercise="Squat"   
    
    #mph = min peak height // mpd = min peak distance (période)
    #Pour que ça marche bien, il faut filrer beaucoup plus.
    
    #les paramteres pour la detection des max et min doit etre changé  
    #pour le min c'est -50 parceque,c'est on met valley a true il va faire flex=-flex donc le min sera -50 
    maxx=detect_peaks(filtreflexMAX,mph=30,mpd=40,edge='rising')
    minn=detect_peaks(filtreflexMAX,mph=-50,mpd=20,edge='rising',valley='true')
    k=0
    maxxx=[]
    minnn=[]  
#################################################################################################
#////////////////////////////Filtrage minimal des mins et MAx\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################    
    for i in range(0,len(maxx-k)):
        if filtreflex[maxx[i]]>35:
            maxxx.append(maxx[i])     
    for i in range(0,len(minn)):
        if filtreflex[minn[i]]<30:
            minnn.append(minn[i]) 
    minlen=len(minnn)
    #associer chaque max a ses  deux minimums 
    mouvements=[]
    size=0
#################################################################################################
#////////////////////////////Separation des Mouvements    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
    maximus=np.zeros([len(maxxx),11])              
    try :
        maximus=getMaximus(maximus)           
    except :
        print('Pb with getMaximus')
        error = True
    #getMaximus() est là pour ordonner les max, min dans un tableau propre.              
    maximus=maximus[maximus[:, 0]-maximus[:, 3] > 15]
    maximus=maximus[maximus[:, 0]-maximus[:, 3] > 15]
    maximus=maximus[maximus[:, 0]-maximus[:, 3] < 180]
    maximus=maximus[maximus[:, 4] > maximus[:, 1]]
    
#################################################################################################
#//////////////////////////////    Notes des Mouvements   \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
    if len(maximus) > 0:    
        maximus2=pd.DataFrame(maximus,columns=Champs)
        
    return maximus


#Par Arnaud########################
#A partir des max et mins, on trouve les débuts de montée et début de descente
def SquatAnalyse2():

    global maximus

    SEUIL = 5
    
    if len(maximus) > 0:

        tab = np.zeros([len(maximus),7])              

        for i in range(0, len(maximus)):
   
            p1 = maximus[i][2]  #Indice du premier min
            p4 = maximus[i][1]  #Indice du max
            p7 = maximus[i][4]  #Indice du deuxième min
            
            if p4 > p1:
               
                j = p1
                while j < p4:
                    j = j + 1                    
                    if filtreflexMAX[j] > filtreflexMAX[p1] + SEUIL :
                        p2 = j  #Indice du début de montée
                        j = p4
            
                j = p2
                while j < p4:
                    j = j + 1
                    if filtreflexMAX[j] > filtreflexMAX[p4] - SEUIL :
                        p3 = j
                        j = p4  #Indice de fin de montée
            
            #print(p4)
            #print(p7)
            if p7 > p4:
                
                j = p4
                while j < p7:
                    j = j + 1                    
                    if filtreflexMAX[j] < filtreflexMAX[p4] - SEUIL :
                        p5 = j
                        #print('LA')
                        j = p7  #Indice de début de descente
                    else : 
                        p5 = p4
            
                j = p5
                while j < p7:
                    j = j + 1
                    if filtreflexMAX[j] < filtreflexMAX[p7] + SEUIL :
                        p6 = j
                        j = p7 #Indice de fin de descente
                    else : 
                        p6 = p7
                
#            print(tab)
            tab[i] = [p1, p2, p3, p4, p5, p6, p7]
    
    if 'tab'in locals():
        return tab
    
    else:
        tab = np.zeros([len(maximus),7])              
    
    return tab


###################################


def Proprio_Analyse():
    exercise="prop"         
    global maximus
    gx=savgol_filter(hexdata[:,3],61,3); 
    gy=savgol_filter(hexdata[:,4],21,5);   
    gz=savgol_filter(hexdata[:,5],21,5);
    ff=savgol_filter(filtreflex,101,3)
    Min_Mvt=detect_peaks(ff,mph=-30,mpd=150,edge='rising',valley='true')
    Max_GyrX=detect_peaks(gx,mph=20,mpd=200,edge='rising')
    me=mean(gx[Max_GyrX])
    Max_GyrX=detect_peaks(gx,mph=me-20,mpd=70,edge='rising')
    Min_GyrX=detect_peaks(gx,mph=20,mpd=100,edge='rising',valley='true')
    me=mean(gx[Min_GyrX])
    Min_GyrX=detect_peaks(gx,mph=-me-20,mpd=100,edge='rising',valley='true')    
    maximus=maximus=np.zeros([len(Min_GyrX),11])
    #for i in range(0,len(Min_Mvt)-1):
        
    for i in range (len(Min_GyrX)):
        maximus[i][0]=searchmin_Avant(Min_GyrX[i],exer,Min_Mvt,ff)
        maximus[i][1]=Min_GyrX[i]
        maximus[i][2]=searchmin_Apres(Min_GyrX[i],Max_GyrX,ff)      
        maximus[i][3]=searchmin_Apres(Min_GyrX[i],Min_Mvt,ff) 
        maximus[i][4]=maximus[i][3]-maximus[i][0]
        maximus[i][5]=variationsignal(int(maximus[i][0]),int(maximus[i][3]),int(maximus[i][1]),'flex',filtreflex,'prop')          
    filtreMvtsProprio()    
 

    return maximus    
    

def proprio():

    proprio = False
    
    if exer == "proprioception_static" or exer == "proprioception_pillow" or exer == "proprioception_forthback" or exer == "proprioception_leftright" or exer == "proprioception_compass":
        proprio = True
    
    else : 
        proprio = False
        
    return proprio
    
    
def getMaximus(maximus):
    global exer
    global maxxx
    global minnn    

    if proprio() == False :
    
        for i in range (len(maximus)):        
            maximus[i][0]=filtreflex[maxxx[i]]
            maximus[i][1]=maxxx[i]
            maximus[i][2]=searchmin_Avant(maxxx[i],exer,minnn,filtreflex)
            maximus[i][3]=filtreflex[int(maximus[i][2])]
            maximus[i][4]=searchmin_Apres(maxxx[i],minnn,filtreflex) 
            maximus[i][5]=filtreflex[int(maximus[i][4])]
            maximus[i][6]=maximus[i][4]-maximus[i][2]
     
        maximus=maximus[maximus[:, 6] != 0]
        maximus=maximus[maximus[:, 6] > 0]
        
        #maximus=maximus[maximus[:, 6] < 200]
        maximus=maximus[maximus[:, 2] != 0]
        maximus=np.delete(maximus,trim_table(maximus),0)
        # maximus=maximus[maximus[:, 7] <200]  
        
    

    return maximus
    
    #Structure de maximus:
    #

##########################################################################################
#//////////////////////////////  PDf Generation    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
##########################################################################################

def exportPdf(Series,x2):
    x=x2
    global PNum
    #plt.ioff()
    global c
    if pp==0:        
        fichier="C:/datatest/rapports/Rapport_"+"_%s.pdf"%idd
        c = canvas.Canvas(fichier)
        c.addPageLabel(1)
        c.drawImage("C:/datatest/panda15.jpg",500,750,70,70)
        c.setFontSize(20)
        c.drawString(210,800," Rapport De Séance ")
        c.setFontSize(13)
        c.drawString(20,750," Nom:%s"%x[pp]["user"]["firstname"])
        c.drawString(20,730," Prenom:%s"%x[pp]["user"]["firstname"])
        c.drawString(20,710," Date:%s"%str(x[pp]["createdAt"].ctime()))
        c.drawString(20,690," physiotherapist:%s"%x[pp]["user"]["physiotherapist"])
        c.drawString(20,670," pathology:%s"%x[pp]["user"]["pathology"])
        c.drawString(20,650," Nombre D'Exercices :%s"%len(x2))
        c.setFontSize(20)
        c.drawString(210,520," Recommendations")
        c.setFontSize(13)
        c.save()


      
    for i in range(0,len(Series)):
        print(Series)
        PNum=PNum+1
        c.addPageLabel(PNum)
        c.drawImage("C:/datatest/panda15.jpg",500,750,70,70)

        c.setFontSize(20)
        c.drawString(210,800," Exercice:%s"%x[pp]["exercise"])                            
        c.setFontSize(13)
        fig = plt.figure(figsize=(32, 16))
        imgdata = cStringIO.StringIO()
        fig.suptitle('Angle de flexion', fontsize=20)
        plt.xlabel('Temps (s) ', fontsize=18)
        plt.ylabel('Amplitude (degres)', fontsize=16)
        plt.plot(Datatime[int(Series[i][0]):int(Series[i][1])]/1000,PHI[int(Series[i][0]):int(Series[i][1])])
        plt.plot(Datatime[int(Series[i][0]):int(Series[i][1])]/1000,Trembl[int(Series[i][0]):int(Series[i][1])])
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        fig.savefig(imgdata, format='png')
        imgdata.seek(1)
        Image = ImageReader(imgdata)
       # c.drawString(250,650,"Angle de Flexion")
        #c.drawAlignedString(250,670,"Angle de Flexion")
        c.drawImage(Image, 50, 370, 500, 250,preserveAspectRatio=True)
        c.drawString(20,750," Serie : %s "%(i+1)+"/%s"%(len(Series)))
        c.drawString(20,710," pathology:%s"%x[pp]["user"]["pathology"])     

        if "comment" in x[pp]:
            c.drawString(20,730," Commentaire:%s"%x[pp]["comment"])
        else :
            c.drawString(20,730," Commentaire:Pas de commentaires ")        
      
        fig2 = plt.figure(figsize=(32, 16))
        imgdata2= cStringIO.StringIO()
        plt.plot(Datatime[int(Series[i][0]):int(Series[i][1])]/1000,filtrerot[int(Series[i][0]):int(Series[i][1])],color='green')
        fig2.suptitle('Angle de Rotation ', fontsize=20)
        plt.xlabel('Temps(s) ', fontsize=18)
        plt.ylabel('Amplitude(degres)', fontsize=16)
        plt.xticks(size = 20)
        plt.yticks(size = 20)
        fig2.savefig(imgdata2, format='png')
        imgdata2.seek(1)
        Image2 = ImageReader(imgdata2)
        c.drawImage(Image2, 50, 80, 500, 250,preserveAspectRatio=True) 
        
        
        
    c.save()
          
        #send
##########################################################################################
#//////////////////////////////  PDf Generation    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
##########################################################################################
    


def getNotes():
    notes=np.zeros([len(maximus),11])
    for i in range(len(maximus)):
        maximus[i][7]=variationsignal(int(maximus[i][2]),int(maximus[i][4]),int(maximus[i][1]),'flex',filtreflex,exercise)          
        maximus[i][8]=Symetrieflex(int(maximus[i][2]),int(maximus[i][1]),int(maximus[i][4]),filtreflex)
        maximus[i][9]=np.float64(SymetrieRot(int(maximus[i][2]),int(maximus[i][1]),int(maximus[i][4]),filtrerot))
        maximus[i][10]=variationsignal(int(maximus[i][2]),int(maximus[i][4]),int(maximus[i][1]),'rot',filtrerot,exercise)      
        notes[i][0]=maximus[i][1]
        
        #note Bruit
        notes[i][1]=1-(maximus[i][7]/maximus[i][6])
        notes[i][2]=1-(maximus[i][10]/maximus[i][6])
        #note Symetrie Flex
        notes[i][3]=abs(maximus[i][8])
        #note symetrie Rot
        notes[i][4]=abs(maximus[i][9])   
        #correlation
        notes[i][5]=Corr_flex_rot(int(maximus[i][2]),int(maximus[i][1]),int(maximus[i][4]),exercise,filtrerot,filtreflex)/(maximus[i][4]-maximus[i][2])
        #note Max
        notes[i][6]=noterMax(int(notes[i][0]),Ref,exercise,filtreflex)
        #bruit fourier
        #print(i)
        notes[i][7]=fourierTransform(int(maximus[i][1]),int(maximus[i][4]),'flex',notes[i][0],filtreflex,filtrerot)
        notes[i][8]=fourierTransform(int(maximus[i][1]),int(maximus[i][4]),'rot',notes[i][0],filtrerot,filtrerot)    
        notes[i][9]=(notes[i][1]*2+notes[i][3]*2+notes[i][6]*3+notes[i][7])/8
        notes[i][10]=(notes[i][2]+notes[i][4]+notes[i][8])/3
        notes[i][0]=Datatime[int(maximus[i][1])]/1000

    return notes

#def getIndiceSquat():
#    indicators=np.zeros([len(maximus),4])
#    for i in range(len(maximus)):
#        indicators[i][0]=getFluidMvt(int(maximus[i][2]),int(maximus[i][1]),int(maximus[i][4]),filtreflex)     
#        indicators[i][1]=getCordination(int(maximus[i][2]),int(maximus[i][4]),filtreflex,filtrerot)
#        #indicators[i][2]=getPuissance(Datatime[int(maximus[i][2])],Datatime[int(maximus[i][1])],filtreflex/1000)
#        indicators[i][2]=(Datatime[int(maximus[i][1])]-Datatime[int(maximus[i][2])])/1000
#        
#        indicators[i][3]=getStabMvt(int(maximus[i][2]),int(maximus[i][4]),filtreflex)
#
#
#    return indicators
#    
#def getIndiceProp():
#    indicators=np.zeros([len(maximus),4])
#    for i in range(len(maximus)):
#        indicators[i][0]=getFluidMvtprop(int(maximus[i][1]),int(maximus[i][2]),int(maximus[i][3]),int(maximus[i][4]),filtreflex)     
#        indicators[i][1]=getCordination(int(maximus[i][1]),int(maximus[i][4]),filtreflex,filtrerot)
#        indicators[i][2]=getPuissance(int(maximus[i][1]),int(maximus[i][2]),filtreflex)
#        indicators[i][3]=getStabMvt(int(maximus[i][2]),int(maximus[i][3]),filtreflex)
#
#
#    return indicators


#Modifié par Arnaud###############


    


#Indicators a 4 colonnes : fluid, coor, puiss, stab // Autant de lignes que de mouvements
def getIndice():

    RANGE_PROPRIO = len(filtreflex)/6  #taille de ce qu'on enlève arbitrairement    
    
    indicators = np.zeros([len(tab), 4])


    if proprio() == True :
        
            indicators = np.zeros([1, 4])

            #La fluidité n'est pas à prendre en compte (default = -1)
            indicators[0][0] = -1
            
            #La coordination n'est pas à prendre en compte (default = -1)
            indicators[0][1] = -1

            #La puissance n'est pas à prendre en compte (default = -1)
            indicators[0][2] = -1
            
            #La stabilité se calcule sur la proprioception uniquement
            #Quand l'algo Alfred sera prêt on le fera à partir de la valeur de tremblements     
            indicators[0][3] = getStabMvt(RANGE_PROPRIO, len(filtreflex) - RANGE_PROPRIO, filtreflex)/ (float( len(filtreflex)) * 2 / 3)


    for i in range(len(tab)):
        
        if proprio() == False and len(maximus) > 0 :        
            
            #La fluidité est prise entre le début et la fin de montée
            indicators[i][0] = 1000 * ( getFluidity(int(tab[i][1]), int(tab[i][2]), filtreflex) / (tab[i][2] -tab[i][1])) + ( getFluidity(int(tab[i][4]), int(tab[i][5]), filtreflex)) / (tab[i][5] - tab[i][4])
            
            #La coordination est prise tout le long du mouvement
            indicators[i][1] = 1000 * getCoordination(int(tab[i][0]), int(tab[i][6]), filtreflex, filtrerot) / ( tab[i][6] - tab[i][0])
    
            #La puissance est calculée pendant la montée et la descente
            indicators[i][2] = getPuissance(int(tab[i][1]), int(tab[i][2]), int(tab[i][4]), int(tab[i][5]),PHI)
            indicators[i][2] = getPuissance(int(tab[i][1]), int(tab[i][2]), int(tab[i][4]), int(tab[i][5]), filtreflex)
            
            #La stabilité n'est pas à prendre en compte (default = -1)
            indicators[i][3] = -1
            
    if (proprio() == False and len(maximus) == 0) or error == True :
        indicators = np.zeros([1, 4])
        indicators[0][0] = -1
        indicators[0][1] = -1
        indicators[0][2] = -1
        indicators[0][3] = -1
            
            

        
    return indicators

#Pour le moment la notation se fait en moyenne sans pondération (dans l'exo)
#Amélioration: pondérer les poids des différentes notes par mvnt par le nm de points 
#que l'on a regardé pendant le mvnt.


def getIndiceEX():
    
    #tableau qui recap les notes pour l'exercice (Fluidité, Coor, Puiss, Stab, 
    #0 pour Squats et 1 pr proprio)
    indicatorsex = np.zeros([1, 6])

    indicators = getIndice()
    
    if proprio() :
        
        indicatorsex[0, 4] = 1        
    else :
        indicatorsex[0, 4] = 0
    
    if len(indicators) > 0 :  
        
        for i in range(0, indicators.shape[1]) :
            
            indicatorsex[0, i] = mean(indicators[:, i])
    
    indicatorsex[0, 5] = len(flex)

    return indicatorsex



   
       
def getIndicesSeance() :

    global tousindices
    zz=copy.deepcopy(tousindices)
    means = np.zeros([1, 4])
    
    for popo in range(0, 4) :
        zz=copy.deepcopy(tousindices)
        for i in range(len(tousindices)-1) :

            if zz[i, popo] == -1 :

                zz[i, 5] = 0
                
        means[0][popo] = sum( zz[:, popo] * zz[:, 5] ) / float(sum( zz[:, 5] ))       

    return means
    
    
#    
#def getIndiceProp():
#
#    indicators=np.zeros([len(maximus),4])
#
#    for i in range(len(maximus)):
#        
#        indicators[i][0]=getFluidMvtprop(int(maximus[i][1]),int(maximus[i][2]),int(maximus[i][3]),int(maximus[i][4]),filtreflex)     
#        indicators[i][1]=getCoordination(int(maximus[i][1]),int(maximus[i][4]),filtreflex,filtrerot)
#        indicators[i][2]=getPuissance(int(maximus[i][1]),int(maximus[i][2]),filtreflex)
#        indicators[i][3]=getStabMvt(int(maximus[i][2]),int(maximus[i][3]),filtreflex)
#
#    return indicators
###################################



def filtreMvtsProprio():
    global maximus
    todelete=[]
    k=0
    bo=False
    bo1=True
    i=0
    kk=0
    while bo1:
        kk+=1
        
        
        if i<=len(maximus)-2:    
            #print(i)
            if maximus[i][0]==maximus[i+1][0] :
                for  j in range(i,len(maximus)-1):
                    if maximus[j][0]==maximus[j+1][0] :
                        todelete.append(j)
                        i=j
                        k=j
                    elif j==len(maximus)-2:
                       # print("sssss")
                        bo=True
                        break
            else:
                i=i+1                
                    
        if bo or kk>50:
            bo1=False
            break  
    
    maximus=np.delete(maximus,todelete,0)
    maximus=maximus[maximus[:, 4] != 0]
    maximus=maximus[maximus[:, 4] > 0]    
    
    
    
    
    
    
#################################################################################################
#//////////////////////////////   Detection Des Series    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################

def detectSeries():
    global maximus
    meanfrq=[]
    for i in range(1,len(maximus)-1):
        meanfrq.append(maximus[i][1]-maximus[i-1][1])
    
    
    Series=np.zeros([len(maximus),2])
    l=0    
    i=0
    b=True
    s=0
    m=0
    #minfrq=180
    minfrq=mean(meanfrq)
    while i <len(maximus) and b==True:
        s=s+1
        m=i+1
        #if s>len(maximus):
            #b=False
        dist=0
        dist2=0
        k=0
        for j in range(m,len(maximus)):

            #↓print(j)
            dist2=maximus[j-1][1]-maximus[i][1]
            #Sdist3=maximus[j][1]-maximus[i][1]
            dist=maximus[j][1]-maximus[i][1]
           # print("i",i,"j",j)
            #print("d1",dist,"d2",dist2)
            
            #print(dist-dist2)
            if (dist-dist2)<minfrq+30:
                k+=1
                #print(k)
            else:
                if k>=4:
                    Series[l][0]=maximus[i][2]
                    Series[l][1]=maximus[i+4][4]  
                    l+=1
                    i=j       
                    break
                elif k<4:
                  
                    Series[l][0]=maximus[i][2]  
                    Series[l][1]=maximus[i+k][4]
                    l+=1
                    i=j
                    break
            
            if j==(len(maximus)-1) and i!=0 and k!=0:
                b=False        
                if k>=4:
                    Series[l][0]=maximus[i][2]  
                    Series[l][1]=maximus[i+4][4]
                    l+=1
                elif k<4:
                    #print(l)
                    #print(l)               
                    Series[l][0]=maximus[i][2]  
                    Series[l][1]=maximus[i+k][4]
                    l+=1
                    break
            if j==(len(maximus)-1) and i==0 and k!=0:
                if len(maximus)>4 and k>=4:
                    Series[0][0]=maximus[i][2]  
                    Series[0][1]=maximus[i+4][4]
                    l+=1
                    break
                elif len(maximus)>4 and k<4:
                    #∟print(i)
                    #print(l)               
                    Series[0][0]=maximus[i][2]  
                    Series[0][1]=maximus[i+k][4]
                    l+=1
                    break
                elif len(maximus)<=4 :  
                    
                    Series[0][0]=maximus[i][2]  
                    Series[0][1]=maximus[i+len(maximus)-1][4]
                    l+=1    
                    break
                                
                break
                      
        if s>80:
            b=False
            
    
    return Series            
            
def detectSeriesProp():
    meanfrq=[]    
    for i in range(1,len(maximus)-1):
        meanfrq.append(maximus[i][0]-maximus[i-1][0])
            
    Series=np.zeros([len(maximus),2])
    l=0    
    i=0
    b=True
    s=0
    m=0
    #minfrq=100
    minfrq=mean(meanfrq)
    while i <len(maximus) and b==True:
        s=s+1
        m=i+1
        #if s>len(maximus):
            #b=False
        dist=0
        dist2=0
        k=0
        for j in range(m,len(maximus)):
            #↓print(j)
            dist2=maximus[j-1][0]-maximus[i][0]
            #Sdist3=maximus[j][1]-maximus[i][1]
            dist=maximus[j][0]-maximus[i][0]
           # print("i",i,"j",j)
            #print("d1",dist,"d2",dist2)
            
            #print(dist-dist2)
            if (dist-dist2)<minfrq+minfrq/2:
               # print(k)
                
                k+=1
            else:
                if k>=4:
    
                    Series[l][0]=maximus[i][0]
                    Series[l][1]=maximus[i+4][3]  
                    l+=1
                    i=j       
                    break
                elif k<4:
                  
                    Series[l][0]=maximus[i][0]  
                    Series[l][1]=maximus[i+k][3]
                    l+=1
                   
                    i=j
                    break
            
            if j==(len(maximus)-2) and i!=0 and k!=0:
                if k>=4:
                    Series[l][0]=maximus[i][0]  
                    Series[l][1]=maximus[i+4][3]
                    l+=1
                elif k<4:
                    #∟print(i)
                    print(k)               
                    Series[l][0]=maximus[i][0]  
                    Series[l][1]=maximus[i+k][3]
                    l+=1
                    break
                
            if j==(len(maximus)-1) and i==0 and k!=0:
                if len(maximus)>4 and k>=4:
                    Series[0][0]=maximus[i][0]  
                    Series[0][1]=maximus[i+4][3]
                    l+=1
                elif len(maximus)>4 and k<4:
                    #∟print(i)
                    #print(l)               
                    Series[0][0]=maximus[i][0]  
                    Series[0][1]=maximus[i+k][3]
                    l+=1
                elif len(maximus)<4 :                
                    Series[0][0]=maximus[i][0]  
                    Series[0][1]=maximus[i+len(maximus)-1][3]
                    l+=1                
                    
                    break
            
                b=False        
                break 
        if s>80:
            b=False
    
    


    return Series
 







## get Data la fonction qui calcul les angles   
def getData():
    split1,split4,split3,split2=[],[],[],[]   
    spliteddata=[]
    decrypteddata=[]
    global fields
    global hexdata
    for i in range(len(fields)):
                #print(i)
        if i==0:
            val = fields[i].split('[[', 1)[1].split(']')[0]
            #print(val)
        else:
            val = fields[i].split('[', 1)[1].split(']')[0]
                   # print(val)
        d.append(val)
        spliteddata.append(d[i].split(','))
    hexdata=np.zeros([len(spliteddata),17])
    spliteddata2=[]
    for i in range(0,len(spliteddata)):
        spliteddata2.append(spliteddata[i][1:21])
    #Shift and split  of bytes    
        split1.append(splitbits(spliteddata2[i][4]))
        split2.append(splitbits(spliteddata2[i][9]))
        split3.append(splitbits(spliteddata2[i][14]))
        split4.append(splitbits(spliteddata2[i][19]))
    #Bitwise operations     
        for m in range(0,4):
            hexdata[i][m]=int(spliteddata2[i][m])|split1[i][m]<<8 
        for m in range(4,8):
            hexdata[i][m]=int(spliteddata2[i][m+1])|split2[i][m-4]<<8       
        for m in range(8,12):
            hexdata[i][m]=int(spliteddata2[i][m+2])|split3[i][m-8]<<8
        for m in range(12,16):
            hexdata[i][m]=int(spliteddata2[i][m+3])|split4[i][m-12]<<8      
                   # d2.append(d[0][i].split(",",20))
        hexdata[i][0]=hexdata[i][0]/10 -50    
        hexdata[i][1]=hexdata[i][1]/10 -50    
        hexdata[i][2]=hexdata[i][2]/10 -50    
        hexdata[i][3]=hexdata[i][3]-500    
        hexdata[i][4]=hexdata[i][4]-500    
        hexdata[i][5]=hexdata[i][5]-500    
        hexdata[i][6]=hexdata[i][6]/10 -50    
        hexdata[i][7]=hexdata[i][7]/10 -50    
        hexdata[i][8]=hexdata[i][8]/10 -50    
        hexdata[i][9]=hexdata[i][9]-500    
        hexdata[i][10]=hexdata[i][10]-500    
        hexdata[i][11]=hexdata[i][11]-500    
        hexdata[i][12]=(hexdata[i][12]-200)/10     
        hexdata[i][13]=(hexdata[i][13]-200)/10    
        hexdata[i][14]=hexdata[i][14]
        hexdata[i][15]=hexdata[i][15]   
        hexdata[i][16]=float(spliteddata[i][0])   
    
        i=i+1          
    data=hexdata
    del split1,split2,split3,split4
    
    flexion=np.zeros([len(data),3])
    rotation=np.zeros([len(data),4])
    #################################################################################################
    #////////////////////////////Angles Calculation & Filtrage\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
    #################################################################################################
    
    for i in range(0,len(data)):
        flexion[i][0]=atan2(data[i][2],data[i][1])*180/pi
        flexion[i][1]=atan2(data[i][6],data[i][7])*180/pi
#''' A verifier cette methode de calcul parce que si on fait l'addition l'angle est correct 
#
#'''
        flexion[i][2]=flexion[i][0]-flexion[i][1]

        rotation[i][0]=asin(data[i][0]/sqrt(data[i][0]*data[i][0]+data[i][1]*data[i][1]+data[i][2]*data[i][2]))*180/pi
        rotation[i][1]=asin(-data[i][8]/sqrt(data[i][6]*data[i][6]+data[i][7]*data[i][7]+data[i][8]*data[i][8]))*180/pi
        rotation[i][2]=rotation[i][0]-rotation[i][1]





    return flexion,rotation   
    
    
    
    
    
def read_in():
    
    lines = sys.stdin.readlines()
    #Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])    
    
    
    
    
    
    
#################################################################################################
#////////////////////////////////////////////Execution\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################

###la partie qui suit jusqu'a la boucle de lecture doit etre adapté selon la façon de  lecture de données sur la base de donnée 
#bref ca sert a la recuperation des données de la base 

#ide=read_in()
#print(ide)
#idd=ObjectId(ide)
#print(ide)



#connexion a la base de données 
idd=ObjectId("59a4118dbd36e581db0be41d")
MONGO_HOST = "127.0.0.1"
MONGO_DB = "Panda2"
client = pymongo.MongoClient("127.0.0.1",27017)
db = client[MONGO_DB]
collection1 = db.exer
collection2 = db.exercises

#recuperation des données 
#result = collection.find({'$and':[{"exercise":exer},{'user.firstname':fname},{"_id":idd}]})
result = collection1.find({"_id":idd})
result3=collection1.find()
curseur = []

for i in result3:
    curseur.append(i['_id'])


#rows c'est la variables qui contient le resultat de la requete sur la base       
rows = []
for i in result:
    rows.append(i)
    

#idee=str(x[0]["_id"])
exerids=[]
#for i in x[0]["exer_ids"]:
#    exerids.append(ObjectId("%s"%str(i)))
    
result2 = collection2.find({"_id":idd})
result2 = collection2.find({"_id":{'$in': exerids }})


x2 = []
for i in result2:
    x2.append(i)
x2 =rows[0]

#la variable names et champs  c'est pour le stockage des ancien  indicators dans un tabelau avec des champs de colonnes 
names=['id','bruitFlex','bruitRot','symflex','symrot','cor','Note MAx','fftFlex','fftRot','Noteflex','NoteRot']
Champs = ['Valeur MAX','Ind MAX','Ind MIN1','Valeur MIN1','Ind MIN2','Valeur MIN2','Periode', 'Autre1', 'Autre2', 'Autre3', 'Autre4']

#Ref c'est la reference d'angle pour les squats
Ref=110

#j'ai creer cette variable Mode pour choisir si on veut des indicateurs ou seulement Pdf 
mode='onlypdf'
mode='indicators'

#boucle pour la lecture et le traitement de chaque  exercice dans la sceance 
#mmm c'est le compteur 
#pp c'est le compteur dans la form string parceque il faut passer les index comme des string  

tousindices = np.zeros([len(x2)-1, 6])

for mmm in range(0, len(x2) - 2):    
    d=[]
    spliteddata=[]
    decrypteddata=[]
    pp="%s"%mmm   
    #Data c'est la variable qui contient les données brutes 
    Data=x2[pp]["messages"]
    exer=x2[pp]["exercise"]    
    fields = Data.split('],')
    
    #recuperation des angles depuis les données compresées 
    [flexion,rotation]=getData()    
    flex=[flexion[i][2] for i in range(len(flexion))]
    rot=[rotation[i][2] for i in range(len(rotation))]
    #Les données sont stocker dans une variable global Hexdata qu'on va passer au script d'alfred pour le calcul d'angle et tremblement
    
    #Alfred c'est le script d'alfred je l'ai mis comme une fonction 
    [PHI,Trembl]=Alfred(hexdata)
    
    #get Time c'est pour calculer le temps
    Datatime=getTime();        
    
    
    #filtrage des angles 
    filtreflex=savgol_filter(flex,21,5)
    filtreflexMAX=savgol_filter(flex,101,5)
    #filtreflex=PHI
    filtrerot=savgol_filter(rot,21,5)
   #print(len(flex))
    #print(exer)

    
    
    
    if mode=='indicators':
        """
        Pour Ce mode d'excution est differente selon le type d'exercice
        
        #maximus c'est un tableua varibale qui relie les maximum au minimums 
        et les anciens indicateurs pour l'extention et flexion
        
        #Series c'est un tableau qui contient les indices des Series soit avec detection
        des series si on arrive a dtecter les mouvement Separement avec la fonction detectSeries()
        sinon c'est la separation aleatoire  avec la fonction getAleatoireSeries()   
        
        """    
        
#        if exer == 'jumpside_2legs' :     
        error = False
        maximus = squatAnalyse()
        tab = SquatAnalyse2()
        indicators = getIndice()
        indicatorsex = getIndiceEX()
        tousindices[mmm][:] = indicatorsex[:]
                
#            tousindices[mmm][0] = tousindices[mmm][0] / tousindices[mmm][]
        
        
        
        #####ICI Hypothèse: que des squats & proprio#####################
#        if exer=="extention":
#            #filtreflex=savgol_filter(flex,101,5)
#            try:
#                
#                #cette variable contient le resultat de  l'analyse 
#                maximus=extentionAnalyse()
#
#            except:
#                print("impossible de detecter les mouvemnts ")
#
#             
#            try:
#                #cette variable contient les Series 
#                Series=detectSeries()
#                #cette instruction est pour eliminer les zeros qui sont intitalisé avec le tabelau               
#                Series=Series[Series[:, 1] != 0]
#                #cette condition c'est pour  verifier si on a des mouvement mais la series n'est pas detecter
#                #pour cela je prends toute l'exercice je l'ai fait provisoirement pour le debug 
#                if len(Series)==0 and len(maximus)!=0:
#                    Series=np.zeros([1,2])
#                    Series[0][0]=maximus[0][0]
#                    Series[0][1]=maximus[len(maximus)-1][3]
#                elif len(Series)==0 and len(maximus)==0:
#                    Series=np.zeros([1,2])
#                    Series[0][0]=0
#                    Series[0][1]=len(flex)-1
#          
#          
#            except:
#                print("impossible de detecter les Series ")
#     
#                
#            if 'Series' in globals():
#                try:
#                    pppppppp=5555                                 
#                    #exportPdf(Series)
#                except:
#                    print("impossible d'exporter le pdf  les Series ")
#
#            else :               
#                 try:
#                    Series=getAleatoireSeries()           
#
#                    #exportPdf(Series)
#                 except:
#                    print("impossible d'exporter le pdf  les Series ")
#
#                
##pour les Squat c'est pareil pour le cas de les extentions precedent 
#                
#        elif exer=="squats_2feet" or exer=="squats_1foot" :
#            
#             
#            print("hello")
#            #filtreflex=savgol_filter(flex,101,5)
#            exercise="Squat"
#            try :           
#                maximus=squatAnalyse()
#                
#               
#                print(maximus)
#    
#            except :        
#                print("impossible de detecter les mouvemnts ")
#            
#            try:
#                Series=detectSeries()
#                Series=Series[Series[:, 1] != 0]
#                if len(Series)==0 and len(maximus)!=0:
#                    Series=np.zeros([1,2])
#                    Series[0][0]=maximus[0][0]
#                    Series[0][1]=maximus[len(maximus)-1][3]
#                elif len(Series)==0 and len(maximus)==0:
#                    Series=np.zeros([1,2])
#                    Series[0][0]=0
#                    Series[0][1]=len(flex)-1
#            except :        
#                print("impossible de detecter les series ")                
#                    
#                    
#                if 'Series' in globals():  
#                    ppppppppp=2
#
#                    #exportPdf(Series,x2)
#                else :
#                    Series=getAleatoireSeries()           
#                    #exportPdf(Series)
#                    
#                Series=getAleatoireSeries()           
#                #exportPdf(Series)        
#                        
#    
#        
#        elif exer=="proprioception_static":
#           try:
#                #le traitement pour la proprioceprion est different on utilise les gyroscopes 
#                maximus=Proprio_Analyse()
#           except: 
#                print("impossible de decomposer les mouvements")                  
#                #detection des serie pour la proprio           
#           try:
#                Series=detectSeriesProp()
#                print(Series)
#                #verifier si la taille de serie elle doit pas avoir un intervalle null                      
#                Series=Series[Series[:, 1] != 0]
#                #verifier si y'a des mouvement mais pas de serie alors on prend tous les movement                                  
#                if len(Series)==0 and len(maximus)!=0:
#                    Series=np.zeros([1,2])
#                    Series[0][0]=maximus[0][0]
#                    Series[0][1]=maximus[len(maximus)-1][3]
#           except:
#               print("impossible de detetceter Les series ")
#    
#            #verifier si on a des serie sinon generer une serie automatique 
#           if 'Series' in globals():
#                try:        
#                    print("dddd de genere le Pdf ")
#
#                    #exportPdf(Series,x2)
#                    print(1)
#                except:
#                    print("impossible de genere le Pdf ")
#           else :
#                try:
#                    Series=getAleatoireSeries()           
#                    #exportPdf(Series)  
#                except:
#                    print("impossible de genere le Pdf ")
#                        
#
#
#
#
#
#                        
#        
#        #Traitement a faire pour les autres exercices generation du pdf directement jusqua trouver les incdicatuers pour ces exercices 
#        elif exer!="extention" and exer!="proprioception_static" and exer!="squats_2feet" and  exer!="squats_1foot" :
#            try:               
#                Series=getAleatoireSeries()
#                try:
#                    ppppppppp=2
#
#                    #exportPdf(Series,x2)
#                except:
#                    print("impossible d'exporter notype exercice Pdf")
#                   
#    
#            except:
#                print("impossible de detecter les series ")
#                
                
                
        """apres le traitement de l'exercice on  genere le Pdf 
        je sais pas si on genere ici ou dans le bloc de l'exerice 
        reste a confimer
        tmpps===le temps d'execution depuis qu'on lancé le script
        plt.close('all') c'est pour fermer les fentres qui sont dans 
        """          
        #exportPdf(Series,x2)           
        #Series=getAleatoireSeries()           
        #exportPdf(Series,x2,indicators)
        #del(Series)
                        
        tmps2=time.clock()
        plt.close('all')
        #print("%f\n" %(tmps2-tmps1))   
    
    
    
    
    elif mode=='onlypdf':
        """Ce Mode est lorsque on demande seulement le Pdf
        
        
        """  
        
        Series=getAleatoireSeries()           
        #exportPdf(Series,x2)
        del(Series)                
        tmps2=time.clock()
        #plt.close('all')
        #print("%f\n" %(tmps2-tmps1))      
        

meanss = getIndicesSeance()


#Ajouter sur la db
try :
    db.exer.update({'_id' : idd}, {'$set' : {'indices' : {'stabilité' : str(meanss[0][0]), 'fluid' : str(meanss[0][1]), 'puiss' : str(meanss[0][2]), 'stab' : str(meanss[0][3]) } } })
except :
    print('fail to update indicators')
        
"""la fonction send Mail est appelé qu'a la fin de traitement Parceque la generation
du PDf est faite au cours du traitement de la sceance donc on attend la fin de generation 
pour l'envoyé 

"""  
#SendMail()    
