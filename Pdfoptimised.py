# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:47:27 2017
@author: Samorg
"""

#################################################################################################
#////////////////////////////////////////////Load Modules\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
import time
import os
tmps1=time.clock()
import sys,json
#sys.path.append('/home/sami/NodePyhton/functions/')
#sys.path.append('G:\Education\Projects\SmartR\Python Scripts\SmartR\FinalScripts\PandaS')
sys.path.append('C:\Users\Arnaud\Documents\SmartR\Algos server')
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


    mail="xxx@xxx.com"
    password="123456"

    toaddr = "xxx@xxxx"
    recipients=['xxxx','xxxxx']
    fromaddr=mail     
    msg = MIMEMultipart()
    msg['To'] = ", ".join(recipients)

    msg['From'] = fromaddr
    msg['Subject'] = "Rapport pour %s" %x2[pp]["user"]["physiotherapist"]    
    body = "Saut Eric,\n"+"Voila le rapport de %s"%x2[pp]["user"]["firstname"]+" qui travaille avec le kine %s"%x2[pp]["user"]["physiotherapist"] +" il est atteint de %s."%x2[pp]["user"]["pathology"] +"\n"+"Renvoie le rapport a Arnaud des que tu as un moment s'il te plait. \n"+"Bises et vive les Pandas !"
    msg.attach(MIMEText(body, 'plain'))
    fichier="/xxx/xxxx/NodePyhton/rapports/Rapport_"+"_%s.pdf"%idd 
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
    server.sendmail(fromaddr, toaddr, text)
    server.quit()
    attachment.close()
    
    #try:
        #os.remove(str(fichier))
    #except OSError:
        #pass
#################################################################################################
#//////////////////////////////         Mail Service      \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
def squatAnalyse():
    global maxxx
    global minnn
    exercise="Squat"   
    maxx=detect_peaks(filtreflex,mph=30,mpd=40,edge='rising')
    minn=detect_peaks(filtreflex,mph=-30,mpd=20,edge='rising',valley='true')
    k=0
    maxxx=[]
    minnn=[]  
#################################################################################################
#////////////////////////////Filtrage minimal des mins et MAx\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################    
    for i in range(0,len(maxx-k)):
        if flex[maxx[i]]>35:
            maxxx.append(maxx[i])     
    for i in range(0,len(minn)):
        if flex[minn[i]]<30:
            minnn.append(minn[i]) 
    minlen=len(minnn)
    #associer chaque max a ses  deux minimums 
    mouvements=[]
    size=0
#################################################################################################
#////////////////////////////Separation des Mouvements    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################
    maximus=np.zeros([len(maxxx),11])              
    maximus=getMaximus(maximus)                         
    maximus=maximus[maximus[:, 0]-maximus[:, 3] > 15]
    maximus=maximus[maximus[:, 0]-maximus[:, 3] > 15]
    maximus=maximus[maximus[:, 0]-maximus[:, 3] < 150]
#################################################################################################
#//////////////////////////////    Notes des Mouvements   \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################

    return maximus


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
        maximus[i][0]=searchmin_Avant(Min_GyrX[i],exercise,Min_Mvt)
        maximus[i][1]=Min_GyrX[i]
        maximus[i][2]=searchmin_Apres(Min_GyrX[i],Max_GyrX)      
        maximus[i][3]=searchmin_Apres(Min_GyrX[i],Min_Mvt) 
        maximus[i][4]=maximus[i][3]-maximus[i][0]
        maximus[i][5]=variationsignal(int(maximus[i][0]),int(maximus[i][3]),int(maximus[i][1]),'flex',filtreflex,'prop')          
    filtreMvtsProprio()    
 

    return maximus    
    
def getMaximus(maximus):
    global exercise
    global maxxx
    global minnn    
    for i in range (len(maximus)):        
        maximus[i][0]=flex[maxxx[i]]
        maximus[i][1]=maxxx[i]
        maximus[i][2]=searchmin_Avant(maxxx[i],exercise,minnn,filtreflex)
        maximus[i][3]=flex[int(maximus[i][2])]
        maximus[i][4]=searchmin_Apres(maxxx[i],minnn,filtreflex)
        maximus[i][5]=flex[int(maximus[i][4])]
        maximus[i][6]=maximus[i][4]-maximus[i][2]
    #if maximus[i][2] < maximus[i][1]:
    maximus=maximus[maximus[:, 6] != 0]
    maximus=maximus[maximus[:, 6] > 0]
#maximus=maximus[maximus[:, 6] < 200]
    maximus=maximus[maximus[:, 2] != 0]
    maximus=np.delete(maximus,trim_table(maximus),0)
   # maximus=maximus[maximus[:, 7] <200]    
    return maximus







##########################################################################################
#//////////////////////////////  PDf Generation    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
##########################################################################################

def exportPdf(Series,x):
    global PNum
    plt.ioff()
    global c
    if pp==0:        
        fichier="/xxxx/xxxx/xxxx/xxxx/Rapport_"+"_%s.pdf"%idd
        c = canvas.Canvas(fichier)
        c.addPageLabel(1)
        c.drawImage("/xxxx/xxxx/NodePyhton/panda15.jpg",500,750,70,70)
        c.setFontSize(20)
        c.drawString(210,800," Rapport De Séance ")
        c.setFontSize(13)
        c.drawString(20,750," Nom:%s"%x[pp]["user"]["firstname"])
        c.drawString(20,730," Prenom:%s"%x[pp]["user"]["lastname"])
        c.drawString(20,710," Date:%s"%str(x[pp]["createdAt"].ctime()))
        c.drawString(20,690," physiotherapist:%s"%x[pp]["user"]["physiotherapist"])
        c.drawString(20,670," pathology:%s"%x[pp]["user"]["pathology"])
        c.drawString(20,650," Nombre D'Exercices :%s"%(len(x2[0])-1))
        c.setFontSize(20)
        c.drawString(210,520," Recommendations")
        c.setFontSize(13)
        c.save()
     
          
        
    for i in range(0,len(Series)):       
        PNum=PNum+1
        c.addPageLabel(PNum)
        c.drawImage("/xxxx/xxxx/NodePyhton/panda15.jpg",500,750,70,70)

        c.setFontSize(20)
        c.drawString(210,800," Exercice:%s"%x[pp]["exercise"])                            
        c.setFontSize(13)
        fig = plt.figure(figsize=(28, 14))
        imgdata = cStringIO.StringIO()
        fig.suptitle('Angle de flexion', fontsize=20)
        plt.xlabel('Temps (s) ', fontsize=18)
        plt.ylabel('Amplitude (degres)', fontsize=18)
        plt.plot(Datatime[int(Series[i][0]):int(Series[i][1])]/1000,filtreflex[int(Series[i][0]):int(Series[i][1])])
        plt.plot(Datatime[int(Series[i][0]):int(Series[i][1])]/1000,Trembl[int(Series[i][0]):int(Series[i][1])])
        plt.xticks(size=20)
        plt.yticks(size=20)
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
        plt.xlabel('Temps (s) ', fontsize=18)
        plt.ylabel('Amplitude (degres)', fontsize=16)
        plt.xticks(size=20)
        plt.yticks(size=20)
        fig2.savefig(imgdata2, format='png')
        imgdata2.seek(1)
        Image2 = ImageReader(imgdata2)
        c.drawImage(Image2, 50, 80, 500, 250,preserveAspectRatio=True)
        c.save()
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
        notes[i][6]=noterMax(notes[i][0],Ref,exercise,filtreflex)
        #bruit fourier
        #print(i)
        notes[i][7]=fourierTransform(int(maximus[i][1]),int(maximus[i][4]),'flex',notes[i][0],filtreflex,filtrerot)
        notes[i][8]=fourierTransform(int(maximus[i][1]),int(maximus[i][4]),'rot',notes[i][0],filtrerot,filtrerot)    
        notes[i][9]=(notes[i][1]*2+notes[i][3]*2+notes[i][6]*3+notes[i][7])/8
        notes[i][10]=(notes[i][2]+notes[i][4]+notes[i][8])/3
    return notes

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
    
#################################################################################################
#////////////////////////////////////////////Db_Data\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#################################################################################################

def read_in():
    
    lines = sys.stdin.readlines()
    #Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])

#ide=read_in()
#print(ide)
#idd=ObjectId(ide)
#print(ide)
idd=ObjectId("599c72288852456eefb4d3ab")
MONGO_HOST = "127.0.0.1"
MONGO_DB = "panda"
names=['id','bruitFlex','bruitRot','symflex','symrot','cor','Note MAx','fftFlex','fftRot','Noteflex','NoteRot']

Ref=110

client = pymongo.MongoClient("127.0.0.1",27017)
db = client[MONGO_DB]
collection1 = db.sceances
collection2 = db.exercises

#result = collection.find({'$and':[{"exercise":exer},{'user.firstname':fname},{"_id":idd}]})

result = collection1.find({"_id":idd})

x = []
for i in result:
    x.append(i)
#print(x)
    

#idee=str(x[0]["_id"])
exerids=[]
for i in x[0]["exer_ids"]:
    exerids.append(ObjectId("%s"%str(i)))

    
result2 = collection2.find({"_id":idd})
result2=collection2.find({"_id":{'$in': exerids }})


x2 = []
for i in result2:
    x2.append(i)

PNum=0
for mmm in range(0,len(x2)):
    d=[]
    spliteddata=[]
    decrypteddata=[]
    pp=mmm   
    #len(x[0])
    Data=x2[pp]["messages"]
#    idd=x[0]["%s"%pp]["_id"]
    exer=x2[pp]["exercise"]    
    #Data=x[0]["messages"]
    #idd=x[0]["_id"]
    #exer=x[0]["exercise"]  
    fields = Data.split('],')
    split1,split4,split3,split2=[],[],[],[]
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
    #[flexion,tremblement]=Alfred(hexdata)    
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
        flexion[i][2]=flexion[i][0]-flexion[i][1]
    #if flexion[i][2]>180:
    #flexion[i][2]=360-flexion[i][2]
        rotation[i][0]=asin(data[i][0]/sqrt(data[i][0]*data[i][0]+data[i][1]*data[i][1]+data[i][2]*data[i][2]))*180/pi
        rotation[i][1]=asin(-data[i][8]/sqrt(data[i][6]*data[i][6]+data[i][7]*data[i][7]+data[i][8]*data[i][8]))*180/pi
        rotation[i][2]=rotation[i][0]-rotation[i][1]
    
    #plt.plot([flexion[i][2] for i in range(len(flexion))])
    flex=[flexion[i][2] for i in range(len(flexion))]
    rot=[rotation[i][2] for i in range(len(rotation))]
    
    del rotation
    
    filtreflex=savgol_filter(flex,21,5)
    filtrerot=savgol_filter(rot,21,5)
    [filtreflex,Trembl]=Alfred(hexdata)
    Datatime=getTime()
    ##########################################################################################
    #//////////////////////////////  Execution    \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
    ##########################################################################################
    print(exer)    
    Series=getAleatoireSeries()           
    exportPdf(Series,x2)    
    del(Series)                
    tmps2=time.clock()
    
    plt.close('all')
    print("%f\n" %(tmps2-tmps1))   



SendMail()    
