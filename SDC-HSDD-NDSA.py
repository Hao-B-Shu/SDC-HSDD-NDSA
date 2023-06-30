from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distinctipy import distinctipy

class SDCHSDDNDSA():#MinKNNClusterPoint and  MinClusterPoint should >=SearchNeiborK should >=RhoCalculateK, IsoNeiborK

    def __init__(self,data, MaxIsoPointRho=0.07,MinClusterPoint=35,MidResult=False,Dataname='Cluster Index: 1',DataClustername='ID Cluster',
                 SearchNeiborK=7,RhoCalculateK=4,eps=0.075,IsoNeiborK=4,adjust=0.005,Mineps=0.045,Maxeps=0.075,MinKNNClusterPoint=7,IOC=True):#data should be a matrix in ndarray form
        self.data=data
        self.MaxIsoPointRho = MaxIsoPointRho
        self.MinClusterPoint = MinClusterPoint
        self.MidResult = MidResult
        self.Dataname = Dataname
        self.SearchNeiborK = SearchNeiborK
        self.RhoCalculateK = RhoCalculateK
        self.eps = eps
        self.IsoNeiborK = IsoNeiborK
        self.DataClustername=DataClustername
        self.adjust = adjust
        self.Mineps = Mineps
        self.Maxeps=Maxeps
        self.MinKNNClusterPoint=MinKNNClusterPoint
        self.IOC=IOC
        self.TotalIP=[]

    def DistanceMatrix(self,data):
        Datanum=[]
        L=len(data)
        for i in range (0,L):
            Datanum.append(i)
        m=np.zeros((L,L))
        for i in range (0,L):
            for j in range (0,L):
                a=np.linalg.norm(np.array(data[i]) - np.array(data[j]))
                m[i][j]=a
        m=np.row_stack((m,Datanum))
        return m

    def KNN(self,point,dis):
        L=len(dis)-1
        p=np.argsort(dis[point])
        epsr = 1.001*dis[point][p[self.SearchNeiborK - 1]]
        r=[]
        for i in range (0,L):
            if dis[point][p[i]]<epsr:
               r.extend([p[i]])
            else:
                break
        r1=0
        for i in range(1,self.RhoCalculateK):
                r1=r1+dis[point][p[i]]
        is1 = 0
        for i in range(1,self.IsoNeiborK):
                is1=is1+dis[point][p[i]]
        r2=r1/(self.RhoCalculateK-1)
        is2 = is1 / (self.IsoNeiborK - 1)
        rho=1/(r2*r2*np.pi)
        isrho = 1 / (is2 * is2 * np.pi)
        return rho,r,isrho

    def DensityDifferential(self,rho,kNN):
        Drho = np.zeros((len(rho), len(rho)))
        for i in range (0,len(rho)):
            for j in range (1,len(kNN[i])):
                a=kNN[i][j]
                Drho[i][a]=(rho[i]-rho[a])
        return Drho

    def IsoCluster(self,rho):
        L=len(rho)
        IP=[]
        if self.MaxIsoPointRho>0:
            for i in range (0,L):
                if rho[i]<self.MaxIsoPointRho:
                    IP.extend([i])
        return IP

    def ClusterSingle(self,Data,Cri,rho,close,IP,EPS=0.075,mode='SD'):
        Rho=list(rho)
        L=len(Data)
        if mode!='SD':
            delta=4
        else:
            delta=EPS
        C=[]
        CP = [-1 for i in range(0, L)]
        UCNumber=L
        if len(IP)!=0:
            C.extend([IP])
            UCNumber=UCNumber-len(IP)
            for i in range (0,len(IP)):
                CP[IP[i]]=0
                Rho[IP[i]] = -1
        else:
            C.extend([[]])
        Clusternum=0
        while UCNumber>0:
            Clust=Rho.index(max(Rho))
            Rho[Clust]=-1
            TC=[Clust]
            Clusternum=Clusternum+1
            CP[Clust]=Clusternum
            Tem=[Clust]
            UCNumber=UCNumber-1
            while len(Tem)!=0:
                a=Tem[0]
                c=close[a]
                for i in range (1,len(c)):
                    b=c[i]
                    if CP[b]==-1:
                        rhoab=Cri[a][b]
                        EkNNb=close[b]
                        for j in range (1,len(EkNNb)):
                            e = EkNNb[j]
                            if CP[e]!=0:
                                rhobe=Cri[b][e]
                                if abs(rhoab-rhobe)<=delta:
                                    TC.extend([b])
                                    Rho[b] = -1
                                    CP[b] = Clusternum
                                    Tem.extend([b])
                                    UCNumber=UCNumber-1
                                    Tem = set(Tem)
                                    Tem = list(Tem)
                                    break
                Tem.remove(a)
            C.extend([TC])
        return C,CP

    def Merge(self,rclu,rcp,dis,miniclusterpoint):
        MCP=miniclusterpoint
        clu=deepcopy(rclu)
        cp=deepcopy(rcp)
        LC=[len(clu[i]) for i in range (0,len(clu))]
        CFinal = [clu[0]]
        lc=deepcopy(LC)
        lc.remove(lc[0])
        if len(lc)>=2:
            if min(lc)<MCP:
                c = []
                if max(lc) < MCP:
                    for i in range (1,len(clu)):
                        c.extend(clu[i])
                    CFinal.extend([c])
                else:
                    for i in range(1, len(clu)):
                        if len(clu[i])<MCP:
                            c.extend(clu[i])
                            clu[i] = []
                    D=[]
                    D.extend(clu[0])
                    D.extend(c)
                    Mergedis = np.delete(dis, D, 1)
                    for i in range (0,len(c)):
                        P=c[i]
                        dP=Mergedis[P]
                        clord=np.argsort(np.array([dP[0],dP[1]]))
                        closestPoint = clord[1]
                        closest=dP[closestPoint]
                        for j in range (0,len(dP)):
                            if 0<dP[j]<closest:
                                closest=dP[j]
                                closestPoint=j
                        cp[P]=cp[int(Mergedis[len(Mergedis)-1][closestPoint])]
                    for i in range (0,len(c)):
                        P=c[i]
                        clu[cp[P]].extend([P])
                    for i in range (1,len(clu)):
                        if len(clu[i])!=0:
                            CFinal.extend([clu[i]])
            else:
                CFinal=clu
        else:
            CFinal=clu

        return CFinal

    def ClusterTotal(self,data,dataname,Eps=0.075,MinEps=0.045,KON=False,adjust=0.005,mode='SD'):
        L=len(data)
        if mode!='SD':
            adjust=0
            MCP=self.MinKNNClusterPoint
        else:
            MCP=self.MinClusterPoint
        kNNpoint = []
        kNrho = []
        Nrho = []
        iskNrho = []
        Nisrho = []
        Distance = self.DistanceMatrix(data)
        Datanum=[]
        for j in range(0, L):
            Datanum.append(j)
            Tem = self.KNN(j,Distance)
            kNNpoint.extend([Tem[1]])
            kNrho.extend([Tem[0]])
            iskNrho.extend([Tem[2]])
        MRho = max(kNrho)
        AisRho = np.sum(np.array(iskNrho)) / len(iskNrho)
        for j in range(0, L):
            Nrho.extend([kNrho[j] / (MRho)])
            Nisrho.extend([iskNrho[j] / (AisRho)])
        DRho = self.DensityDifferential(rho=Nrho,kNN=kNNpoint)
        if KON==True:
            IP=self.IsoCluster(rho=Nisrho)
        else:
            IP=[]
        if adjust>0:
            EPS=MinEps
        else:
            EPS = Eps
        if L - len(IP) >= MCP:
            SeedNumber=0
            while adjust>0 and mode=='SD':
                C= self.ClusterSingle(Data=Datanum, Cri=DRho, rho=Nrho,close=kNNpoint,IP=IP, EPS=EPS, mode=mode)[0]
                C.remove(C[0])
                SeedC=0
                for i in range (0,len(C)):
                    if len(C[i])>=self.MinClusterPoint:
                        SeedC=SeedC+1
                if SeedC>=SeedNumber:
                    SeedNumber=SeedC
                    EPS=EPS+adjust
                else:
                    EPS=EPS-adjust
                    break
                if EPS>=self.Maxeps:
                    EPS=self.Maxeps
                    break
        [C, CP] = self.ClusterSingle(Data=Datanum,Cri=DRho,rho=Nrho,close=kNNpoint,IP=IP,EPS=EPS,mode=mode)
        if mode=='KNN':
            P=[]
            P.extend(IP)
            i=1
            while i<len(C):
                if len(C[i])<MCP:
                    P.extend(C[i])
                    C.remove(C[i])
                else:
                    i=i+1
            C[0]=P
            CFinal=C
        else:
            CFinal = self.Merge(rclu=C,rcp=CP,dis=Distance,miniclusterpoint=MCP)

        Cxy = []
        for i in range(0, len(CFinal)):
            Ci = []
            for j in range(0, len(CFinal[i])):
                Ci.extend([data[CFinal[i][j]]])
            Cxy.extend([[np.array(Ci), dataname + ',' + str(i)]])
            if self.MidResult == True:
                print('Length of Cluster ',str(i),' =',len(Ci))
        return Cxy,EPS

    def Plot2(self,C,Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):
        plt.figure()
        colors = distinctipy.get_colors(len(C))
        for i in range(0, len(C)):
            x = []
            y = []
            for j in range(0, len(C[i][0])):
                x.extend([C[i][0][j][0]])
                y.extend([C[i][0][j][1]])
            plt.scatter(x, y, color=colors[i], label=C[i][1] + ' N=' + str(len(C[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=1,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()

    def Plot1(self, C, Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):
        plt.figure()
        colors = distinctipy.get_colors(len(C))
        for i in range(0, len(C)):
            x = []
            y = []
            for j in range(0, len(C[i][0])):
                x.extend([C[i][0][j][0]])
                y.extend([0])
            plt.scatter(x, y, color=colors[i], label=C[i][1] + ' N=' + str(len(C[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()

    def ClusterRefine(self):
        CAF = []
        CRefine = []
        mode = 'KNN'
        CRC = self.ClusterTotal(data=self.data, dataname=self.Dataname, KON=False,adjust=0, Eps=4,mode=mode)[0]
        a=0
        if len(CRC[0])==0:
            CRC.remove(CRC[0])
            a=1
        if self.MidResult == True:
            print('Cluster Number=',len(CRC))
            if len(self.data[0]) == 2:
                self.Plot2(C=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
            elif len(self.data[0]) == 1:
                self.Plot1(C=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
        if a==0:
            if self.IOC==True:
                self.TotalIP.extend(CRC[0][0])
            else:
                CAF.extend([CRC[0]])
            CRC.remove(CRC[0])
        for i in range (0,len(CRC)):
            if len(CRC[i][0])<self.MinClusterPoint:
                T=self.ClusterTotal(data=CRC[i][0], dataname=CRC[i][1], KON=True, adjust=0, Eps=4, mode=mode)[0]
                if self.MidResult == True:
                    print('Cluster Number=',len(T))
                    if len(CRC[i][0][0]) == 2:
                        self.Plot2(C=T, Name=CRC[i][1] + ' Final', StrEps=' eps=' + str(4), bbox_to_anchor=(1, 1),
                                   loc=2, handletextpad=0.05, labelspacing=0.3)
                    elif len(CRC[i][0][0]) == 1:
                        self.Plot1(C=T, Name=CRC[i][1] + ' Final', StrEps=' eps=' + str(4), bbox_to_anchor=(1, 1),
                                   loc=2, handletextpad=0.05, labelspacing=0.3)
                if self.IOC==True:
                    self.TotalIP.extend(T[0][0])
                    T.remove(T[0])
                    CAF.extend(T)
                else:
                    CAF.extend(T)
            else:
                CRefine.extend([CRC[i]])
        mode = 'SD'
        while len(CRefine)>0:
            DataR = CRefine[0]
            [CRC,EPS]=self.ClusterTotal(data=DataR[0],dataname=DataR[1],KON=False,Eps=self.eps,MinEps=self.Mineps,adjust=self.adjust,mode=mode)
            CRC.remove(CRC[0])
            CRefine.remove(CRefine[0])
            if self.MidResult == True:
                print('Cluster Number=',len(CRC))
                print('EPS=',EPS)
                if len(DataR[0][0]) == 2:
                    self.Plot2(C=CRC,Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                elif len(DataR[0][0]) == 1:
                    self.Plot1(C=CRC, Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
            if len(CRC)<=1:
                T=self.ClusterTotal(data=DataR[0],dataname=DataR[1],adjust=0,Eps=EPS,KON=True,mode=mode)[0]
                if self.MidResult == True:
                    print('Cluster Number=',len(T))
                    if len(DataR[0][0]) == 2:
                        self.Plot2(C=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                    elif len(DataR[0][0]) == 1:
                        self.Plot1(C=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                if self.IOC==True:
                    self.TotalIP.extend(T[0][0])
                    T.remove(T[0])
                    CAF.extend(T)
                else:
                    CAF.extend(T)
            else:
                CRefine.extend(CRC)
        if self.IOC == True:
            CAF.insert(0,[np.array(self.TotalIP),'IsoPoint'])
        i=0
        while i<len(CAF):
            if len(CAF[i][0])==0:
                CAF.remove(CAF[i])
            else:
                print('Cluster Length=',len(CAF[i][0]))
                i=i+1

        print('Final Cluster Number=', len(CAF))
        if len(self.data[0]) == 2:
            self.Plot2(C=CAF, Name=self.DataClustername,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.01,labelspacing=0.3)
        elif len(self.data[0]) == 1:
            self.Plot1(C=CAF, Name=self.DataClustername,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.01,labelspacing=0.3)
        return CAF

df=pd.read_csv('Sample/CombineNoise100.csv', index_col=0).values
ClusterSample=SDCHSDDNDSA(data=df, MaxIsoPointRho=0.07,MinClusterPoint=35,MidResult=False,Dataname='ID',DataClustername='Cluster',SearchNeiborK=7,RhoCalculateK=4,
                  eps=0.075,IsoNeiborK=4,adjust=0.005,Mineps=0.045,Maxeps=0.075,MinKNNClusterPoint=7,IOC=True)
ClusterSample.ClusterRefine()
