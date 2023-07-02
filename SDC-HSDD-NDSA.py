import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distinctipy import distinctipy
from sklearn.neighbors import KNeighborsClassifier

class SDCHSDDNDSA():#The SDC-HSDD-NDSA class. MinClusterPoint should larger than max(SearchNeiborK,RhoCalculateK,IsoNeiborK)

    def __init__(self,data, MaxIsoPointRho=0.07,MinClusterPoint=35,MinFractionPoint=0.01,MidResult=False,Dataname='Cluster Index: 1',DataClustername='ID Cluster',
                 SearchNeiborK=7,RhoCalculateK=4,eps=0.075,IsoNeiborK=4,adjust=0.005,Mineps=0.045,Maxeps=0.075,MinKNNClusterPoint=7,IOC=True):
        self.data=data
        self.MaxIsoPointRho = MaxIsoPointRho
        self.MinClusterPoint = MinClusterPoint
        self.MidResult = MidResult
        self.Dataname = Dataname
        self.SearchNeiborK = SearchNeiborK
        self.RhoCalculateK = RhoCalculateK
        self.eps = eps
        self.IsoNeiborK = IsoNeiborK
        self.MaxK=max([self.SearchNeiborK,self.RhoCalculateK,self.IsoNeiborK])
        self.DataClustername=DataClustername
        self.adjust = adjust
        self.Mineps = Mineps
        self.Maxeps=Maxeps
        self.MinKNNClusterPoint=MinKNNClusterPoint
        self.IOC=IOC
        self.TotalIP=[]
        self.MinFractionPoint=MinFractionPoint

    def RhoCalculate(self,kNNDistance):#The function to calculate densities. Return the densities of all data
        L=len(kNNDistance)
        kNrho = []
        IskNrho = []
        if self.RhoCalculateK==self.IsoNeiborK:
            for i in range(0, L):
                r = 0
                for j in range(1, self.RhoCalculateK):
                    r = r + kNNDistance[i, j]
                r = r / (self.RhoCalculateK - 1)
                rho = 1 / (r * r * np.pi)
                kNrho.append(rho)
            MRho = max(kNrho)
            kNrho = np.array(kNrho)
            AIsRho = np.sum(kNrho) / L
            Nrho = kNrho / MRho
            Nisrho = kNrho / AIsRho
        else:
            for i in range(0, L):
                r=0
                for j in range (1,self.RhoCalculateK):
                    r=r+kNNDistance[i,j]
                r = r / (self.RhoCalculateK - 1)
                rho = 1 / (r * r * np.pi)
                kNrho.append(rho)
                r = 0
                for j in range(1, self.IsoNeiborK):
                    r = r + kNNDistance[i, j]
                r = r / (self.IsoNeiborK - 1)
                rho = 1 / (r * r * np.pi)
                IskNrho.append(rho)
            MRho = max(kNrho)
            kNrho=np.array(kNrho)
            IskNrho=np.array(IskNrho)
            AIsRho = np.sum(IskNrho) / L
            Nrho=kNrho /MRho
            Nisrho=IskNrho/AIsRho
        return Nrho,Nisrho

    def DensityDifferential(self,Nrho,kNNPoint):#The function to calculate the differentials of densities. Return the differentials to the kNN points of each point
        Drho = []
        for i in range (0,len(Nrho)):
            Drhoi=[0]
            for j in range (1,self.SearchNeiborK):
                a=kNNPoint[i][j]
                Drhoi.append(Nrho[i]-Nrho[a])
            Drho.append(np.array(Drhoi))
        return Drho

    def IsoCluster(self,Nisrho):#The function to detect isolated points. Return the isolated-point cluster
        L=len(Nisrho)
        IP=[]
        if self.MaxIsoPointRho>0:
            for i in range (0,L):
                if Nisrho[i]<self.MaxIsoPointRho:
                    IP.extend([i])
        return IP

    def ClusterSingle(self,Drho,Nrho,kNNPoint,IP,EPS=0.075,mode='SD'):#The core algorithm. Return the cluster result without merging
        Rho=list(Nrho)
        L=len(Nrho)
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
                c=kNNPoint[a]
                for i in range (1,self.SearchNeiborK):
                    b=c[i]
                    if CP[b]==-1:
                        rhoab=Drho[a][i]
                        EkNNb=kNNPoint[b]
                        for j in range (1,self.SearchNeiborK):
                            e = EkNNb[j]
                            if CP[e]!=0:
                                rhobe=Drho[b][j]
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

    def Merge(self,Clu,CP,MCP,data,mode):#The merge function. Return clusters after merging
        if mode=='KNN':
            i = 1
            while i < len(Clu):
                if len(Clu[i]) < MCP:
                    Clu[0].extend(Clu[i])
                    Clu.pop(i)
                else:
                    i = i + 1
            CFinal = Clu
        else:
            LC=[len(Clu[i]) for i in range (0,len(Clu))]
            if len(LC)>=3:
                if min(LC[1:len(LC)])<MCP:
                    CFinal = [Clu[0]]
                    c = []
                    if max(LC[1:len(LC)]) < MCP:
                        for i in range (1,len(Clu)):
                            c.extend(Clu[i])
                        CFinal.append(c)
                    else:
                        datat=[]
                        datay=[]
                        for i in range(1, len(Clu)):
                            if LC[i]<MCP:
                                c.extend(Clu[i])
                                Clu[i] = []
                            else:
                                for j in range (0,LC[i]):
                                    datat.append(data[Clu[i][j]])
                                    datay.append(CP[Clu[i][j]])
                        datac=[]
                        for i in range (0,len(c)):
                            datac.append(data[c[i]])
                        kNNMerge=KNeighborsClassifier(n_neighbors=1)
                        kNNMerge.fit(datat,datay)
                        Mergec=kNNMerge.predict(datac)
                        for i in range (0,len(c)):
                            Clu[Mergec[i]].append(c[i])
                        for i in range (1,len(Clu)):
                            if len(Clu[i])!=0:
                                CFinal.append(Clu[i])
                else:
                    CFinal=Clu
            else:
                CFinal=Clu
        return CFinal

    def ClusterTotal(self,data,dataname,Eps=0.075,MinEps=0.045,KON=False,adjust=0.005,mode='SD'):#Cluster function in each hierarchy. Return the cluster result
        L=len(data)
        if mode=='KNN':
            adjust=0
            MCP=self.MinKNNClusterPoint
        else:
            MCP=max([self.MinClusterPoint,L*self.MinFractionPoint])
        kNNClass=KNeighborsClassifier(n_neighbors=self.MaxK)
        kNNClass.fit(data,[0]*L)
        kNNNeighbor=kNNClass.kneighbors(data)
        kNNDistance=kNNNeighbor[0]
        kNNpoint =kNNNeighbor[1]
        [Nrho,Nisrho]= self.RhoCalculate(kNNDistance)
        DRho = self.DensityDifferential(Nrho=Nrho,kNNPoint=kNNpoint)
        if KON==True:
            IP=self.IsoCluster(Nisrho=Nisrho)
        else:
            IP=[]
        if adjust>0:
            EPS=MinEps
        else:
            EPS = Eps
        if L - len(IP) >= MCP:
            SeedNumber=0
            while adjust>0 and mode=='SD':
                C= self.ClusterSingle(Drho=DRho, Nrho=Nrho,kNNPoint=kNNpoint,IP=IP, EPS=EPS, mode=mode)[0]
                C.remove(C[0])
                SeedC=0
                for i in range (0,len(C)):
                    if len(C[i])>=MCP:
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
        [C, CP] = self.ClusterSingle(Drho=DRho,Nrho=Nrho,kNNPoint=kNNpoint,IP=IP,EPS=EPS,mode=mode)
        CFinal = self.Merge(Clu=C,CP=CP,MCP=MCP,data=data,mode=mode)
        Cxy = []
        for i in range(0, len(CFinal)):
            Ci = []
            for j in range(0, len(CFinal[i])):
                Ci.append(data[CFinal[i][j]])
            Cxy.append([np.array(Ci), dataname + ',' + str(i)])
            if self.MidResult == True:
                print('Length of cluster ',str(i),' =',len(Ci))
        return Cxy,EPS

    def Plot2(self,CLU,Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):#2-dim plot function
        plt.figure()
        colors = distinctipy.get_colors(len(CLU))
        for i in range(0, len(CLU)):
            x = []
            y = []
            for j in range(0, len(CLU[i][0])):
                x.extend([CLU[i][0][j][0]])
                y.extend([CLU[i][0][j][1]])
            plt.scatter(x, y, color=colors[i], label=CLU[i][1] + ' N=' + str(len(CLU[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=1,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()

    def Plot1(self, CLU, Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):#1-dim plot function
        plt.figure()
        colors = distinctipy.get_colors(len(CLU))
        for i in range(0, len(CLU)):
            x = []
            y = []
            for j in range(0, len(CLU[i][0])):
                x.extend([CLU[i][0][j][0]])
                y.extend([0])
            plt.scatter(x, y, color=colors[i], label=CLU[i][1] + ' N=' + str(len(CLU[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()

    def ClusterRefine(self):#Cluster method. Return final clusters
        CAF = []
        CRefine = []
        mode = 'KNN'
        CRC = self.ClusterTotal(data=self.data, dataname=self.Dataname, KON=False,Eps=4,mode=mode)[0]
        a=0
        if len(CRC[0])==0:
            CRC.pop(0)
            a=1
        if self.MidResult == True:
            print('Cluster Number=',len(CRC))
            if len(self.data[0]) == 2:
                self.Plot2(CLU=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
            elif len(self.data[0]) == 1:
                self.Plot1(CLU=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
        if a==0:
            if self.IOC==True:
                self.TotalIP.extend(CRC[0][0])
            else:
                CAF.append(CRC[0])
            CRC.pop(0)
        for i in range (0,len(CRC)):
            if len(CRC[i][0])<self.MinClusterPoint:
                T=self.ClusterTotal(data=CRC[i][0], dataname=CRC[i][1], KON=True, adjust=0, Eps=4, mode=mode)[0]
                if self.MidResult == True:
                    print('Cluster Number=',len(T))
                    if len(CRC[i][0][0]) == 2:
                        self.Plot2(CLU=T, Name=CRC[i][1] + ' Final', StrEps=' eps=' + str(4), bbox_to_anchor=(1, 1),
                                   loc=2, handletextpad=0.05, labelspacing=0.3)
                    elif len(CRC[i][0][0]) == 1:
                        self.Plot1(CLU=T, Name=CRC[i][1] + ' Final', StrEps=' eps=' + str(4), bbox_to_anchor=(1, 1),
                                   loc=2, handletextpad=0.05, labelspacing=0.3)
                if self.IOC==True:
                    self.TotalIP.extend(T[0][0])
                    T.pop(0)
                    CAF.extend(T)
                else:
                    CAF.extend(T)
            else:
                CRefine.extend([CRC[i]])
        mode = 'SD'
        while len(CRefine)>0:
            DataR = CRefine[0]
            [CRC,EPS]=self.ClusterTotal(data=DataR[0],dataname=DataR[1],KON=False,Eps=self.eps,MinEps=self.Mineps,adjust=self.adjust,mode=mode)
            CRC.pop(0)
            CRefine.pop(0)
            if self.MidResult == True:
                print('Cluster Number=',len(CRC))
                print('EPS=',EPS)
                if len(DataR[0][0]) == 2:
                    self.Plot2(CLU=CRC,Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                elif len(DataR[0][0]) == 1:
                    self.Plot1(CLU=CRC, Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
            if len(CRC)<=1:
                T=self.ClusterTotal(data=DataR[0],dataname=DataR[1],adjust=0,Eps=EPS,KON=True,mode=mode)[0]
                if self.MidResult == True:
                    print('Cluster Number=',len(T))
                    if len(DataR[0][0]) == 2:
                        self.Plot2(CLU=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                    elif len(DataR[0][0]) == 1:
                        self.Plot1(CLU=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                if self.IOC==True:
                    self.TotalIP.extend(T[0][0])
                    T.pop(0)
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
                CAF.pop(i)
            else:
                print('Cluster Length=',len(CAF[i][0]))
                i=i+1
        print('Final Cluster Number=', len(CAF))
        if len(self.data[0]) == 2:
            self.Plot2(CLU=CAF, Name=self.DataClustername,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.01,labelspacing=0.3)
        elif len(self.data[0]) == 1:
            self.Plot1(CLU=CAF, Name=self.DataClustername,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.01,labelspacing=0.3)
        return CAF

df=pd.read_csv('Sample/ThreeGauss185.csv', index_col=0).values#The data set should input here

ClusterSample=SDCHSDDNDSA(data=df, MaxIsoPointRho=0.07,MinClusterPoint=35,MinFractionPoint=0.01,MidResult=False,Dataname='ID',DataClustername='Cluster',SearchNeiborK=7,RhoCalculateK=4,
                  eps=0.075,IsoNeiborK=4,adjust=0.005,Mineps=0.045,Maxeps=0.075,MinKNNClusterPoint=7,IOC=True)

ClusterSample.ClusterRefine()
