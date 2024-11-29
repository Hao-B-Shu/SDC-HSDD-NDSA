import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from distinctipy import distinctipy
from sklearn.neighbors import KNeighborsClassifier

# 先KNN，小于MKCP的为散点，在MKCP和MCP之间的单成类，大于的加细
# 离群点KNN时可以按区域成类，只有离大集很近但没有临近的才被视为散点

class SDCHSDDNDSA():#要求最小类长度不小于检索邻域，代码中一般data表示坐标数据，Data表示标号数据，距离矩阵最后增加一行标号，称为列名称
    # data为数据集(坐标必须为向量（一维要括号变成一维向量，不能是值）)，numpy: [N,D]                                          True表示返回中间值（获取每次加细的结果，包括类数，类点数，散点类大小，作图）
    def __init__(self,data, MaxIsoPointRho=0.07, MinClusterPoint=35,       MinFractionPoint=0.0,                  MidResult=False,Dataname='Cluster 1',DataClustername='ID Cluster',
                 #          正则化后散点密度上限，   mode=SD时最小类点数最小下界，  mode=SD时最小类点数占数据集大小的比例，最终取MCP=max(N*比例，MinClusterPoint)（每次加细重新算当前比例）
                 SearchNeiborK=7,RhoCalculateK=4,eps=0.075,                        IsoNeiborK=4,      adjust=0.005,Mineps=0.045,Maxeps=0.075,MinKNNClusterPoint=7,IOC=True,       Redistribute_Isolated_Clusters=False, Plot=True):
                 # kNN的k，       计算密度的k，     正则密度二阶差上限（小于这个值扩类），    判断散点时计算密度的k，自适应步长，    最小eps，     最大eps，     mode=kNN时最小类点数，  True：散点类合并， True：散点重新分配到最近类，              True：1D和2D时作散点图
        self.data=data
        self.datadim=len(self.data[0])#数据的维数
        self.MaxIsoPointRho = MaxIsoPointRho
        self.MinClusterPoint = MinClusterPoint
        self.MidResult = MidResult
        self.Dataname = Dataname
        self.SearchNeiborK = SearchNeiborK
        self.RhoCalculateK = RhoCalculateK
        self.eps = eps
        self.IsoNeiborK = IsoNeiborK
        self.MaxK=max([self.SearchNeiborK,self.RhoCalculateK,self.IsoNeiborK])#需要的近邻最大数
        self.DataClustername=DataClustername
        self.adjust = adjust
        self.Mineps = Mineps
        self.Maxeps=Maxeps
        self.MinKNNClusterPoint=MinKNNClusterPoint
        self.RDIC=Redistribute_Isolated_Clusters
        if self.RDIC==True:
            self.IOC = True
        else:
            self.IOC=IOC
        self.Plot=Plot
        self.TotalIP=[]#全体散点类
        self.MinFractionPoint=MinFractionPoint#最小类占总点数比例，MCP=max(MinClusterPoint,MinFractionPoint*总点数)

    def RhoCalculate(self,kNNDistance):
        L=len(kNNDistance)
        kNrho = []  # 所有密度
        IskNrho = []  # 所有计算散点的密度
        if self.RhoCalculateK==self.IsoNeiborK:#只需算一次
            for i in range(0, L):
                r = 0  # 总距离
                for j in range(1, self.RhoCalculateK):
                    r = r + kNNDistance[i, j]
                r = r / (self.RhoCalculateK - 1)  # 平均距离（计算密度）
                rho = 1 / (r ** self.datadim)  # 密度
                kNrho.append(rho)  # 密度
            MRho = max(kNrho)  # 最大密度
            kNrho = np.array(kNrho)
            AIsRho = np.sum(kNrho) / L  # 计算散点的平均密度
            Nrho = kNrho / MRho  # 密度正则化
            Nisrho = kNrho / AIsRho  # 散点用平均
        else:
            for i in range(0, L):
                r=0#总距离
                for j in range (1,self.RhoCalculateK):
                    r=r+kNNDistance[i,j]
                r = r / (self.RhoCalculateK - 1)  # 平均距离（计算密度）
                rho = 1 / (r **self.datadim)  # 密度
                kNrho.append(rho)  # 密度
                r = 0  # 总距离
                for j in range(1, self.IsoNeiborK):
                    r = r + kNNDistance[i, j]
                r = r / (self.IsoNeiborK - 1)  # 平均距离（计算密度）
                rho = 1 / (r **self.datadim)  # 密度
                IskNrho.append(rho)  # 密度
            MRho = max(kNrho)  # 最大密度
            kNrho=np.array(kNrho)
            IskNrho=np.array(IskNrho)
            AIsRho = np.sum(IskNrho) / L  # 计算散点的平均密度
            Nrho=kNrho /MRho# 密度正则化
            Nisrho=IskNrho/AIsRho  # 散点用平均
        return Nrho,Nisrho#返回正则密度，正则散点密度

    def DensityDifferential(self,Nrho,kNNPoint):#计算密度变化矩阵（只算有效近邻，自己和其它为0），Nrho为需计算的量，kNNPoint近邻列表
        Drho = []
        for i in range (0,len(Nrho)):#每个点
            Drhoi=[0]#i点密度差
            for j in range (1,self.SearchNeiborK):#每个有效近邻点
                a=kNNPoint[i][j]
                Drhoi.append(Nrho[i]-Nrho[a])#i到近邻rho变化
            Drho.append(np.array(Drhoi))
        return Drho#返回密度变化，只有近邻

    def IsoCluster(self,Nisrho):#杀散点，如果Nisrho小于MaxIsoPointRho，判定为散点
        L=len(Nisrho)
        IP=[]
        if self.MaxIsoPointRho>0:#不用Kill的时候直接返回空，不用检索
            for i in range (0,L):
                if Nisrho[i]<self.MaxIsoPointRho:
                    IP.extend([i])
        return IP#返回散点类

    def ClusterSingle(self,Drho,Nrho,kNNPoint,IP,EPS=0.075,mode='SD'):#分类(SD表示用eps，否则用kNN)，Data为数据集点标号列表（按密度从小到大排序），Drho为指标(密度变化矩阵)，要求Drho的差小于delta的点为一类
        Rho=list(Nrho)
        L=len(Nrho)
        if mode!='SD':
            delta=4
        else:
            delta=EPS

        C=[]#类
        CP = [-1 for i in range(0, L)]  # 记录每个点的类，初始为-1，散点为0，其它已分类点>0
        UCNumber=L#未分类点数
        if len(IP)!=0:#如果有散点
            C.extend([IP])#散点类加入类集
            UCNumber=UCNumber-len(IP)
            for i in range (0,len(IP)):#从未分类集之中移除散点
                CP[IP[i]]=0
                Rho[IP[i]] = -1  # 将该点的密度设置为-1
        else:
            C.extend([[]])#否则加入一个空类

        Clusternum=0#记录非散点类数
        while UCNumber>0:#如果还有点未分类
            Clust=Rho.index(max(Rho))#密度最大的点
            Rho[Clust]=-1#将该点的密度设置为-1，以使每次找到的都是未分类点
            TC=[Clust]#将密度最大的点分到一个新类，从密度大的开始检索
            Clusternum=Clusternum+1
            CP[Clust]=Clusternum
            Tem=[Clust]#临时，用于记录已判断点的待判断近邻（如果b加入了类，将它放入Tem以判断是否还需基于它扩张）
            UCNumber=UCNumber-1
            while len(Tem)!=0:#如果有需要判断是否吸点的点
                a=Tem[0]#取首个a
                c=kNNPoint[a]#a的近邻
                for i in range (1,self.SearchNeiborK):#对a的每个近邻点b，不包括自己，判断是否需要吸收b
                    b=c[i]#a的第i个近邻b的标号
                    if CP[b]==-1:
                        rhoab=Drho[a][i]#a和第i个近邻b的密度变化
                        EkNNb=kNNPoint[b]#b的近邻
                        for j in range (1,self.SearchNeiborK):#对b的第j个有效
                            e = EkNNb[j]  # e为近邻点中的第j个
                            if CP[e]!=0:#如果不是散点
                                rhobe=Drho[b][j]#b和第j个近邻e的密度变化
                                if abs(rhoab-rhobe)<=delta:#如果变化够小
                                    TC.extend([b])#将b放入类
                                    Rho[b] = -1  # 将该点的密度设置为-1，以使每次找到的都是未分类点
                                    CP[b] = Clusternum#记录b的类号
                                    Tem.extend([b])#放入待判断集合
                                    UCNumber=UCNumber-1
                                    Tem = set(Tem)  # 去重复
                                    Tem = list(Tem)  # 变回列表，因为要检索
                                    break
                Tem.remove(a)#a的近邻点都已判断，将a从待判断集中移除
            C.extend([TC])#获取类

        return C,CP#返回类和点的类号

    def Merge(self,Clu,CP,MCP,data,mode):#Clu为类（标号）(Merge后会变，但没关系，因为不用用之前的)，不带名字，Clu[i]为第i类标号列表，要求首类为散点，无论是否为空，CP为每点对应的类，MCP为类最小点数
        if mode=='KNN':
            i = 1
            while i < len(Clu):
                if len(Clu[i]) < MCP:
                    Clu[0].extend(Clu[i])
                    Clu.pop(i)  # 删除第i项
                else:
                    i = i + 1
            CFinal = Clu
        else:#mode=SD
            LC=[len(Clu[i]) for i in range (0,len(Clu))]#类大小
            if len(LC)>=3:#有两个不是散点的类才有合并与否
                if min(LC[1:len(LC)])<MCP:#如果有小类
                    CFinal = [Clu[0]]  # 合并后结果，首为为散点类，无论是否为空（因为作图时可能会指定颜色），如果都是散点，最后只有散点类
                    c = []  # 小类合并成一类
                    if max(LC[1:len(LC)]) < MCP:  # 所有类都不够大
                        for i in range (1,len(Clu)):#检索所有除散点的类
                            c.extend(Clu[i])#加入c
                        CFinal.append(c)#将c加入CFinal
                    else:#否则（有大类）
                        datat=[]#kNN训练集
                        datay=[]#kNN训练集标签
                        for i in range(1, len(Clu)):  # 检索所有除散点的类
                            if LC[i]<MCP:#小的拆掉
                                c.extend(Clu[i])  # 加入待合并
                                Clu[i] = []  # 此类设置为空（不能移除，因为类号可能变）
                            else:#大的提取标签
                                for j in range (0,LC[i]):
                                    datat.append(data[Clu[i][j]])#加入坐标
                                    datay.append(CP[Clu[i][j]])#加入类标
                        datac=[]#c中点的坐标
                        for i in range (0,len(c)):#对c的每一个点i
                            datac.append(data[c[i]])#i的坐标
                        kNNMerge=KNeighborsClassifier(n_neighbors=1)
                        kNNMerge.fit(datat,datay)
                        Mergec=kNNMerge.predict(datac)#c的点的新类标
                        for i in range (0,len(c)):#对于c的每一个点
                            Clu[Mergec[i]].append(c[i])
                        for i in range (1,len(Clu)):#将非空类加入CFinal
                            if len(Clu[i])!=0:
                                CFinal.append(Clu[i])
                else:#没有小类
                    CFinal=Clu#最终类就是原来的类
            else:#没有两类非散点
                CFinal=Clu#最终类就是原来的类

        return CFinal#返回最终类

    def ClusterTotal(self,data,dataname,Eps=0.075,MinEps=0.045,KON=False,adjust=0.005,mode='SD'):#KON表示是否杀散点，data无名
        L=len(data)
        if mode=='KNN':
            adjust=0
            MCP=self.MinKNNClusterPoint#其实KNN时可以直接用MinKNNClusterPoint而无需设置，但这样可以方便以后改标准
        else:
            MCP=max([self.MinClusterPoint,L*self.MinFractionPoint])

        kNNClass=KNeighborsClassifier(n_neighbors=self.MaxK)
        kNNClass.fit(data,[0]*L)
        kNNNeighbor=kNNClass.kneighbors(data)
        kNNDistance=kNNNeighbor[0]#kNN距离
        kNNpoint =kNNNeighbor[1]   # kNN近点标号
        [Nrho,Nisrho]= self.RhoCalculate(kNNDistance)  # 正则密度，正则散点计算密度
        DRho = self.DensityDifferential(Nrho=Nrho,kNNPoint=kNNpoint)  # 近邻密度变化矩阵

        if KON==True:
            IP=self.IsoCluster(Nisrho=Nisrho)
        else:
            IP=[]
        if adjust>0:
            EPS=MinEps
        else:
            EPS = Eps

        if L - len(IP) >= MCP:  # 存在足够非散点，在SD时决定EPS
            SeedNumber=0#之前类数
            while adjust>0 and mode=='SD':
                C= self.ClusterSingle(Drho=DRho, Nrho=Nrho,kNNPoint=kNNpoint,IP=IP, EPS=EPS, mode=mode)[0]
                C.remove(C[0])
                SeedC=0#当前类数
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
        CFinal = self.Merge(Clu=C,CP=CP,MCP=MCP,data=data,mode=mode)#返回散点类和其它非空类（标号）

        Cxy = []  # 取每类坐标
        for i in range(0, len(CFinal)):
            Ci = []
            for j in range(0, len(CFinal[i])):
                Ci.append(data[CFinal[i][j]])
            Cxy.append([np.array(Ci), dataname + ',' + str(i)])  # 将每类的点标号换成坐标，将集合和名字加入，第一项是坐标类，第二项是类名字，第三项是标号类
            if self.MidResult == True:
                print('Length of cluster ',str(i),' =',len(Ci))  # 输出每类点数

        return Cxy, EPS  # Cxy中是类列表（Cxy=[[类集(标号或坐标)，名字]])，EPS是最终使用的判定指标

    def Redistribute_Isolated_Points(self,Isolated_Clusters,Effective_Clusters,K_neighbor=1):
        #Effective_Clusters: [[类的坐标集，类名]],Isolated_Clusters: [类的坐标集，类名]
        clf = KNeighborsClassifier(n_neighbors=K_neighbor)
        X_train=[]
        Y_train=[]
        eff_cluster=[]
        for i in range (len(Effective_Clusters)):
            eff_cluster.append(list(Effective_Clusters[i][0]))
            X_train.extend(list(Effective_Clusters[i][0]))
            Y_train.extend([i]*len(Effective_Clusters[i][0]))

        clf.fit(np.array(X_train), np.array(Y_train))
        X_Pred=list(Isolated_Clusters[0])

        Y=clf.predict(X_Pred)
        for i in range (len(Y)):
            eff_cluster[Y[i]].append(X_Pred[i])

        Final=[]
        for i in range (len(eff_cluster)):
            Final.append([np.array(eff_cluster[i]),str(i)])
        return Final

    def Plot2(self,CLU,Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):#CLU为类集，CLU[i]为第i类，带名字，CLU[i][0]为第i类集合（坐标），CLU[i][1]为第i类名字，StrEps为Name后要打印的字符串，最后一个是图例位置
        plt.figure()  # 作图
        colors = distinctipy.get_colors(len(CLU))
        for i in range(0, len(CLU)):
            x = []
            y = []
            for j in range(0, len(CLU[i][0])):
                x.extend([CLU[i][0][j][0]])  # 这类第j个点的坐标
                y.extend([CLU[i][0][j][1]])
            plt.scatter(x, y, color=colors[i], label=CLU[i][1] + ' N=' + str(len(CLU[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=1,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()
        # plt.savefig('D:/Pydatas/Project PythonProcedure/Plots/New/{}.png'.format(Name))#Name

    def Plot1(self, CLU, Name,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):
        plt.figure()  # 作图
        colors = distinctipy.get_colors(len(CLU))
        for i in range(0, len(CLU)):
            x = []
            y = []
            for j in range(0, len(CLU[i][0])):
                x.extend([CLU[i][0][j][0]])  # 这类第j个点的坐标
                y.extend([0])
            plt.scatter(x, y, color=colors[i], label=CLU[i][1] + ' N=' + str(len(CLU[i][0])))
        plt.title(Name+StrEps)
        plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc,handletextpad=handletextpad,labelspacing=labelspacing)
        plt.subplots_adjust(right=SubplotsAdjust)
        plt.show()

    def ClusterRefine(self):#聚类算法的调用
        CAF = []  # 最终类
        CRefine = []  # 还要加细的类

        mode = 'KNN'
        CRC = self.ClusterTotal(data=self.data, dataname=self.Dataname, KON=False,Eps=4,mode=mode)[0]
        a=0#指示是否已处理散点类
        if len(CRC[0])==0:#没有散点类直接移除，不用作图
            CRC.pop(0)
            a=1
        if self.MidResult == True:
            print('Cluster Number=',len(CRC))  # 最终类数（散点类无论是否为空都计一类）
            if len(self.data[0]) == 2:
                self.Plot2(CLU=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
            elif len(self.data[0]) == 1:
                self.Plot1(CLU=CRC, Name=self.Dataname, StrEps=' eps=4', bbox_to_anchor=(1, 1), loc=2, handletextpad=0.05,
                           labelspacing=0.3)
        if a==0:#如果散点类不空，按IOC与否合并或单成类，作图后
            if self.IOC==True:
                self.TotalIP.extend(CRC[0][0])
            else:
                CAF.append(CRC[0])
            CRC.pop(0)
        for i in range (0,len(CRC)):#对于每个类，大的加入加细，小的加入最终类
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
                if self.IOC==True:#如果散点单独分
                    self.TotalIP.extend(T[0][0])
                    T.pop(0)
                    CAF.extend(T)
                else:
                    CAF.extend(T)
            else:
                CRefine.extend([CRC[i]])

        mode = 'SD'
        while len(CRefine)>0:#有加细类
            DataR = CRefine[0]
            [CRC,EPS]=self.ClusterTotal(data=DataR[0],dataname=DataR[1],KON=False,Eps=self.eps,MinEps=self.Mineps,adjust=self.adjust,mode=mode)
            CRC.pop(0)
            CRefine.pop(0)
            if self.MidResult == True:
                print('Cluster Number=',len(CRC))  # 最终类数（散点类无论是否为空都计一类）
                print('EPS=',EPS)
                if len(DataR[0][0]) == 2:
                    self.Plot2(CLU=CRC,Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                elif len(DataR[0][0]) == 1:
                    self.Plot1(CLU=CRC, Name=DataR[1],StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
            if len(CRC)<=1:#只有一类
                T=self.ClusterTotal(data=DataR[0],dataname=DataR[1],adjust=0,Eps=EPS,KON=True,mode=mode)[0]
                if self.MidResult == True:
                    print('Cluster Number=',len(T))
                    if len(DataR[0][0]) == 2:
                        self.Plot2(CLU=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                    elif len(DataR[0][0]) == 1:
                        self.Plot1(CLU=T, Name=DataR[1]+' Final',StrEps=' eps='+str(EPS),bbox_to_anchor=(1, 1), loc=2,handletextpad=0.05,labelspacing=0.3)
                if self.IOC==True:#如果散点单独分
                    self.TotalIP.extend(T[0][0])
                    T.pop(0)
                    CAF.extend(T)
                else:
                    CAF.extend(T)
            else:
                CRefine.extend(CRC)
        if self.IOC == True:  # 如果散点单独分
            CAF.insert(0,[np.array(self.TotalIP),'IsoPoint'])

        i=0
        while i<len(CAF):#移除空类并打印每类大小
            if len(CAF[i][0])==0:
                CAF.pop(i)
            else:
                print('Cluster Length=',len(CAF[i][0]))# 输出每类点数
                i=i+1
        print('Final Cluster Number=', len(CAF))  # 最终类数

        if self.RDIC==True and len(self.TotalIP)>0:
            Iso_Cluster=CAF[0]
            CAF.pop(0)
            Final=self.Redistribute_Isolated_Points(Isolated_Clusters=Iso_Cluster,Effective_Clusters=CAF)
            if self.Plot == True:
                if len(self.data[0]) == 2:
                    self.Plot2(CLU=Final, Name=self.DataClustername, StrEps='', bbox_to_anchor=(1, 1.15), loc=2,
                               handletextpad=0.01, labelspacing=0.3)
                elif len(self.data[0]) == 1:
                    self.Plot1(CLU=Final, Name=self.DataClustername, StrEps='', bbox_to_anchor=(1, 1.15), loc=2,
                               handletextpad=0.01, labelspacing=0.3)
            return Final
        else:
            if self.Plot==True:
                if len(self.data[0]) == 2:
                    self.Plot2(CLU=CAF, Name=self.DataClustername, StrEps='', bbox_to_anchor=(1, 1.15), loc=2,
                               handletextpad=0.01, labelspacing=0.3)
                elif len(self.data[0]) == 1:
                    self.Plot1(CLU=CAF, Name=self.DataClustername, StrEps='', bbox_to_anchor=(1, 1.15), loc=2,
                               handletextpad=0.01, labelspacing=0.3)
            return CAF#返回最终的类[[每个类集（数组），名字]]

def Img2Coo(img,channel=3):#Image to coordinary [H,W,3]-->list [(x,y,R,G,B)]
    C=[]
    H=len(img)
    W=len(img[-1])
    m=img.max()
    for i in range(H):
        for j in range(W):
            if channel==1:
                C.append(np.array([i/H,j/W,img[i,j]/m]))
            if channel==3:
                C.append(np.array([i / H, j / W, img[i, j,0] / m, img[i, j,1] / m, img[i, j,2] / m]))
    return np.array(C)

def Show_Cluster(Clu,H,W,save_dir=None,Name=''):#Draw image
    colors = distinctipy.get_colors(len(Clu))
    img=np.ones((H,W,3))
    for i in range(len(Clu)):
        for j in range (len(Clu[i][0])):
            h=round(Clu[i][0][j][0]*H)
            w = round(Clu[i][0][j][1]*W)
            img[h,w]=colors[i]
    if save_dir != None:
        plt.imsave(save_dir + '/'+Name + '.png', img)
    plt.imshow(img)  # 显示图片
    plt.show()
    return img

def Show_Cluster_DBSCAN(core_samples, cluster_ids,H,W,save_dir=None,Name=''):
    colors = distinctipy.get_colors(len(list(set(cluster_ids))))
    img = np.ones((H, W, 3))
    for i in range(len(core_samples)):
        h = core_samples[i]//W
        w = core_samples[i]-h*W
        img[h, w] = colors[cluster_ids[i]]
    if save_dir != None:
        plt.imsave(save_dir + '/'+Name + '.png', img)
    plt.imshow(img)  # 显示图片
    plt.show()
    return img

def Show_3D(Coo):

    fig = plt.figure(1, figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
    ax.scatter(
        Coo[:, 0],
        Coo[:, 1],
        Coo[:, 2],
        c=[0]*len(Coo),
        s=40,
    )

    ax.set_title("3D fig")
    ax.set_xlabel("1st Eigenvector")
    ax.xaxis.set_ticklabels([])
    ax.set_ylabel("2nd Eigenvector")
    ax.yaxis.set_ticklabels([])
    ax.set_zlabel("3rd Eigenvector")
    ax.zaxis.set_ticklabels([])
    plt.show()

def Cluster_add_Pred(Coo_set,Pred_label,relabel=False):# Transform the style to the one accepted by Show_Cluster, relabel means reset the labels to remove the uncontinuous
    #Pred_label 为预测类号，Coo_set为坐标数据
    n=len(list(set(list(Pred_label))))
    if relabel:
        y_label = list(set(Pred_label))
        for i in range(len(Pred_label)):
            for j in range(len(y_label)):
                if Pred_label[i] == y_label[j]:
                    Pred_label[i] = j
                    break
    cl=[]
    for i in range(n):
        cl.append([])
    for i in range(0,len(Coo_set)):
        cl[Pred_label[i]].append(Coo_set[i])
    Clu=[]
    for i in range(0,n):
        Clu.append([cl[i],str(i)])
    return Clu#[[类（坐标list），类名]]

def Plot_SC(CLU,Name='CN',save_dir=None,StrEps='',bbox_to_anchor=(1, 1.15), loc=2,handletextpad=0.05,labelspacing=0.3,SubplotsAdjust=0.67):
    #CLU为类集，CLU[i]为第i类，带名字，CLU[i][0]为第i类集合（坐标），CLU[i][1]为第i类名字，StrEps为Name后要打印的字符串，最后一个是图例位置，用于保存散点图
    plt.figure()  # 作图
    colors = distinctipy.get_colors(len(CLU))
    for i in range(0, len(CLU)):
        x = []
        y = []
        for j in range(0, len(CLU[i][0])):
            x.extend([CLU[i][0][j][0]])  # 这类第j个点的坐标
            y.extend([CLU[i][0][j][1]])
        plt.scatter(x, y, color=colors[i], label=CLU[i][1] + ' N=' + str(len(CLU[i][0])))
    plt.title(Name+StrEps)
    plt.legend(bbox_to_anchor=bbox_to_anchor, loc=loc, ncol=1,handletextpad=handletextpad,labelspacing=labelspacing)
    plt.subplots_adjust(right=SubplotsAdjust)
    if save_dir!=None:
        plt.savefig(save_dir+'/'+'{}.png'.format(Name))#Name
    plt.show()

def Coo2Label(data,coo_set):#将坐标转成标号，data为数据集（np坐标），coo_set为类（np坐标类）
    Label=[]
    for i in range (len(coo_set)):
        for j in range (len(data)):
            if (coo_set[i]==data[j]).all():
                Label.append(j)
                break
            assert j<len(data)-1
    return Label

def CooClu2LabClu(data,Cooclu):#CooClu [[np坐标类，类名]]
    LabClu=[]
    for i in range (len(Cooclu)):
        LabClu.append([Coo2Label(data=data,coo_set=Cooclu[i][0]),Cooclu[i][1]])
    return LabClu#[[类标号列表，类名]]

def Return_Pred_Label(data,Cooclu):#将预测坐标类[[np坐标类，类名]]转成类列表（每个数据的类标），data为数据（np坐标），Cooclu为预测坐标类[[np坐标类，类名]]
    Label_Cluster = CooClu2LabClu(data=data, Cooclu=Cooclu)  # 把坐标变成标号[[类标号列表，类名]]
    Label=[-1]*len(data)
    for i in range (len(Label_Cluster)):
        for j in range (len(Label_Cluster[i][0])):
            Label[Label_Cluster[i][0][j]]=i
    return Label#返回每个数据类标列表

def ARI_NMI(Pred,Label):#Calculate ARI and NMI
    ARI = adjusted_rand_score(Label, Pred)
    NMI = metrics.normalized_mutual_info_score(Pred, Label)
    return ARI,NMI
