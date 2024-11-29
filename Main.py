import pandas as pd
from sklearn import preprocessing
from SDC_HSDD_NDSA import *
from sklearn.cluster import dbscan,Birch,OPTICS

Pre=preprocessing.MinMaxScaler()

x=pd.read_csv('Toy_Dataset/SDD.csv', index_col=0).values
Label=pd.read_csv('Toy_Dataset/SDD_Label.csv', index_col=0).values
Label=Label.squeeze()
x=Pre.fit_transform(x)#For comparison
name='SDD_'

Plot_SC(Cluster_add_Pred(x,Label),save_dir=None,Name='Dataset')#Plot Groundtruth, set save_dir=None if not need to save, else set save_dir to be the path

######################################
ClusterSample=SDCHSDDNDSA(data=x,Redistribute_Isolated_Clusters=True,Plot=False)#for Toy

# ClusterSample=SDCHSDDNDSA(data=df, MaxIsoPointRho=0,MinClusterPoint=35,MinFractionPoint=0,MidResult=False,Dataname='ID',DataClustername='Name',SearchNeiborK=9,RhoCalculateK=7,
#                   eps=0.18,IsoNeiborK=4,adjust=0,Mineps=0,Maxeps=0,MinKNNClusterPoint=13,IOC=True,Redistribute_Isolated_Clusters=False,Plot=False) #for clustering image

Clu=ClusterSample.ClusterRefine()
# Show_Cluster(Clu,H,W,save_dir=None,Name=name+'SDC')#Display images, set save_dir=None if not need to save, else set save_dir to be the path
Plot_SC(Clu,save_dir=None,Name=name+'Ours')#Display the scatter plot 2D, set save_dir=None if not need to save, else set save_dir to be the path
Label_Pred=Return_Pred_Label(data=x,Cooclu=Clu)
ARI,NMI=ARI_NMI(Pred=Label_Pred,Label=Label)
print('SDC-HSDD-NDSA: ARI NMI',ARI,NMI)
########################################################

core_samples, cluster_ids = dbscan(x, eps=0.13, min_samples=4)  # eps=0.05 for Image
# Show_Cluster_DBSCAN(core_samples, cluster_ids,H,W,save_dir=None,Name=name+'DBSCAN')
y=[-1]*len(x)
for i in range(len(core_samples)):
    y[core_samples[i]]=cluster_ids[i]
Clu=Cluster_add_Pred(x,y,relabel=True)
Plot_SC(Clu,save_dir=None,Name=name+'DBSCAN')
ARI,NMI=ARI_NMI(Pred=y,Label=Label)
print('DBSCAN: ARI NMI',ARI,NMI)

###################################################################
y= Birch(threshold=0.1,n_clusters=None).fit_predict(x)
# Show_Cluster(Clu,H,W,save_dir=None,Name=name+'Birch')
Clu=Cluster_add_Pred(x,y) # Transform the type accepted by Plot_SC
Plot_SC(Clu,save_dir=None,Name=name+'Brich')
ARI,NMI=ARI_NMI(Pred=y,Label=Label)
print('Brich: ARI NMI',ARI,NMI)

#####################################################################
y=OPTICS(min_samples=15).fit(x)#for image, min_sample=35
y=y.labels_
# Show_Cluster(Clu,H,W,save_dir=None,Name=name+'OPTICS')
Clu=Cluster_add_Pred(x,y) # Transform the type accepted by Plot_SC
Plot_SC(Clu,save_dir=None,Name=name+'OPTICS')
ARI,NMI=ARI_NMI(Pred=y,Label=Label)
print('OPTICS: ARI NMI',ARI,NMI)
