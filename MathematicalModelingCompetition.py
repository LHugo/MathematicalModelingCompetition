import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import r2_score

#数据的预处理与特征选择
#数据集导入
data_molecular = pd.read_excel("./Molecular_Descriptor.xlsx")
data_ER = pd.read_excel("./ERα_activity.xlsx")
#对data_molecular和data_ER两个数据集做列合并
data_ER_X = data_ER.drop(["SMILES"], axis=1)
data_concat = pd.concat([data_molecular,data_ER_X], axis=1)
#X和y的划分
X = data_concat.drop(['SMILES','IC50_nM','pIC50'],axis=1)
y = data_concat.loc[:, ['IC50_nM','pIC50']]
#训练测试数据集的划分
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
y_IC50_train = y_train.loc[:,"IC50_nM"]
y_pIC50_train = y_train.loc[:,"pIC50"]
y_IC50_test = y_test.loc[:,"IC50_nM"]
y_pIC50_test = y_test.loc[:,"pIC50"]
#通过随机森林算法来对数据集进行特征提取
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200)
rf_model.fit(X_train,y_pIC50_train)
predict = rf_model.predict(X_train)
print("rt_train_r2score:",r2_score(y_pIC50_train,predict))
predict = rf_model.predict(X_test)
print("rt_test_r2score:",r2_score(y_pIC50_test,predict))
features = X.columns
feature_importances = rf_model.feature_importances_
features_df = pd.DataFrame({'Features':features,'Importance':feature_importances})
features_df.sort_values('Importance',inplace=True,ascending=False)
#对所选择的特征重要性进行可视化
sns.set(rc={"figure.figsize":(21,4)})
sns.barplot(features_df['Features'][:20],features_df['Importance'][:20])
plt.ylabel('Word count')
sns.despine(bottom=True)
plt.show()
#只保留X训练和测试数据集重要性最高的20个特征
X_train = X_train.loc[:,features_df[:20]["Features"]]
X_test = X_test.loc[:,features_df[:20]["Features"]]
#对20个特征进行pca主成分分析
pca = PCA(n_components=20)
pca.fit(X_train)
#查看各个特征的方差，并可视化
pca.explained_variance_ratio_
plt.xlabel("Demensions")
plt.ylabel("explained_variance_ratio")
plt.plot([i for i in range(X_train.shape[1])],[np.sum(pca.explained_variance_ratio_[:i+1]) for i in range(X_train.shape[1])])
plt.legend()
#选取贡献率99%方差的特征
pca = PCA(0.99)
pca.fit(X_train)
pca.n_components_
#通过pca对训练以及测试数据集进行降维处理，维度为8
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#数值均值方差归一化处理
std = StandardScaler()
std.fit(X_train)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)
#①建立SVM模型并对模型的R2值进行评估
from sklearn.svm import SVR
svr = SVR(tol=1e-5, kernel='rbf',C=1e1)
svr.fit(X_train_std,y_pIC50_train)
print("SVR_train_data的r2值为：",svr.score(X_train_std, y_pIC50_train))
print("SVR_test_data的r2值为：",svr.score(X_test_std, y_pIC50_test))
SVR_train_data_r2score = svr.score(X_train_std, y_pIC50_train)
SVR_test_data_r2score = svr.score(X_test_std, y_pIC50_test)
#②建立KNN模型并对模型的R2值进行评估
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
param_grid = [
    {
        "weights": ["uniform"],
        "n_neighbors": [i for i in range(1, 11)]
    },
    {
        "weights": ["distance"],
        "n_neighbors": [i for i in range(1, 11)],
        "p": [i for i in range(1,6)]
    }
]
knn = KNeighborsRegressor()
grid_search = GridSearchCV(knn, param_grid, n_jobs=-1, verbose=1)
grid_search.fit(X_train_std, y_pIC50_train)
print("最优超参数为：",grid_search.best_params_)
print("KNN_train_data的r2值为：",grid_search.best_estimator_.score(X_train_std, y_pIC50_train))
print("KNN_test_data的r2值为：",grid_search.best_estimator_.score(X_test_std, y_pIC50_test))
KNN_train_data_r2score = grid_search.best_estimator_.score(X_train_std, y_pIC50_train)
KNN_test_data_r2score = grid_search.best_estimator_.score(X_test_std, y_pIC50_test)

#③建立随机森林模型并对模型的R2值进行评估
param_grid_rf = [
    {
        "max_depth": [30,40,50],
        'min_samples_leaf': [1,2,3],
        'n_estimators': [300,400,500],
    }
]
rf_model = RandomForestRegressor()
rf_grid = GridSearchCV(rf_model, param_grid_rf, n_jobs=-1, verbose=1)
rf_grid.fit(X_train,y_pIC50_train)
print("最优超参数为：",rf_grid.best_params_)
print("RandomForest_train_r2score:",rf_grid.best_estimator_.score(X_train,y_pIC50_train))
print("RandomForest_test_r2score:",rf_grid.best_estimator_.score(X_test,y_pIC50_test))
RandomForest_train_r2score = rf_grid.best_estimator_.score(X_train,y_pIC50_train)
RandomForest_test_r2score = rf_grid.best_estimator_.score(X_test,y_pIC50_test)
#导入要预测的数据集
origin_molecular = pd.read_excel("./Molecular_Descriptor.xlsx",sheet_name='test')
origin_ER = pd.read_excel("./ERα_activity.xlsx",sheet_name='test')
#测试集的特征选择
X_final = origin_molecular.drop(['SMILES'],axis=1)
X_final = X_final.loc[:,features_df[:20]["Features"]]
#调用训练好的随机森林模型对数据集进行预测
y_final_pIC50 = rf_grid.predict(X_final)
origin_ER.loc[:,["pIC50"]] = y_final_pIC50
y_final_IC50 = np.power(10,-y_final_pIC50)/(10**-9)
origin_ER.loc[:,["IC50_nM"]] = y_final_IC50
origin_ER.to_excel("ERα_activity_predict.xlsx",index=False)
#构建化合物的Caco-2、CYP3A4、hERG、HOB、MN的分类预测模型
#数据导入与预处理
#X和y的划分
data_ADMET = pd.read_excel("./ADMET.xlsx")
X = pd.read_excel("./Molecular_Descriptor.xlsx")
X = X.drop(["SMILES"], axis=1)
y = data_ADMET.drop(["SMILES"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
#对数据集做均值方差归一化处理
std = StandardScaler()
std.fit(X_train)
X_train_std = std.transform(X_train)
X_test_std = std.transform(X_test)
#五个指标训练集和测试集的划分
y_Caco_2_train = y_train.loc[:,"Caco-2"]
y_CYP3A4_train = y_train.loc[:,"CYP3A4"]
y_hERG_train = y_train.loc[:,"hERG"]
y_HOB_train = y_train.loc[:,"HOB"]
y_MN_train = y_train.loc[:,"MN"]

y_Caco_2_test = y_test.loc[:,"Caco-2"]
y_CYP3A4_test = y_test.loc[:,"CYP3A4"]
y_hERG_test = y_test.loc[:,"hERG"]
y_HOB_test = y_test.loc[:,"HOB"]
y_MN_test = y_test.loc[:,"MN"]
#建立神经网络模型训练数据
from keras.models import Sequential
from keras.layers import Dense,Activation
mlp_1 = Sequential()
mlp_1.add(Dense(units=365,activation="relu", input_dim=729))
mlp_1.add(Dense(units=365,activation="softmax"))
mlp_1.add(Dense(units=1,activation="sigmoid"))
mlp_1.summary()
mlp_1.compile(optimizer='adam',loss='binary_crossentropy')
#对Caco-2进行训练并预测
mlp_1.fit(X_train_std,y_Caco_2_train,epochs=400)
from sklearn.metrics import accuracy_score
y_Caco_2_test_predict = mlp_1.predict(X_test_std)
for i in range(y_Caco_2_test_predict.shape[0]):
    if y_Caco_2_test_predict[i,:]>0.5:
        y_Caco_2_test_predict[i,:]=1
    else:
        y_Caco_2_test_predict[i,:]=0
y_Caco_2_test_predict = np.array(y_Caco_2_test_predict,dtype=int)
Caco_2_accuracy = accuracy_score(y_Caco_2_test,y_Caco_2_test_predict)
print("Caco_2_accuracy:",Caco_2_accuracy)
#对CYP3A4进行训练并预测
mlp_2 = Sequential()
mlp_2.add(Dense(units=365,activation="relu", input_dim=729))
mlp_2.add(Dense(units=365,activation="softmax"))
mlp_2.add(Dense(units=1,activation="sigmoid"))
mlp_2.compile(optimizer='adam',loss='binary_crossentropy')
mlp_2.fit(X_train_std,y_CYP3A4_train,epochs=400)
y_CYP3A4_test_predict = mlp_2.predict(X_test_std)
for i in range(y_CYP3A4_test_predict.shape[0]):
    if y_CYP3A4_test_predict[i,:]>0.5:
        y_CYP3A4_test_predict[i,:]=1
    else:
        y_CYP3A4_test_predict[i,:]=0
y_CYP3A4_test_predict = np.array(y_CYP3A4_test_predict,dtype=int)
CYP3A4_accuracy = accuracy_score(y_CYP3A4_test,y_CYP3A4_test_predict)
print("CYP3A4_accuracy:",CYP3A4_accuracy)
#对hERG进行训练并预测
mlp_3 = Sequential()
mlp_3.add(Dense(units=365,activation="relu", input_dim=729))
mlp_3.add(Dense(units=365,activation="softmax"))
mlp_3.add(Dense(units=1,activation="sigmoid"))
mlp_3.compile(optimizer='adam',loss='binary_crossentropy')
mlp_3.fit(X_train_std,y_hERG_train,epochs=400)
y_hERG_test_predict = mlp_3.predict(X_test_std)
for i in range(y_hERG_test_predict.shape[0]):
    if y_hERG_test_predict[i,:]>0.5:
        y_hERG_test_predict[i,:]=1
    else:
        y_hERG_test_predict[i,:]=0
y_hERG_test_predict = np.array(y_hERG_test_predict,dtype=int)
hERG_accuracy = accuracy_score(y_hERG_test,y_hERG_test_predict)
print("hERG_accuracy:",hERG_accuracy)
#对HOB进行训练并预测
mlp_4 = Sequential()
mlp_4.add(Dense(units=365,activation="relu", input_dim=729))
mlp_4.add(Dense(units=365,activation="softmax"))
mlp_4.add(Dense(units=1,activation="sigmoid"))
mlp_4.compile(optimizer='adam',loss='binary_crossentropy')
mlp_4.fit(X_train_std,y_HOB_train,epochs=400)
y_HOB_test_predict = mlp_4.predict(X_test_std)
for i in range(y_HOB_test_predict.shape[0]):
    if y_HOB_test_predict[i,:]>0.5:
        y_HOB_test_predict[i,:]=1
    else:
        y_HOB_test_predict[i,:]=0
y_HOB_test_predict = np.array(y_HOB_test_predict,dtype=int)
HOB_accuracy = accuracy_score(y_HOB_test,y_HOB_test_predict)
print("HOB_accuracy:",HOB_accuracy)
#对MN进行训练并预测
mlp_5 = Sequential()
mlp_5.add(Dense(units=365,activation="relu", input_dim=729))
mlp_5.add(Dense(units=365,activation="softmax"))
mlp_5.add(Dense(units=1,activation="sigmoid"))
mlp_5.compile(optimizer='adam',loss='binary_crossentropy')
mlp_5.fit(X_train_std,y_MN_train,epochs=400)
y_MN_test_predict = mlp_5.predict(X_test_std)
for i in range(y_MN_test_predict.shape[0]):
    if y_MN_test_predict[i,:]>0.5:
        y_MN_test_predict[i,:]=1
    else:
        y_MN_test_predict[i,:]=0
y_MN_test_predict = np.array(y_MN_test_predict,dtype=int)
MN_accuracy = accuracy_score(y_MN_test,y_MN_test_predict)
print("MN_accuracy:",MN_accuracy)
#对excel的五项指标值进行预测并填充
#预测数据集导入
data_ADMET_test = pd.read_excel("./ADMET.xlsx",sheet_name="test")
origin_molecular = pd.read_excel("./Molecular_Descriptor.xlsx",sheet_name='test')
X_final = origin_molecular.drop(['SMILES'],axis=1)
#对预测值做均值方差归一化
std = StandardScaler()
std.fit(X_final)
X_final_std = std.transform(X_final)
#预测Caco_2指标并填入excel表
Caco_2_predict = mlp_1.predict(X_final_std)
for i in range(Caco_2_predict.shape[0]):
    if Caco_2_predict[i,:]>0.5:
        Caco_2_predict[i,:]=1
    else:
        Caco_2_predict[i,:]=0
Caco_2_predict = np.array(Caco_2_predict,dtype=int)
data_ADMET_test.loc[:,["Caco-2"]] = Caco_2_predict
#预测CYP3A4指标并填入excel表
CYP3A4_predict = mlp_2.predict(X_final_std)
for i in range(CYP3A4_predict.shape[0]):
    if CYP3A4_predict[i,:]>0.5:
        CYP3A4_predict[i,:]=1
    else:
        CYP3A4_predict[i,:]=0
CYP3A4_predict = np.array(CYP3A4_predict,dtype=int)
data_ADMET_test.loc[:,["CYP3A4"]] = CYP3A4_predict
#预测hERG指标并填入excel表
hERG_predict = mlp_3.predict(X_final_std)
for i in range(hERG_predict.shape[0]):
    if hERG_predict[i,:]>0.5:
        hERG_predict[i,:]=1
    else:
        hERG_predict[i,:]=0
hERG_predict = np.array(hERG_predict,dtype=int)
data_ADMET_test.loc[:,["hERG"]] = hERG_predict
#预测HOB指标并填入excel表
HOB_predict = mlp_4.predict(X_final_std)
for i in range(HOB_predict.shape[0]):
    if HOB_predict[i,:]>0.5:
        HOB_predict[i,:]=1
    else:
        HOB_predict[i,:]=0
HOB_predict = np.array(HOB_predict,dtype=int)
data_ADMET_test.loc[:,["HOB"]] = HOB_predict
#预测MN指标并填入excel表
MN_predict = mlp_5.predict(X_final_std)
for i in range(MN_predict.shape[0]):
    if MN_predict[i,:]>0.5:
        MN_predict[i,:]=1
    else:
        MN_predict[i,:]=0
MN_predict = np.array(MN_predict,dtype=int)
data_ADMET_test.loc[:,["MN"]] = MN_predict
data_ADMET_test.to_excel("ADMET_predict.xlsx",index=False)
# 寻找使化合物对抑制ERα具有更好的生物活性，同时具有更好的ADMET性质的分子描述符的取值范围
#筛选出ADMET中五项指标其中有三项指标值为1的样本
ADMET_data = pd.read_excel("ADMET.xlsx").drop(["SMILES"],axis=1)
ER_data = pd.read_excel("ERα_activity.xlsx").drop(["SMILES","IC50_nM"],axis=1)
ADMET_data.loc[:,["hERG","MN"]] = ADMET_data.loc[:,["hERG","MN"]].replace({0:1, 1:0})
comprehensive_value = []
for i in range(ADMET_data.shape[0]):
    if ADMET_data.loc[:,"Caco-2"][i]+ADMET_data.loc[:,"CYP3A4"][i]+ADMET_data.loc[:,"hERG"][i]+ADMET_data.loc[:,"HOB"][i]+ADMET_data.loc[:,"MN"][i]>2:
        comprehensive_value.append(1)
    else:
        comprehensive_value.append(0)
ADMET_data["comprehensive_value"]=comprehensive_value
ADMET_data = ADMET_data.loc[:,"comprehensive_value"]
Molecular_data = pd.read_excel("Molecular_Descriptor.xlsx").loc[:,features_df[:20]["Features"]]
Data_Comprehensive = pd.concat([ADMET_data,Molecular_data,ER_data],axis=1)
from collections import Counter
Counter(Data_Comprehensive.loc[:,"comprehensive_value"])
Data_Comprehensive = Data_Comprehensive[Data_Comprehensive["comprehensive_value"]==1]
Data_Comprehensive = Data_Comprehensive.drop(["comprehensive_value"],axis=1)
X = Data_Comprehensive.drop(["pIC50"],axis=1)
y = Data_Comprehensive.loc[:,"pIC50"]
#用线性回归来求解各个特征的系数，寻找对抑制生物活性具有正性影响的分子描述符
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X,y)
print(X.columns[np.argsort(lr.coef_)])
y_columns = X.columns[np.argsort(lr.coef_)][np.sort(lr.coef_)>0]
print(y_columns)
X_select = X.loc[:,y_columns]
data = pd.concat([X_select,y],axis=1)
#对样本数据按照pIC50进行降序排序并取出前5%的样本数据
data = data.sort_values(by='pIC50',ascending=False)
data = data[:int(data.shape[0]*0.05)]
#把样本中五个指标的最小最大值作为取值区间
print("MDEC-23_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"MDEC-23"]),np.max(data.loc[:,"MDEC-23"])))
print("SHBint10_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"SHBint10"]),np.max(data.loc[:,"SHBint10"])))
print("LipoaffinityIndex_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"LipoaffinityIndex"]),np.max(data.loc[:,"LipoaffinityIndex"])))
print("minHBint5_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"minHBint5"]),np.max(data.loc[:,"minHBint5"])))
print("minsssN_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"minsssN"]),np.max(data.loc[:,"minsssN"])))
print("nC_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"nC"]),np.max(data.loc[:,"nC"])))
print("minHsOH_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"minHsOH"]),np.max(data.loc[:,"minHsOH"])))
print("VC-5_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"VC-5"]),np.max(data.loc[:,"VC-5"])))
print("MLFER_A_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"MLFER_A"]),np.max(data.loc[:,"MLFER_A"])))
print("maxHsOH_range:{:.3f}~{:.3f}".format(np.min(data.loc[:,"maxHsOH"]),np.max(data.loc[:,"maxHsOH"])))