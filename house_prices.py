import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm, skew
import seaborn as sns
import numpy as np
from scipy.special import boxcox1p
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
# 自定義分類器，繼承BaseEstimator, RegressorMixin, TransformerMixin三個物件
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X,y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

def rmse_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train)
    rmse = np.sqrt(-cross_val_score(model, train, y_train, scoring='neg_mean_squared_error', cv=kf))
    return rmse

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

train = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\house-prices\train.csv')
test = pd.read_csv(r'C:\Users\Blake\PycharmProjects\Image_Recognition\data\house-prices\test.csv')
train_id = train['Id']
test_id = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
train = train.drop(train[(train['GrLivArea']>4000)&(train['SalePrice']<300000)].index)
train['SalePrice'] = np.log1p(train['SalePrice'])

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio':all_data_na})

all_data['PoolQC'] = all_data['PoolQC'].fillna('None')
all_data['MiscFeature'] = all_data['MiscFeature'].fillna('None')
all_data['Alley'] = all_data['Alley'].fillna('None')
all_data['Fence'] = all_data['Fence'].fillna('None')
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
for col in ('GarageType','GarageFinish','GarageQual','GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt','GarageArea','GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode())[0]
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode())[0]
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode())[0]
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode())[0]
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode())[0]
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode())[0]
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x:x.fillna(x.median()))
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data = all_data.drop(['Utilities'], axis=1)
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data['FireplaceQu'] = all_data['FireplaceQu'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['GarageQual'] = all_data['GarageQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['GarageCond'] = all_data['GarageCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['GarageFinish'] = all_data['GarageFinish'].map({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})
all_data['BsmtQual'] = all_data['BsmtQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['BsmtCond'] = all_data['BsmtCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['BsmtExposure'] = all_data['BsmtExposure'].map({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})
all_data['ExterQual'] = all_data['ExterQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['ExterCond'] = all_data['ExterCond'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['HeatingQC'] = all_data['HeatingQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['PoolQC'] = all_data['PoolQC'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['KitchenQual'] = all_data['KitchenQual'].map({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
all_data['Functional'] = all_data['Functional'].map({'Typ':8, 'Min1':7, 'Min2':6, 'Mod':5, 'Maj1':4, 'Maj2':3, 'Sev':2, 'Sal':1, 'NA':0})
all_data['Fence'] = all_data['Fence'].map({'GdPrv':4, 'MnPrv':3, 'GdWo':2, 'MnWw':1, 'NA':0})
all_data['LandSlope'] = all_data['LandSlope'].map({'Gtl':3, 'Mod':2, 'Sev':1, 'NA':0})
all_data['LotShape'] = all_data['LotShape'].map({'Reg':4, 'IR1':3, 'IR2':2, 'IR3':1, 'NA':0})
all_data['PavedDrive'] = all_data['PavedDrive'].map({'Y':3, 'P':2, 'N':1, 'NA':0})
all_data['Street'] = all_data['Street'].map({'Pave':2, 'Grvl':1, 'NA':0})
all_data['Alley'] = all_data['Alley'].map({'Pave':2, 'Grvl':1, 'NA':0})
all_data['CentralAir'] = all_data['CentralAir'].map({'Y':1, 'N':0})
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['OverallQual_TotalSF'] = all_data['OverallQual'] * all_data['TotalSF']
all_data['OverallQual_GrLivArea'] = all_data['OverallQual'] * all_data['GrLivArea']
all_data['OverallQual_TotRmsAbvGrd'] = all_data['OverallQual'] * all_data['TotRmsAbvGrd']
all_data['GarageArea_YearBuilt'] = all_data['GarageArea'] + all_data['YearBuilt']

numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x:skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew':skewed_feats})
skewness = skewness[abs(skewness['Skew']) > 0.75]
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat],lam)

all_data = pd.get_dummies(all_data)
train = all_data[:ntrain]
test = all_data[ntrain:]

scaler = RobustScaler()
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)

#Validation function
n_folds = 5
lasso = Lasso(alpha=0.0005, random_state=1) #Lasso回歸，計算稀疏係數[值大多為0]
ENet = ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3) #彈性網路，Lasso[L1,使用W權重絕對值來正規化]及Ridge[L2,使用W權重平方來正規化]的結合
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5) #彈性網路進階版，使用內核速度更快
GBoost = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4,
                                   max_features='sqrt', min_samples_leaf=15,
                                   min_samples_split=10, loss='huber', random_state=5)  #梯度提升樹
model_xgb = xgb.XGBRegressor(colsample_bytree=0.5, gamma=0.05, learning_rate=0.05,
                             max_depth=3, min_child_weight=1.8, n_estimators=1000,
                             reg_alpha=0.5, reg_lambda=0.8, subsample=0.5, silent=1,
                             random_state=7, nthread=-1)    #梯度提升樹優化版
model_lgb = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                              n_estimators=1000, max_bin=55, bagging_fraction=0.8,
                              bagging_freq=5, feature_fraction=0.2, feature_fraction_seed=9,
                              bagging_seed=9, min_data_in_leaf=6, min_sum_hessian_in_leaf=11)   #垂直梯度提升樹, 速度快並支援GPU
stacked_averageed_models = StackingAveragedModels(base_models=(ENet, GBoost, KRR), meta_model=lasso)
train = pd.DataFrame(train)
train.dropna(axis=1, inplace=True)
train = train.values
test = pd.DataFrame(test)
test.dropna(axis=1, inplace=True)
test = test.values
stacked_averageed_models.fit(train, y_train)
stacked_train_pred = stacked_averageed_models.predict(train)
stacked_pred = np.expm1(stacked_averageed_models.predict(test)) #test也進行轉換因應train log1p
print(rmsle(y_train, stacked_train_pred))
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))
model_lgb.fit(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test))
print(rmsle(y_train, lgb_train_pred))
ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15

sub = pd.DataFrame()
sub['Id'] = test_id
sub['SalePrice'] = ensemble
sub.to_csv('submission_1.csv', index=False)
# score = rmse_cv(lasso)
# print('Lasso score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmse_cv(lasso)
# print('ElasticNet score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmse_cv(lasso)
# print('Kernel Ridge score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmse_cv(lasso)
# print('Gradient Boosting score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmse_cv(lasso)
# print('Xgboost score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# score = rmse_cv(lasso)
# print('LGBM score: {:.4f} ({:.4f})\n'.format(score.mean(), score.std()))
# sns.distplot(train['SalePrice'], fit=norm)
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot=plt)
# plt.show()
# fig, ax = plt.subplots()
# ax.scatter(train['GrLivArea'], train['SalePrice'])
# plt.ylabel('SalePrice', fontsize=13)
# plt.xlabel('GrLivArea', fontsize=13)
# plt.show()