# KDD-CUP-2017
KDD CUP 2017 task1

MAPE = 0.1872, 名次 57 / 3754

程序共有三个模型
m1: Pred_big_XGB是直接对整个路段总时间进行预测,使用XGB模型
最后取定参数为:
params={
    'booster':'gbtree',
    'objective':'reg:logistic', #逻辑回归问题
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':8, # 构建树的深度，越大越容易过拟合
    # 'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.9, # 随机采样训练样本
    'colsample_bytree':0.9, # 生成树时进行的列采样
    'min_child_weight':4,
    'eta': 0.01, # 如同学习率
    'seed':1000,
    "silent": 1
    }
迭代次数为218


m2: Pred_cut_XGB是将大路段切分为小路后分别对每个小路段进行预测,然后重组,使用XGB模型
最后参数取定为:
params={
    'booster':'gbtree',
    'objective':'reg:logistic', #逻辑回归问题
    'gamma':0.3,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':8, # 构建树的深度，越大越容易过拟合
    'subsample':0.9, # 随机采样训练样本
    'colsample_bytree':0.9, # 生成树时进行的列采样
    'min_child_weight':2,
    'eta': 0.01, # 如同学习率
    'seed':1000,
    "silent": 1
    }
迭代次数为627


m3: Pred_cut_NN也是将大路段切分后进行预测,使用nn模型
因为这个模型做的比较早,问题比较多,当时没有设seed,所以无法复现单模型最好成绩


最终成绩由四部分构成:
m1 * 0.5 + (m1' * 0.35 + m2 * 0.35 + m3 * 0.3) * 0.5

m1' * 0.35 + m2 * 0.35 + m3 * 0.3的MAPE = 0.1886
其中m1'为m1的eta取0.05,为中间提交的一个结果

最终线上测评为MAPE = 0.1872

最后的模型加权的权重不一定为最佳,没来得及仔细调
