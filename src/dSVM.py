import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from args import *
from sklearn import svm
from requirements import *
from utils import CheckOrder

root_path = "../../raw_data/rois_aal_pkl_pearson/"
# root_path = "/root/autodl-tmp/rois_aal_pkl_pearson/"
label_path = "../description/label_674.csv"

files = os.listdir(root_path)
files.sort()
y = pd.read_csv(label_path)

if not CheckOrder(files, y):
    exit()
y = np.array(y.group_1.values)
data = pd.read_csv("../description/Kendall_sort_all.csv")
index = np.array(data.iloc[:kendall_nums, 1])

X = []
for file in tqdm(files, desc="read_data", file=sys.stdout):
    data = np.array(pd.read_pickle(root_path + file).iloc[:, index].values)
    data = data.reshape(-1)
    X.append(data)

X = np.array(X)

kf = KFold(10, shuffle=True, random_state=seed)
acc_list = []
sen_list = []
spe_list = []
f1_list = []
auc_list = []
k = 0
start_time = time.time()
for train_index, test_index in kf.split(X):
    k += 1
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    predictor = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # 进行训练
    predictor.fit(x_train, y_train)
    y_pred = predictor.predict(x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    AUC = auc(fpr, tpr)
    auc_list.append(AUC)
    f1_list.append(f1)
    acc_list.append(acc)
    sen_list.append(sen)
    spe_list.append(spe)
print("平均准确率：", np.mean(acc_list), "%")
print("平均SEN：", np.mean(sen_list))
print("平均SPE：", np.mean(spe_list))
print("平均F1：", np.mean(f1_list))
print("平均AUC：", np.mean(auc_list))

end_time = time.time()
print("训练总用时：", end_time - start_time)
