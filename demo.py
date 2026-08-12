import os, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, accuracy_score
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
X, y = make_classification(n_samples=600, n_features=20, n_informative=6, n_redundant=2,
                           class_sep=0.9, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0).fit(Xtr, ytr)
rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xtr, ytr)
lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
fig, ax = plt.subplots(2, 2, figsize=(12, 9))
imp = gb.feature_importances_; top = np.argsort(imp)[::-1][:15]
ax[0,0].bar(range(len(top)), imp[top], color="#C44E52"); ax[0,0].set_xticks(range(len(top)))
ax[0,0].set_xticklabels([f"f{i}" for i in top], rotation=90, fontsize=7); ax[0,0].set_title("Gradient boosting feature importance")
for m, name, c in [(gb,"Gradient boosting","#4C72B0"),(rf,"Random forest","#55A868"),(lr,"Logistic","#DD8452")]:
    fpr,tpr,_=roc_curve(yte, m.predict_proba(Xte)[:,1]); ax[0,1].plot(fpr,tpr,color=c,label=f"{name} (AUC={auc(fpr,tpr):.3f})")
ax[0,1].plot([0,1],[0,1],"k--"); ax[0,1].set_title("ROC on held-out set"); ax[0,1].legend(fontsize=8,loc="lower right"); ax[0,1].set_xlabel("FPR"); ax[0,1].set_ylabel("TPR")
tr_err=[1-accuracy_score(ytr,p) for p in gb.staged_predict(Xtr)]
te_err=[1-accuracy_score(yte,p) for p in gb.staged_predict(Xte)]
ax[1,0].plot(tr_err,label="train error"); ax[1,0].plot(te_err,label="test error"); ax[1,0].set_xlabel("boosting rounds"); ax[1,0].set_ylabel("error"); ax[1,0].set_title("Boosting: train vs test error"); ax[1,0].legend(fontsize=8)
accs={"Logistic":accuracy_score(yte,lr.predict(Xte)),"Random forest":accuracy_score(yte,rf.predict(Xte)),"Gradient boosting":accuracy_score(yte,gb.predict(Xte))}
ax[1,1].bar(list(accs),list(accs.values()),color=["#DD8452","#55A868","#4C72B0"]); ax[1,1].set_ylim(0.5,1); ax[1,1].set_title("Test accuracy")
for i,v in enumerate(accs.values()): ax[1,1].text(i,v+0.01,f"{v:.2f}",ha="center")
fig.suptitle("Gradient boosting classifier (demo data)", fontsize=14); fig.tight_layout(rect=[0,0,1,0.97]); fig.savefig("figures/demo.png", dpi=120)
open("results/summary.csv","w").write("model,test_accuracy\n"+"\n".join(f"{k},{v:.3f}" for k,v in accs.items())+"\n")
print(accs); print("ok")
