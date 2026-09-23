from pathlib import Path
import json, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

from preprocessing import load_data, clean_data, add_engineered_features, save_eda_summary

BASE = Path(__file__).resolve().parents[1]
MODELS, OUTPUTS = BASE/"models", BASE/"outputs"
MODELS.mkdir(exist_ok=True); OUTPUTS.mkdir(exist_ok=True)
RANDOM_STATE = 42

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    num = Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())])
    cat = Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),
                    ("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num",num,numeric),("cat",cat,categorical)])

def make_pipeline(model, preprocessor):
    return ImbPipeline([("preprocess",preprocessor),("smote",SMOTE(random_state=RANDOM_STATE)),("model",model)])

def specificity(y_true,y_pred):
    tn,fp,fn,tp = confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    return tn/(tn+fp) if tn+fp else 0.0

def evaluate(name,estimator,X,y):
    pred=estimator.predict(X); prob=estimator.predict_proba(X)[:,1]
    return {
        "Model":name,"Accuracy":accuracy_score(y,pred),
        "Precision":precision_score(y,pred,zero_division=0),
        "Recall":recall_score(y,pred,zero_division=0),
        "F1":f1_score(y,pred,zero_division=0),
        "ROC_AUC":roc_auc_score(y,prob),
        "Sensitivity":recall_score(y,pred,zero_division=0),
        "Specificity":specificity(y,pred)
    },pred,prob

def main():
    df=load_data()
    df,dups=clean_data(df)
    df=add_engineered_features(df)
    save_eda_summary(df,dups)

    X=df.drop(columns=["Outcome"]); y=df["Outcome"].astype(int)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,stratify=y,random_state=RANDOM_STATE)
    # Keep a small representative background set for SHAP explanations.
    Xtr.sample(min(100,len(Xtr)), random_state=RANDOM_STATE).to_csv(OUTPUTS/"shap_background.csv", index=False)
    prep=build_preprocessor(Xtr)

    candidates={
      "Logistic Regression":(LogisticRegression(max_iter=3000),{"model__C":[.1,1,10]}),
      "Decision Tree":(DecisionTreeClassifier(random_state=RANDOM_STATE),{"model__max_depth":[3,5,8,None],"model__min_samples_leaf":[1,3,5]}),
      "Random Forest":(RandomForestClassifier(n_estimators=250,random_state=RANDOM_STATE,n_jobs=-1),{"model__max_depth":[None,5,10],"model__min_samples_leaf":[1,3,5]}),
      "Gradient Boosting":(GradientBoostingClassifier(random_state=RANDOM_STATE),{"model__n_estimators":[100,200],"model__learning_rate":[.03,.1],"model__max_depth":[2,3]}),
      "SVM":(SVC(probability=True,random_state=RANDOM_STATE),{"model__C":[.5,1,2],"model__kernel":["rbf","linear"]})
    }
    if XGB_AVAILABLE:
        candidates["XGBoost"]=(XGBClassifier(n_estimators=200,max_depth=3,learning_rate=.05,subsample=.9,colsample_bytree=.9,eval_metric="logloss",random_state=RANDOM_STATE),
                               {"model__max_depth":[2,3,4],"model__learning_rate":[.03,.05,.1]})

    rows=[]; fitted={}
    for name,(model,grid) in candidates.items():
        search=GridSearchCV(make_pipeline(model,prep),grid,cv=5,scoring="roc_auc",n_jobs=-1)
        search.fit(Xtr,ytr)
        m,pred,prob=evaluate(name,search.best_estimator_,Xte,yte)
        m["Best_Params"]=str(search.best_params_)
        rows.append(m); fitted[name]=search.best_estimator_

    comparison=pd.DataFrame(rows).sort_values("ROC_AUC",ascending=False)
    comparison.to_csv(OUTPUTS/"model_comparison.csv",index=False)

    best_name=str(comparison.iloc[0]["Model"]); best=fitted[best_name]
    metrics,pred,prob=evaluate(best_name,best,Xte,yte)
    pd.DataFrame({"Actual":yte.to_numpy(),"Predicted":pred,"Probability":prob}).to_csv(OUTPUTS/"test_predictions.csv",index=False)

    fpr,tpr,_=roc_curve(yte,prob)
    pd.DataFrame({"FPR":fpr,"TPR":tpr}).to_csv(OUTPUTS/"roc_curve.csv",index=False)
    precision,recall,_=precision_recall_curve(yte,prob)
    pd.DataFrame({"Recall":recall,"Precision":precision}).to_csv(OUTPUTS/"pr_curve.csv",index=False)

    joblib.dump(best,MODELS/"disease_pipeline.pkl")
    meta={"best_model":best_name,"feature_columns":X.columns.tolist(),"target":"Outcome",
          "test_metrics":metrics,
          "risk_bands":{"LOW":[0,.33],"MEDIUM":[.33,.66],"HIGH":[.66,1]},
          "note":"Risk bands are application-level probability bands, not clinical diagnostic thresholds."}
    (MODELS/"model_metadata.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")

    try:
        model=best.named_steps["model"]; names=best.named_steps["preprocess"].get_feature_names_out()
        values=model.feature_importances_ if hasattr(model,"feature_importances_") else np.abs(model.coef_[0]) if hasattr(model,"coef_") else None
        if values is not None:
            pd.DataFrame({"Feature":names,"Importance":values}).sort_values("Importance",ascending=False).to_csv(OUTPUTS/"feature_importance.csv",index=False)
    except Exception as exc:
        print("Feature importance export skipped:",exc)

    print(comparison[["Model","Accuracy","Precision","Recall","F1","ROC_AUC","Sensitivity","Specificity"]].to_string(index=False))
    print("Best pipeline:",MODELS/"disease_pipeline.pkl")

if __name__=="__main__":
    main()
