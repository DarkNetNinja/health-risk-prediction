from pathlib import Path
import json
import pandas as pd
import plotly.express as px
import streamlit as st
import requests

BASE=Path(__file__).resolve().parents[1]
OUTPUTS=BASE/"outputs"; MODELS=BASE/"models"; API="http://127.0.0.1:8000"

st.set_page_config(page_title="Healthcare Risk Prediction",page_icon="🩺",layout="wide")
st.title("🩺 Intelligent Healthcare Disease Risk Prediction")
st.caption("Educational clinical decision-support system — not a medical diagnosis tool.")

if not (MODELS/"model_metadata.json").exists():
    st.warning("Run `python training/train.py` first.")
    st.stop()

meta=json.loads((MODELS/"model_metadata.json").read_text())
page=st.sidebar.radio("Navigate",[
    "Project Dashboard","Individual Risk Prediction","Explainable Prediction",
    "Batch Prediction","Population Analytics","Model Evaluation","About"
])

def patient_form(key):
    with st.form(key):
        a,b,c=st.columns(3)
        age=a.number_input("Age",18,120,45,key=key+"age")
        bmi=b.number_input("BMI",1.0,80.0,29.5,key=key+"bmi")
        glucose=c.number_input("Blood Glucose",1.0,500.0,145.0,key=key+"glucose")
        bp=a.number_input("Blood Pressure",1.0,300.0,88.0,key=key+"bp")
        insulin=b.number_input("Insulin",0.0,3000.0,120.0,key=key+"insulin")
        pregnancies=c.number_input("Pregnancies",0,30,2,key=key+"preg")
        skin=a.number_input("Skin Thickness",0.0,100.0,20.0,key=key+"skin")
        pedigree=b.number_input("Diabetes Pedigree Function",0.0,5.0,.47,key=key+"ped")
        submit=st.form_submit_button("Run")
    return submit,{"age":age,"bmi":bmi,"glucose":glucose,"blood_pressure":bp,"insulin":insulin,
                   "pregnancies":pregnancies,"skin_thickness":skin,"diabetes_pedigree_function":pedigree}

if page=="Project Dashboard":
    st.header("Project Dashboard")
    p=OUTPUTS/"test_predictions.csv"; total=positive=high=0; avg=None
    if p.exists():
        d=pd.read_csv(p); total=len(d); positive=int(d.Actual.sum()); avg=float(d.Probability.mean()); high=int((d.Probability>=.66).sum())
    a,b,c,d=st.columns(4)
    a.metric("Evaluation Records",total); b.metric("Positive Cases",positive)
    c.metric("Average Predicted Risk",f"{avg:.1%}" if avg is not None else "—"); d.metric("High-Risk Predictions",high)
    st.subheader("Model Used"); st.info(meta["best_model"])
    if (OUTPUTS/"model_comparison.csv").exists(): st.dataframe(pd.read_csv(OUTPUTS/"model_comparison.csv"),use_container_width=True)

elif page=="Individual Risk Prediction":
    st.header("Individual Risk Prediction")
    submit,payload=patient_form("individual")
    if submit:
        try:
            r=requests.post(f"{API}/predict",json=payload,timeout=10); r.raise_for_status(); res=r.json()
            x,y,z=st.columns(3); x.metric("Prediction",res["prediction"]); y.metric("Probability",f"{res['probability']:.1%}"); z.metric("Risk Category",res["risk_level"])
        except Exception as e: st.error(f"Start FastAPI first. Details: {e}")

elif page=="Explainable Prediction":
    st.header("Explainable Prediction")
    submit,payload=patient_form("explain")
    if submit:
        try:
            r=requests.post(f"{API}/explain",json=payload,timeout=30); r.raise_for_status(); res=r.json()
            st.info(f"Explanation method: {res['explanation_method']}")
            x,y,z=st.columns(3); x.metric("Prediction",res["prediction"]); y.metric("Probability",f"{res['probability']:.1%}"); z.metric("Risk Category",res["risk_level"])
            inc=res.get("factors_increasing_risk",[]); dec=res.get("factors_reducing_risk",[])
            c1,c2=st.columns(2)
            with c1:
                st.subheader("Factors increasing predicted risk")
                st.dataframe(pd.DataFrame(inc),use_container_width=True)
            with c2:
                st.subheader("Factors reducing predicted risk")
                st.dataframe(pd.DataFrame(dec),use_container_width=True)
            if res.get("explanation_note"): st.caption(res["explanation_note"])
        except Exception as e: st.error(f"Explanation failed. Details: {e}")

elif page=="Batch Prediction":
    st.header("Batch Prediction")
    st.write("Upload a CSV containing patient records. The target `Outcome` column is not required.")
    uploaded=st.file_uploader("patients.csv",type=["csv"])
    if uploaded:
        df=pd.read_csv(uploaded)
        st.dataframe(df.head(),use_container_width=True)
        required=["patient_id","Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
        missing=[c for c in required if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}")
        elif st.button("Generate Batch Predictions"):
            patients=[]
            for _,r in df.iterrows():
                patients.append({"patient_id":str(r["patient_id"]),"pregnancies":int(r["Pregnancies"]),
                    "glucose":float(r["Glucose"]),"blood_pressure":float(r["BloodPressure"]),
                    "skin_thickness":float(r["SkinThickness"]),"insulin":float(r["Insulin"]),
                    "bmi":float(r["BMI"]),"diabetes_pedigree_function":float(r["DiabetesPedigreeFunction"]),
                    "age":int(r["Age"])})
            try:
                resp=requests.post(f"{API}/batch-predict",json={"patients":patients},timeout=30)
                resp.raise_for_status()
                out=pd.DataFrame(resp.json()["results"]); st.dataframe(out,use_container_width=True)
                st.download_button("Download predictions.csv",out.to_csv(index=False),"predictions.csv","text/csv")
            except Exception as e: st.error(f"Batch API failed. Details: {e}")

elif page=="Population Analytics":
    st.header("Population Analytics")
    p=OUTPUTS/"test_predictions.csv"
    if not p.exists(): st.warning("Train the model first."); st.stop()
    d=pd.read_csv(p)
    x,y=st.columns(2)
    x.plotly_chart(px.histogram(d,x="Probability",nbins=20,title="Predicted Risk Distribution"),use_container_width=True)
    y.plotly_chart(px.histogram(d,x="Actual",title="Actual Disease Distribution"),use_container_width=True)

elif page=="Model Evaluation":
    st.header("Model Evaluation")
    p=OUTPUTS/"model_comparison.csv"
    if p.exists():
        comp=pd.read_csv(p); st.dataframe(comp,use_container_width=True)
        st.bar_chart(comp.set_index("Model")[["Precision","Recall","F1","ROC_AUC"]])
    for name,xcol,ycol,title in [("roc_curve.csv","FPR","TPR","ROC Curve"),("pr_curve.csv","Recall","Precision","Precision-Recall Curve")]:
        p=OUTPUTS/name
        if p.exists(): st.plotly_chart(px.line(pd.read_csv(p),x=xcol,y=ycol,title=title),use_container_width=True)
    p=OUTPUTS/"feature_importance.csv"
    if p.exists():
        f=pd.read_csv(p).head(15).sort_values("Importance")
        st.plotly_chart(px.bar(f,x="Importance",y="Feature",orientation="h",title="Top Feature Importance"),use_container_width=True)

elif page=="About":
    st.header("About")
    st.markdown("Dataset → Cleaning → EDA → Missing values → Outlier analysis → Feature engineering → Encoding → Scaling → Train/Test split → SMOTE → Model training → Hyperparameter tuning → Model comparison → Best pipeline → FastAPI → Streamlit.")
    st.warning("This application is for educational demonstration. Model probability is not a medical diagnosis.")
