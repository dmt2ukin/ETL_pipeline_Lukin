from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Добавляем путь к etl/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'etl'))

from load_data import load_dataset
from preprocess_data import preprocess
from train_model import train
from evaluate_model import evaluate
from save_results import save_metrics
import joblib

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1
}

dag = DAG(
    'breast_cancer_etl_pipeline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description='ETL pipeline for breast cancer prediction'
)

# Глобальные переменные
DATA_PATH = 'data/breast_cancer_dataset.csv'
MODEL_PATH = 'results/model.joblib'
METRICS_PATH = 'results/metrics.json'

def task_load_data(**context):
    df = load_dataset(DATA_PATH)
    context['ti'].xcom_push(key='df', value=df.to_json())

def task_preprocess(**context):
    import pandas as pd
    df_json = context['ti'].xcom_pull(task_ids='load_data', key='df')
    df = pd.read_json(df_json)
    X, y, scaler = preprocess(df)
    joblib.dump((X, y, scaler), 'results/preprocessed.pkl')

def task_train_model():
    X, y, _ = joblib.load('results/preprocessed.pkl')
    model = train(X, y)
    joblib.dump(model, MODEL_PATH)

def task_evaluate_model():
    model = joblib.load(MODEL_PATH)
    X, y, _ = joblib.load('results/preprocessed.pkl')
    metrics = evaluate(model, X, y)
    joblib.dump(metrics, 'results/metrics.pkl')

def task_save_results():
    metrics = joblib.load('results/metrics.pkl')
    save_metrics(metrics, METRICS_PATH)

load_data = PythonOperator(
    task_id='load_data',
    python_callable=task_load_data,
    provide_context=True,
    dag=dag
)

preprocess_data = PythonOperator(
    task_id='preprocess_data',
    python_callable=task_preprocess,
    provide_context=True,
    dag=dag
)

train_model = PythonOperator(
    task_id='train_model',
    python_callable=task_train_model,
    dag=dag
)

evaluate_model = PythonOperator(
    task_id='evaluate_model',
    python_callable=task_evaluate_model,
    dag=dag
)

save_results = PythonOperator(
    task_id='save_results',
    python_callable=task_save_results,
    dag=dag
)

load_data >> preprocess_data >> train_model >> evaluate_model >> save_results
