from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "airflow",
    "retries": 0,
}


with DAG(
    dag_id="mlops_pipeline",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="*/5 * * * *",
    catchup=False,
) as dag:

    data_stage = BashOperator(
        task_id="data_engineering",
        bash_command="python /opt/project/code/datasets/prepare_data.py",
    )

    model_stage = BashOperator(
        task_id="model_engineering",
        bash_command="python /opt/project/code/models/train_model.py",
    )

    deploy_stage = BashOperator(
        task_id="deploy",
        bash_command=(
            "export DOCKER_BUILDKIT=0 COMPOSE_DOCKER_CLI_BUILD=0; "
            "docker image rm -f docker.io/library/project-api:latest project-api:latest || true && "
            "docker image rm -f docker.io/library/project-app:latest project-app:latest || true && "
            "docker builder prune -f || true && "
            "docker compose -f /opt/project/docker-compose.yml build --pull --no-cache && "
            "docker compose -f /opt/project/docker-compose.yml up -d --force-recreate"
        ),
    )

    data_stage >> model_stage >> deploy_stage


