import os
import mlflow
import logging

logging.basicConfig(
    level=logging.INFO,  # показывать INFO и важнее
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

mlflow.set_tracking_uri(mlflow_uri)
client = mlflow.MlflowClient()
best = max(client.search_runs(experiment_ids=['0']),
            key=lambda r: r.data.metrics.get('f1', 0))
log.info(f"лучший f1:, {best.data.metrics['f1']}")