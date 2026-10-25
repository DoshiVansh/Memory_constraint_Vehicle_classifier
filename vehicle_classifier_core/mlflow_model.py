"""PyFunc wrapper for loading a registered checkpoint through the legacy predictor."""

import mlflow.pyfunc


class VehicleClassifierPyfunc(mlflow.pyfunc.PythonModel):
    """Accept a DataFrame with ``image_path`` and return class indices."""

    def load_context(self, context):
        from .predictor import VehicleClassifier

        self.classifier = VehicleClassifier(context.artifacts["checkpoint"])

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "columns"):
            paths = model_input["image_path"].tolist()
        elif isinstance(model_input, (list, tuple)):
            paths = model_input
        else:
            paths = [model_input]
        return [self.classifier.predict(str(path)) for path in paths]
