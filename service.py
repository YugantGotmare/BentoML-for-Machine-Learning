import numpy as np
import bentoml as bl
from bentoml.io import NumpyNdarray

model_runner = bl.sklearn.get("iris_clf:latest").to_runner()

svc = bl.Service("iris_classifier", runners=[model_runner])

@svc.api(input = NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = model_runner.predict.run(input_series)
    return result