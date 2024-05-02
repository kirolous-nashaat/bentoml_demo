import bentoml
import pickle
import numpy as np

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class Wine:
    def __init__(self) -> None:
        with open('wine_quality_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        with open('wine_quality_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)


    @bentoml.api
    def rateQuality(self,
                    sample_input: str
) -> str:
        sample_list = [float(x) for x in sample_input.split(";")]
        sample_input_array = np.array([sample_list])
        if self.scaler is not None:
           sample_input_array = self.scaler.transform(sample_input_array)
        result = self.model.predict(sample_input_array)[0]
        return str(result)