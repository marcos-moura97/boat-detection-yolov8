import numpy as np
import pandas as pd
from typing import Tuple, List, Any
import itertools

class ExperimentsDesign:
    def __init__(self, factors: Tuple[List[Any]]):
        self.factors = factors
        self.full_factorial = self._create_full_factorial_design()
        self.fractional_factorial = self._sample_fractional_factorial(0.25)

    def _create_full_factorial_design(self) -> np.ndarray:
        # Generate the full factorial design using itertools.product
        return np.array(list(itertools.product(*self.factors)))

    def _sample_fractional_factorial(self, fraction: float) -> np.ndarray:
        np.random.seed(0)  # For reproducibility
        sample_size = int(len(self.full_factorial) * fraction)
        return self.full_factorial[np.random.choice(self.full_factorial.shape[0], size=sample_size, replace=False)]

    def to_dataframe(self) -> pd.DataFrame:
        columns = ['model_size', 'class_levels', 'epochs', 'img_resolutions']
        df = pd.DataFrame(self.fractional_factorial, columns=columns)
        return df

    def save_to_csv(self, file_path: str) -> None:
        df = self.to_dataframe()
        df.to_csv(file_path, sep=',', encoding='utf-8', index=False, header=True)
