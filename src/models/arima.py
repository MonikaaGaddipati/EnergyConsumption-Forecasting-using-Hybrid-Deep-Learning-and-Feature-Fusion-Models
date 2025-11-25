 
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
except Exception as e:
    raise ImportError("statsmodels is required for ARIMAModel. Install with `pip install statsmodels`.") from e


OrderType = Union[Tuple[int, int, int], int, None]


class ARIMAModel:


    def __init__(self, order: OrderType = (3, 1, 2), *args, **kwargs):

        if "order" in kwargs:
            provided = kwargs.get("order")
        else:
            provided = order


        if isinstance(provided, (tuple, list)) and len(provided) == 3:
            self.order = tuple(int(x) for x in provided)

        elif isinstance(provided, int):
 
            self.order = (int(provided), 1, 0)
            warnings.warn(
                f"ARIMAModel received an integer for `order` ({provided}). Interpreting as (p,d,q)=({provided},1,0). "
                "If this was an accidental positional `input_dim`, call ARIMAModel() with no args or with order=(p,d,q).",
                UserWarning,
            )
        else:

            self.order = (3, 1, 2)

            if args and isinstance(args[0], int):
                warnings.warn(
                    "ARIMAModel was constructed with a positional int (likely `input_dim`) — "
                    "ignoring it and using default ARIMA order=(3,1,2). To pick a custom order, pass order=(p,d,q).",
                    UserWarning,
                )
            else:
   
                warnings.warn(
                    "ARIMAModel constructor received unexpected `order` input; using default (3,1,2).",
                    UserWarning,
                )

        self.model_fit: Optional[ARIMAResults] = None
        self.is_fitted: bool = False

    def fit(self, y_train):

       
        if isinstance(y_train, pd.DataFrame):

            if y_train.shape[1] > 1:
                warnings.warn(
                    "ARIMAModel.fit received a DataFrame with multiple columns; using the first column."
                )
            y_train = y_train.iloc[:, 0].values
        elif isinstance(y_train, pd.Series):
            y_train = y_train.values
        elif isinstance(y_train, (list, tuple)):
            y_train = np.array(y_train)
        elif not isinstance(y_train, np.ndarray):
            y_train = np.asarray(y_train)

        y_train = y_train.squeeze().astype(float)

        if y_train.ndim != 1:
            raise ValueError("ARIMAModel.fit requires a 1-D series-like input (shape (n,)).")

        try:
            model = ARIMA(y_train, order=self.order)
            self.model_fit = model.fit()
            self.is_fitted = True
        except Exception as e:
            raise RuntimeError(f"ARIMA fitting failed (order={self.order}): {e}") from e

    def forecast(self, steps: int = 1):
        """Forecast next `steps` values. Returns numpy array shape (steps,)."""
        if not self.is_fitted or self.model_fit is None:
            raise RuntimeError("ARIMAModel must be fitted before forecasting.")
        preds = self.model_fit.forecast(steps=steps)
        return np.asarray(preds)

    def summary(self):
        if not self.is_fitted or self.model_fit is None:
            return "ARIMAModel: not fitted"
        return self.model_fit.summary()

    def save(self, path: str):

        if not self.is_fitted or self.model_fit is None:
            raise RuntimeError("Fit the model before saving.")
        self.model_fit.save(path)

    def load(self, path: str):

        from statsmodels.tsa.arima.model import ARIMAResults

        self.model_fit = ARIMAResults.load(path)
        self.is_fitted = True


 





