from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
FEATURE_NAMES = ("age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(strict=True)

    age: FiniteFloat
    sex: FiniteFloat
    bmi: FiniteFloat
    bp: FiniteFloat
    s1: FiniteFloat
    s2: FiniteFloat
    s3: FiniteFloat
    s4: FiniteFloat
    s5: FiniteFloat
    s6: FiniteFloat

    def ordered_values(self) -> list[float]:
        return [getattr(self, name) for name in FEATURE_NAMES]
