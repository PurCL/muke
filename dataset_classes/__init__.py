from .counterfact import CounterFactDataset
from .mquake import MQUAKEDataset
from .editevery import EditeveryDataset
from .unke import UnKEDataset
from .hallucination import HallucinationDataset

from .templates import (
    TEMPLATE_DICT,
)

DS_DICT = {
    "unke": UnKEDataset,
    "cf": CounterFactDataset,
    "mquake": MQUAKEDataset,
    "editevery": EditeveryDataset,
    "hallucination": HallucinationDataset,
}

__all__ = [
    "DS_DICT",
    "TEMPLATE_DICT",
]