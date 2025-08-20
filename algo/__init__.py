from .memit import MEMITHyperParams, apply_memit_to_model
from .memit_Mat import MEMITMatHyperParams, apply_memit_Mat_to_model
from .memit_ARE import MEMITAREHyperParams, apply_memit_ARE_to_model

from .unke import unkeHyperParams, apply_unke_to_model
from .unke_Mat import unkeMatHyperParams, apply_unke_Mat_to_model
from .unke_ARE import unkeAREHyperParams, apply_unke_ARE_to_model

from .WISE import WISEHyperParams, apply_wise_to_model

UNKE_DICT = {
    "unke_Mat": (unkeMatHyperParams, apply_unke_Mat_to_model),
    "unke_ARE": (unkeAREHyperParams, apply_unke_ARE_to_model),
    "unke": (unkeHyperParams, apply_unke_to_model),
}

ALPHAEDIT_DICT = {
    # "AlphaEdit_ARE": (AlphaEditAREHyperParams, apply_AlphaEdit_ARE_to_model),
    # "AlphaEdit": (AlphaEditHyperParams, apply_AlphaEdit_to_model),
}

MEMIT_DICT = {
    "MEMIT_Mat": (MEMITMatHyperParams, apply_memit_Mat_to_model),
    "MEMIT_ARE": (MEMITAREHyperParams, apply_memit_ARE_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
}

WISE_DICT = {
    "WISE": (WISEHyperParams, apply_wise_to_model),
}


ALG_DICT = {
    **UNKE_DICT,
    **ALPHAEDIT_DICT,
    **MEMIT_DICT,
    **WISE_DICT,
}

__all__ = [
    "UNKE_DICT",
    "ALPHAEDIT_DICT",
    "MEMIT_DICT",
    "WISE_DICT",
    "ALG_DICT",
]