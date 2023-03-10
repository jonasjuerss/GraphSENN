"""
The custom motif dataset is copied from my MPhil Thesis
"""
import motif_dataset.motifs as motifs
from motif_dataset.motif_dataset import UniqueMotifCategorizationDataset

__all__ = [motifs.HouseMotif, motifs.TriangleMotif, motifs.FullyConnectedMotif, motifs.BinaryTreeMotif,
           UniqueMotifCategorizationDataset]

from motif_dataset.serializer import ArgSerializable

def _from_dict_obj(o):
    if isinstance(o, dict) and "_type" in o:
        obj_class = next((x for x in __all__ if x.__name__ == o["_type"]), None)
        if obj_class is None:
            raise ValueError(f"There is no motif type named {o['_type']}!")
        kwargs = {k: _from_dict_obj(v) for k, v in o["args"].items()}
        return obj_class(**kwargs)
    elif isinstance(o, list):
        return [_from_dict_obj(i) for i in o]
    else:
        return o

def from_dict(dict_repr: dict) -> ArgSerializable:
    return _from_dict_obj(dict_repr)