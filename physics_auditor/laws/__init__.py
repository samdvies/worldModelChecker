"""Registry of all FLOOR-suite physical laws.

ALL_LAWS contains every law, including EVAL-ONLY laws that must never enter
the predictor/encoder training distribution. TRAIN_LAWS is the original
four-law subset that physics_auditor/generator/dataset.py's
train_clips()/val_clips() must iterate over instead of ALL_LAWS.

physics_auditor/generator/dataset.py currently iterates ALL_LAWS in both
train_clips() and val_clips() (see `_obey_clips`). That is a one-line change
this module does not make itself (dataset.py is owned by another
workstream): those two functions should be switched to iterate
TRAIN_LAWS.values() instead of ALL_LAWS.values(). Everything else that
consumes ALL_LAWS (eval_pairs, probe_pairs, the report card, the gallery,
mechanistic data, train_stacks' own seed-disjoint probe/stack training) is
fine iterating the full ALL_LAWS set -- the critical invariant is only about
the predictor/encoder's own pretraining clips.
"""
from physics_auditor.laws.gravity import GravityLaw
from physics_auditor.laws.permanence import PermanenceLaw
from physics_auditor.laws.permanence_ext import PermanenceExtLaw
from physics_auditor.laws.solidity import SolidityLaw
from physics_auditor.laws.support import SupportLaw
from physics_auditor.laws.support_hard import SupportHardLaw

TRAIN_LAWS: dict[str, type] = {
    "support": SupportLaw,
    "permanence": PermanenceLaw,
    "solidity": SolidityLaw,
    "gravity": GravityLaw,
}

ALL_LAWS: dict[str, type] = {
    **TRAIN_LAWS,
    "permanence-ext": PermanenceExtLaw,
    "support-hard": SupportHardLaw,
}
