from rvlm.classes.dna import DNA
from rvlm.classes.experimental import Experimental

# Backward-compatibility alias: the research class was previously named RVLM.
RVLM = Experimental

__all__ = ["DNA", "Experimental", "RVLM"]
