from dna.classes.main import DNA
from dna.classes.experimental import Experimental

# Backward-compatibility alias: the research class was previously named RVLM.
RVLM = Experimental

__all__ = ["DNA", "Experimental", "RVLM"]
