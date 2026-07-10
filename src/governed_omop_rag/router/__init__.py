"""Router : match déterministe d'abord (alignement officiel CIM-10 <-> SNOMED-CT),
RAG agentique uniquement sur le résidu. Borne le coût et garantit une baseline.
"""

from governed_omop_rag.router.deterministic import (
    JUSTIFICATION_MATCH,
    JUSTIFICATION_NO_MATCH,
    DeterministicRouter,
    OfficialMap,
    normalize_code,
    route_deterministic,
)
from governed_omop_rag.router.hybrid import HybridRouter

__all__ = [
    "JUSTIFICATION_MATCH",
    "JUSTIFICATION_NO_MATCH",
    "DeterministicRouter",
    "HybridRouter",
    "OfficialMap",
    "normalize_code",
    "route_deterministic",
]
