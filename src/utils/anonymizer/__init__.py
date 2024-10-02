from src.utils.anonymizer.base import Anonymizer
from src.utils.anonymizer.global_anonymizer import GlobalAnonymizer
from src.utils.anonymizer.session_anonymizer import SessionAnonymizer


def get_anonymizer(
        strategy: str,
        anonymize_entity: bool = True,
        anonymize_rel: bool = True,
        use_tqdm: bool = False,
) -> Anonymizer:
    """Get anonymizer by strategy name."""
    if strategy == "global":
        return GlobalAnonymizer(
            anonymize_entity=anonymize_entity,
            anonymize_rel=anonymize_rel,
            use_tqdm=use_tqdm
        )
    else: # strategy == "session":
        return SessionAnonymizer(
            anonymize_entity=anonymize_entity,
            anonymize_rel=anonymize_rel,
            use_tqdm=use_tqdm
        )
