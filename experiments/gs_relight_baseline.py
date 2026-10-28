"""The user's explicit rollback selection; never rewrite historical viewers."""
import json
from pathlib import Path

CONFIG = Path(__file__).with_suffix('.json')


def selected_baseline():
    if not CONFIG.exists():
        return None
    config = json.loads(CONFIG.read_text())
    return config if config.get('active') == 'pre_spec_transport_v3' else None
