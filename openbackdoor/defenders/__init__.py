from .defender import Defender
from .cube_defender import CUBEDefender, CasualCUBEDefender
from .graceful_defender import GraCeFulDefender
from .onion_defender import ONIONDefender
from .bki_defender import BKIDefender
from .rap_defender import RAPDefender
from .protegofed_defender import ProtegoFedDefender

DEFENDERS = {
    "base": Defender,
    'cube': CUBEDefender,
    'casualcube':CasualCUBEDefender,
    'graceful':GraCeFulDefender,
    'onion':ONIONDefender,
    'bki':BKIDefender,
    'rap':RAPDefender,
    'protegofed':ProtegoFedDefender

    
}

def load_defender(config):
    return DEFENDERS[config["name"].lower()](**config)
