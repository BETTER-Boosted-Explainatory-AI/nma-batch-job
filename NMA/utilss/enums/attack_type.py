from enum import Enum

class AttackType(Enum):
    """Enum for supported adversarial attack types"""
    FGSM = "fgsm"
    PGD = "pgd"
    DEEPFOOL = "deepfool"