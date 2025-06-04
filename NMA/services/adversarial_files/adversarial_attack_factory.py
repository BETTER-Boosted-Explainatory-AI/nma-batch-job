from adversarial_files.PGD import PGDAttack
from adversarial_files.FGSM import FGSMAttack
from adversarial_files.DeepFool import DeepFoolAttack
from utilss.enums.attack_type import AttackType

def get_attack(attack_name: str, **kwargs):
    """
    Factory function to get the appropriate attack class based on the attack name.

    Args:
        attack_name (AttackType): Enum value representing the attack type.

    Returns:
        AdversarialAttack: An instance of the corresponding attack class.
    """
    attack_name = attack_name.lower()
    
    attack_classes = {
        AttackType.PGD.value: PGDAttack,
        AttackType.FGSM.value: FGSMAttack,
        AttackType.DEEPFOOL.value: DeepFoolAttack
    }

    if attack_name not in attack_classes:
        raise ValueError(f"Unknown attack name: {attack_name}")

    return attack_classes[attack_name](**kwargs)