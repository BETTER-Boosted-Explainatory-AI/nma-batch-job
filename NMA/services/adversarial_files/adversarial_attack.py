from abc import ABC, abstractmethod

class AdversarialAttack(ABC):
    """
    Abstract base class for adversarial attacks.
    Defines the interface that all specific attack implementations must follow.
    Focuses on core attack functionality and attack-specific analysis.
    """

    def __init__(self, epsilon, alpha, num_steps):
        """
        Initialize the attack.
        
        Args:
            model: The model to attack
            image: The input image to perturb
            epsilon: Perturbation magnitude
            alpha: Step size for the attack
            num_steps: Number of steps for the attack
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps

    @abstractmethod
    def attack(self, model, image):
        """
        Generate adversarial examples from input images.
        """
        pass
