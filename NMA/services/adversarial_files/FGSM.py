import tensorflow as tf
import numpy as np
from adversarial_files.adversarial_attack import AdversarialAttack


class FGSMAttack(AdversarialAttack):
    """
    Fast Gradient Sign Method attack implementation.
    FGSM is a one-step attack that perturbs an image in the direction of the gradient
    of the loss with respect to the input.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    
    def __init__(self, **kwargs):
        super().__init__(
            epsilon=kwargs.get("epsilon") if kwargs.get("epsilon") is not None else 7,
            alpha=kwargs.get("alpha") if kwargs.get("alpha") is not None else 0.1,
            num_steps=kwargs.get("num_steps") if kwargs.get("num_steps") is not None else 1
            )
    
    def attack(self, model, image):
        """
        Implements Fast Gradient Sign Method (FGSM) attack.
        
        Args:
            model: The model to attack
            image: Input image (with batch dimension)
            epsilon: Attack strength parameter
            
        Returns:
            Adversarial image
        """
        print("Starting FGSM attack...")
        # Important: Make sure image is in the format expected by the model
        # For ResNet, this is typically BGR with specific mean subtraction
        
        # Create a copy to avoid modifying the original
        x_adv = tf.identity(image)
        
        # Get number of classes
        num_classes = model.output_shape[1]
        
        # Get current prediction
        current_pred = model(image)
        original_class_idx = tf.argmax(current_pred[0]).numpy()
        
        # Choose a target approximately opposite in the class space
        target_class_idx = (original_class_idx + num_classes // 2) % num_classes
        
        # Create one-hot encoded target
        target = tf.one_hot(target_class_idx, num_classes)
        target = tf.expand_dims(target, axis=0)  # Add batch dimension

        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            pred = model(x_adv)
            # For targeted attack, we want to maximize the probability of the target class
            loss = tf.keras.losses.categorical_crossentropy(target, pred)

        # Compute gradient
        gradients = tape.gradient(loss, x_adv)
        
        # For targeted attack, use negative sign
        perturbation = -self.epsilon * tf.sign(gradients)
        
        # Add perturbation to images
        x_adv = x_adv + perturbation
        
        # Make sure to clip in the same range as your original image
        # For ResNet preprocessed images, this might not be [0,1]
        # Clip relative to the original range
        mean = np.array([103.939, 116.779, 123.68])

        # We need to clip in a way that after deprocessing, the values stay in [0,255]
        x_adv = tf.clip_by_value(x_adv, -mean, 255.0 - mean)

        x_adv = tf.squeeze(x_adv, axis=0)
        
        return x_adv