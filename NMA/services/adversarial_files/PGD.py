import tensorflow as tf
from adversarial_files.adversarial_attack import AdversarialAttack

class PGDAttack(AdversarialAttack):
    """
    Projected Gradient Descent attack implementation.
    PGD is one of the strongest first-order attacks, iteratively taking gradient
    steps and projecting back onto an ε-ball around the original image.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    def __init__(self, **kwargs):
        super().__init__(
            epsilon=kwargs.get("epsilon") if kwargs.get("epsilon") is not None else 0.7,
            alpha=kwargs.get("alpha") if kwargs.get("alpha") is not None else 0.1,
            num_steps=kwargs.get("num_steps") if kwargs.get("num_steps") is not None else 70
            )

    def attack(self, model, image):
        """
        Perform targeted PGD attack to force prediction toward a class far from the original class.
        Automatically selects a target class far from the original class group.
        """
        print("Starting PGD attack...")
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
        
        # Store the original image for proper clipping
        x_orig = tf.identity(image)
        x_adv = tf.identity(image)

        for _ in range(self.num_steps):
            with tf.GradientTape() as tape:
                tape.watch(x_adv)
                pred = model(x_adv)
                # For targeted attack, we want to maximize the probability of the target class
                # or minimize the distance between prediction and target
                loss = tf.keras.losses.categorical_crossentropy(target, pred)

            # Compute gradient
            grad = tape.gradient(loss, x_adv)

            # Apply perturbation
            x_adv = x_adv - self.alpha * tf.sign(grad)  # Note the minus sign

            # Project back to the epsilon ball around the original image
            x_adv = tf.clip_by_value(x_adv, x_orig - self.epsilon, x_orig + self.epsilon)
            
            # Ensure values stay in the valid range
            x_adv = tf.clip_by_value(x_adv, -123.68, 151.061)
        
        x_adv = tf.squeeze(x_adv, axis=0)

        return x_adv