import tensorflow as tf
import numpy as np
from adversarial_files.adversarial_attack import AdversarialAttack
from utilss.photos_utils import get_cached_preprocess_function

class DeepFoolAttack(AdversarialAttack):
    """
    DeepFool attack implementation.
    Finds the minimal perturbation needed to change the classification by iteratively
    linearizing the decision boundary.
    
    Enhanced to support both CIFAR-100 and ImageNet datasets, and both similarity
    and distance-based hierarchical clustering.
    """
    
    def __init__(self, class_names, batch_gradients=True, **kwargs):
        """
        Initialize DeepFool attack.
        
        Args:
            model: The model to attack
            class_names: List of class names
        """

        super().__init__(
            epsilon=kwargs.get("epsilon") if kwargs.get("epsilon") is not None else 1,
            alpha=kwargs.get("alpha") if kwargs.get("alpha") is not None else 0.1,
            num_steps=kwargs.get("num_steps") if kwargs.get("num_steps") is not None else 70
            )
        self.class_names = class_names
        self.overshoot = kwargs.get("overshoot") if kwargs.get("overshoot") is not None else 0.02
        self.num_classes = kwargs.get("num_classes") if kwargs.get("num_classes") is not None else 10
        self.batch_gradients = batch_gradients

    def preprocess_image_array(self, model, img_array):
        """
        Preprocess a numpy array image for ResNet
        
        Args:
            img_array: Numpy array of image data (can be in [0,1] or [0,255] range)
            
        Returns:
            Preprocessed image tensor ready for model input
        """
        # Make sure image is in [0, 255] range
        if np.max(img_array) <= 1.0:
            img_array = img_array * 255.0
            
        # Apply ResNet preprocessing
        preprocess_input = get_cached_preprocess_function(model)
        preprocessed = preprocess_input(img_array)
        return tf.convert_to_tensor(preprocessed, dtype=tf.float32)

    def batch_compute_gradients(self, x_tensor, model, current_class, target_classes):
        """
        Compute gradients for multiple target classes in a single batch operation
        
        Args:
            x_tensor: Input tensor
            model: Model to attack
            current_class: Current predicted class
            target_classes: List of target classes
            
        Returns:
            List of gradients for each target class
        """
        gradients = []
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image using our new function
            x_preprocessed = self.preprocess_image_array(model, x_tensor)
            
            # Get predictions
            preds = model(x_preprocessed)
            
            # Store current class prediction
            current_pred = preds[0, current_class]
        
        # Compute gradient for each target class
        for k in target_classes:
            try:
                # Target loss function: maximize target class and minimize current class
                loss = preds[0, k] - current_pred
                
                gradient = tape.gradient(loss, x_tensor)
                if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                    gradients.append(None)
                else:
                    gradients.append(gradient.numpy())
            except Exception as e:
                print(f"Error computing gradient for class {k}: {e}")
                gradients.append(None)
        
        # Free resources
        del tape
                
        return gradients

    def compute_gradient(self, x_tensor, model, current_class, target_class):
        """
        Compute gradient of (target_class - current_class) with respect to x
        
        Args:
            x_tensor: Input tensor
            model: Model to attack
            current_class: Current predicted class
            target_class: Target class
            
        Returns:
            Gradient
        """
        
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image using our new function
            x_preprocessed = self.preprocess_image_array(model, x_tensor)
            
            # Get predictions
            preds = model(x_preprocessed, training=False)
            
            # Target loss function: maximize target class and minimize current class
            loss = preds[0, target_class] - preds[0, current_class]
        
        # Compute gradient
        try:
            gradient = tape.gradient(loss, x_tensor)
            if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                return None
            return gradient.numpy()
        except Exception as e:
            print(f"Error computing gradient: {e}")
            return None

    def fgsm_gradient_optimized(self, x_tensor, model, target_class):
        """
        Optimized FGSM gradient computation using TensorFlow
        
        Args:
            x_tensor: Input tensor
            model: Model to attack
            target_class: Target class
            
        Returns:
            Gradient
        """
        
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            
            # Preprocess the image using our new function
            x_preprocessed = self.preprocess_image_array(model, x_tensor)
            
            # Get predictions
            preds = model(x_preprocessed)
            
            # Target class score
            target_score = preds[0, target_class]
        
        # Compute gradient
        try:
            gradient = tape.gradient(target_score, x_tensor)
            if gradient is None or tf.reduce_all(tf.equal(gradient, 0)):
                return None
            return gradient.numpy()
        except Exception as e:
            print(f"Error computing FGSM gradient: {e}")
            return None
    
    def attack(self, model, image):
        """
        Optimized memory-efficient implementation of DeepFool
        
        Args:
            image: Input image (single image without batch dimension) in [0,1] range
            model: The model to attack
            class_names: List of all class names
            num_classes: Number of classes to consider for the attack
            max_iter: Maximum number of iterations
            overshoot: Overshoot parameter
            batch_gradients: Whether to compute gradients in batch (faster but more memory)
            
        Returns:
            perturbed_image, original_label, adversarial_label, original_probs, adversarial_probs
        """
        print("Starting DeepFool attack...")

        # Make a copy of the image to avoid modifying the original
        x_adv = image
        
        # Add batch dimension if needed
        if len(x_adv.shape) == 3:
            x_adv_batch = np.expand_dims(x_adv, axis=0)
        else:
            x_adv_batch = x_adv
        
        # Preprocess the image for initial prediction - using our new function
        x_adv_preprocessed = self.preprocess_image_array(model, x_adv_batch)
        
        # Get initial prediction
        original_preds = model.predict(x_adv_preprocessed)
        original_label = np.argmax(original_preds)
        # original_probs = decode_predictions(original_preds, top=5)[0]
        current_label = original_label
        
        # Convert numpy array to tensor once
        x_tensor = tf.Variable(x_adv_batch, dtype=tf.float32)
        
        # Cache for gradients to avoid recomputation - use tuples of ints as keys (not numpy arrays)
        gradient_cache = {}
        
        # Loop for max_iter iterations
        for i in range(self.num_steps):
            # start_time = time.time()
            
            # Preprocess the current image - using our new function
            x_preprocessed = self.preprocess_image_array(model, x_tensor.numpy())
            
            # Get current prediction
            current_preds = model.predict(x_preprocessed)[0]
            current_label = np.argmax(current_preds)
            # print(f"Current label: {class_names[current_label]}, Original label: {class_names[original_label]}")
            
            # Check if we've already succeeded in finding an adversarial example
            if current_label != original_label:
                # print(f"Found adversarial example at iteration {i}")
                break
            
            # Initialize variables to track minimum perturbation
            min_dist = float('inf')
            best_pert = None
            target_class = None
            
            # Select top classes by prediction score (excluding current class)
            if self.num_classes < len(self.class_names):
                # print(f"Selecting top {num_classes} classes for attack")
                top_indices = np.argsort(current_preds)[::-1]
                classes_to_check = [idx for idx in top_indices[:self.num_classes+1] if idx != current_label][:self.num_classes]
            else:
                classes_to_check = [c for c in range(len(self.class_names)) if c != current_label]
            
            # Compute gradients in batch if enabled (much faster but uses more memory)
            if self.batch_gradients:
                grads = self.batch_compute_gradients(x_tensor, model, current_label, classes_to_check)

            # For each target class, find the minimum perturbation
            for k_idx, k in enumerate(classes_to_check):
                # Get gradient either from batch computation or compute individually
                if self.batch_gradients:
                    grad = grads[k_idx]
                else:
                    # Use cached gradient if available - use string key
                    cache_key = f"{int(current_label)}_{int(k)}"
                    if cache_key in gradient_cache:
                        grad = gradient_cache[cache_key]
                    else:
                        grad = self.compute_gradient(x_tensor, model, current_label, k)
                        gradient_cache[cache_key] = grad
                
                if grad is None:
                    continue
                        
                # Calculate f_k - f_current (loss)
                f_current = current_preds[current_label]
                f_k = current_preds[k]
                loss = f_k - f_current
                
                # Normalize gradient
                grad_norm = np.linalg.norm(grad) + 1e-8
                grad_normalized = grad / grad_norm
                
                # Calculate distance to decision boundary
                dist = abs(loss) / grad_norm
                
                # Check if this is the closest boundary
                if dist < min_dist:
                    min_dist = dist
                    best_pert = grad_normalized
                    target_class = k
            
            # If no valid perturbation found, try fallback strategies
            if best_pert is None:
                # Try harder with FGSM-style update for top classes
                for k in classes_to_check[:5]:  # Try just top 5 classes
                    g = self.fgsm_gradient_optimized(x_tensor, model, k)
                    if g is not None and np.any(g):
                        g_norm = np.linalg.norm(g) + 1e-8
                        best_pert = g / g_norm
                        target_class = k
                        min_dist = 0.01  # Small fixed step size
                        break
                
                # If still no valid perturbation, use random perturbation as last resort
                if best_pert is None:
                    # print("Using random perturbation")
                    best_pert = np.random.normal(0, 1, size=x_adv_batch.shape)
                    best_pert = best_pert / (np.linalg.norm(best_pert) + 1e-8)
                    min_dist = 0.01
            
            # Apply perturbation with overshoot
            pert = (1 + self.overshoot) * min_dist * best_pert
            
            # Update tensor
            x_tensor.assign(tf.clip_by_value(x_tensor + tf.convert_to_tensor(pert, dtype=tf.float32), 0, 1))
            
            # Clear gradient cache at regular intervals to free memory
            if i % 5 == 0:
                gradient_cache.clear()
        
        # Get final image and prediction
        x_adv = x_tensor.numpy()[0]
        x_adv_preprocessed = self.preprocess_image_array(model, np.expand_dims(x_adv, axis=0))
        final_preds = model.predict(x_adv_preprocessed)[0]
        adversarial_label = np.argmax(final_preds)
        
        # Print attack success
        if adversarial_label != original_label:
            print("Attack success!")
        else:
            print("\nAttack failed to change prediction.")
        
        return x_adv