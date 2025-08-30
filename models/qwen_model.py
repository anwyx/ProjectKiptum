from maestro.trainer.models.qwen_2_5_vl.checkpoints import load_model, OptimizationStrategy

class QwenVLModel:
    def __init__(self, model_id_or_path, min_pixels, max_pixels, optimization_strategy=OptimizationStrategy.NONE):
        self.model_id_or_path = model_id_or_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.optimization_strategy = optimization_strategy
        self.processor = None
        self.model = None
        
    def load(self):
        """Load and return the model and processor"""
        self.processor, self.model = load_model(
            model_id_or_path=self.model_id_or_path,
            optimization_strategy=self.optimization_strategy,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )
        return self.processor, self.model