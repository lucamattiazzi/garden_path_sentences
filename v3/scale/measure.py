"""Text-only scoring through the full multimodal checkpoint wrapper."""

from v3.direct.measure import DirectMeasurements


class ScaleMeasurements(DirectMeasurements):
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer
        self.layers = model.model.language_model.layers
        self.device = next(model.parameters()).device
