from transformers import pipeline
from guardrails_project.constants import CLASSIFIER_THRESHOLD

class CheckResponse:

    _classifier = None
    threshold = CLASSIFIER_THRESHOLD  # Default threshold for classification

    @classmethod
    def _get_classifier(cls):
        if cls._classifier is None:
            cls._classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return cls._classifier 

    @classmethod
    def is_refusal(cls, response: str, threshold: float = None) -> bool:
        """
        Restituisce True se la risposta è un rifiuto, altrimenti False.
        Usa zero-shot classification con soglia di confidenza.

        Args:
            response (str): Risposta generata dal modello LLM
            threshold (float): Soglia di confidenza minima per considerare il rifiuto valido

        Returns:
            bool: True se classificata come 'refusal' con sufficiente confidenza
        """

        classifier = cls._get_classifier()
        threshold = threshold if threshold is not None else cls.threshold

        labels = ["refusal", "not refusal"]
        result = classifier(response, candidate_labels=labels)

        label = result["labels"][0]
        score = result["scores"][0]

        return label == "refusal" and score >= threshold