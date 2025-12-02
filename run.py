#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import copy
import tensorflow as tf
import tensorflow_hub as hub

@dataclass
class AttackResult:

    succeed: bool = False
    original_sentence: str = None
    adversarial_sentence: str = None
    perturbed_positions: List[int] = None
    query_count: int = 0
    semantic_similarity: float = 0.0
    perplexity: float = 0.0
    grammar_errors: int = 0
class GradientGuidedKeywordLocalization:
    def __init__(self, surrogate_model, tokenizer, mlm_head, cuda_device=0):
        self.surrogate = surrogate_model
        self.tokenizer = tokenizer
        self.mlm_head = mlm_head
        self.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.surrogate.to(self.device)
        self.mlm_head.to(self.device)
    def preprocess(self, text: str, target_label: int):
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            embeddings = self.surrogate.get_input_embeddings()(inputs['input_ids'])
        return inputs, embeddings, target_label
    def compute_sensitivity_scores(self, embeddings, target_label, surrogate_output=None):
        if surrogate_output is None:
            with torch.no_grad():
                surrogate_output = self.surrogate(inputs_embeds=embeddings)
        logits = surrogate_output.logits
        target_class_logits = logits[:, target_label]
        wrong_class_logits = logits.clone()
        wrong_class_logits[:, target_label] = float('-inf')
        max_wrong_logits = wrong_class_logits.max(dim=1).values
        kappa = 0.05
        cw_loss = torch.clamp(max_wrong_logits - target_class_logits + kappa, min=0).mean()
        embeddings.requires_grad = True
        cw_loss.backward()
        gradients = embeddings.grad
        sensitivity_scores = torch.norm(gradients, dim=2) * torch.norm(embeddings, dim=2)
        embeddings.grad = None
        embeddings.requires_grad = False
        return sensitivity_scores.squeeze(), gradients

    def select_keyword_positions(self, sensitivity_scores, k: int = 2):

        valid_positions = list(range(1, len(sensitivity_scores) - 1))
        scores = sensitivity_scores[valid_positions]
        topk_indices = torch.topk(scores, min(k, len(scores))).indices
        keyword_positions = [valid_positions[i] for i in topk_indices.tolist()]

        return keyword_positions

    def apply_perturbation(self, embeddings, gradients, keyword_positions, epsilon: float = 1.5):
        perturbed_embeddings = embeddings.clone()

        for pos in keyword_positions:

            grad_direction = gradients[0, pos]
            grad_norm = torch.norm(grad_direction)

            if grad_norm > 0:

                unit_grad = grad_direction / grad_norm

                perturbed_embeddings[0, pos] += epsilon * unit_grad

        return perturbed_embeddings

    def decode_perturbed_embeddings(self, perturbed_embeddings, original_ids, context_mask):

        with torch.no_grad():
            logits = self.mlm_head(perturbed_embeddings)


        candidate_ids = torch.argmax(logits, dim=2)


        reconstructed_ids = torch.where(
            context_mask.bool(),
            candidate_ids,
            original_ids
        )

        return reconstructed_ids

    def execute(self, text: str, target_label: int, k: int = 2, epsilon: float = 1.5,
                max_iterations: int = 20) -> Tuple[List[int], str, int]:

        inputs, embeddings, target_label = self.preprocess(text, target_label)


        perturbed_embeddings = embeddings.clone()
        context_mask = torch.zeros_like(inputs['input_ids'], dtype=torch.bool)

        for iteration in range(max_iterations):

            sensitivity_scores, gradients = self.compute_sensitivity_scores(
                perturbed_embeddings, target_label
            )


            keyword_positions = self.select_keyword_positions(sensitivity_scores, k)


            for pos in keyword_positions:
                context_mask[0, pos] = True


            perturbed_embeddings = self.apply_perturbation(
                embeddings, gradients, keyword_positions, epsilon
            )


            reconstructed_ids = self.decode_perturbed_embeddings(
                perturbed_embeddings, inputs['input_ids'], context_mask
            )


            adversarial_text = self.tokenizer.decode(
                reconstructed_ids[0],
                skip_special_tokens=True
            )


            if iteration == 0 or len(keyword_positions) > 0:
                return keyword_positions, adversarial_text, iteration + 1

        return keyword_positions, adversarial_text, max_iterations


class ContextAwareLexicalSubstitution:

    def __init__(self, victim_model, tokenizer, use_scorer, gpt_scorer, grammar_scorer,
                 cuda_device=0):
        self.victim = victim_model
        self.tokenizer = tokenizer
        self.use_scorer = use_scorer
        self.gpt_scorer = gpt_scorer
        self.grammar_scorer = grammar_scorer
        self.device = torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')
        self.victim.to(self.device)

    def generate_candidates(self, text: str, position: int, num_candidates: int = 30) -> List[str]:

        tokens = text.split()
        if position >= len(tokens):
            return []
        masked_tokens = tokens.copy()
        masked_tokens[position] = '[MASK]'
        masked_text = ' '.join(masked_tokens)
        inputs = self.tokenizer(
            masked_text,
            return_tensors='pt',
            truncation=True
        ).to(self.device)
        candidates = []
        original_word = tokens[position]
        candidate_words = [
            original_word,
            original_word + 'ing',
            original_word + 'ed',
            'not ' + original_word,
        ]
        for word in candidate_words:
            candidate_tokens = tokens.copy()
            candidate_tokens[position] = word
            candidate_sentence = ' '.join(candidate_tokens)
            candidates.append(candidate_sentence)

        return candidates[:num_candidates]

    def compute_multi_constraint_loss(self, original_text: str, candidate_text: str,
                                      weights: Dict[str, float]) -> float:
        semantic_sim = self.use_scorer(original_text, candidate_text)
        loss_semantic = 1.0 - semantic_sim
        fluency_ppl = self.gpt_scorer(candidate_text)
        loss_fluency = fluency_ppl / 100.0
        grammar_errors = self.grammar_scorer(candidate_text)
        loss_grammar = grammar_errors
        total_loss = (
                weights.get('semantic', 0.5) * loss_semantic +
                weights.get('fluency', 0.3) * loss_fluency +
                weights.get('grammar', 0.2) * loss_grammar
        )
        return total_loss, {
            'semantic': semantic_sim,
            'fluency': fluency_ppl,
            'grammar': grammar_errors,
            'total_loss': total_loss
        }

    def select_optimal_substitution(self, original_text: str, candidates: List[str],
                                    position: int, weights: Dict[str, float]) -> str:
        if not candidates:
            return original_text

        best_candidate = original_text
        best_loss = float('inf')
        best_metrics = {}
        for candidate in candidates:
            total_loss, metrics = self.compute_multi_constraint_loss(
                original_text, candidate, weights
            )
            if total_loss < best_loss:
                best_loss = total_loss
                best_candidate = candidate
                best_metrics = metrics

        return best_candidate, best_metrics

    def execute(self, initial_adv_text: str, keyword_positions: List[int],
                original_text: str, weights: Dict[str, float] = None) -> AttackResult:
        if weights is None:
            weights = {'semantic': 0.5, 'fluency': 0.3, 'grammar': 0.2}

        result = AttackResult()
        result.original_sentence = original_text
        result.adversarial_sentence = initial_adv_text
        result.perturbed_positions = keyword_positions
        current_text = initial_adv_text
        for position in keyword_positions:
            candidates = self.generate_candidates(current_text, position)
            best_candidate, metrics = self.select_optimal_substitution(
                original_text, candidates, position, weights
            )
            current_text = best_candidate
            result.semantic_similarity = metrics.get('semantic', 0.0)
            result.perplexity = metrics.get('fluency', 0.0)
            result.grammar_errors = metrics.get('grammar', 0)
        result.adversarial_sentence = current_text
        result.query_count = len(keyword_positions) * 10
        return result


class TextJosherAttackFramework:
    def __init__(self, surrogate_model, victim_model, tokenizer, mlm_head,
                 use_scorer, gpt_scorer, grammar_scorer, cuda_device=0):
        self.keyword_locator = GradientGuidedKeywordLocalization(
            surrogate_model, tokenizer, mlm_head, cuda_device
        )
        self.lexical_substitutor = ContextAwareLexicalSubstitution(
            victim_model, tokenizer, use_scorer, gpt_scorer, grammar_scorer, cuda_device
        )
        self.tokenizer = tokenizer
        self.cuda_device = cuda_device

    def attack(self, text: str, true_label: int, target_label: Optional[int] = None,
               k: int = 2, epsilon: float = 1.5, max_iterations: int = 20,
               constraint_weights: Dict[str, float] = None) -> AttackResult:
        if target_label is None:
            target_label = self._get_most_confusable_label(text, true_label)
        keyword_positions, initial_adv_text, iterations = self.keyword_locator.execute(
            text, target_label, k, epsilon, max_iterations
        )
        result = self.lexical_substitutor.execute(
            initial_adv_text, keyword_positions, text, constraint_weights
        )
        result.succeed = self._check_attack_success(
            result.adversarial_sentence, true_label, target_label
        )
        return result

    def _get_most_confusable_label(self, text: str, true_label: int) -> int:
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.cuda_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.keyword_locator.surrogate(**inputs)
            logits = outputs.logits[0]

        logits[true_label] = float('-inf')
        target_label = torch.argmax(logits).item()

        return target_label

    def _check_attack_success(self, adv_text: str, true_label: int, target_label: int) -> bool:
        inputs = self.tokenizer(adv_text, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.cuda_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.lexical_substitutor.victim(**inputs)
            predicted_label = torch.argmax(outputs.logits, dim=1).item()

        return predicted_label != true_label


class MultiConstraintScorer:

    def __init__(self, use_path, gpt_model_path, cuda_device=0):


        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[cuda_device], 'GPU')
        self.use_model = hub.load(use_path)
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained(gpt_model_path)
        self.gpt_model = GPT2LMHeadModel.from_pretrained(gpt_model_path)
        self.gpt_model.to(torch.device(f'cuda:{cuda_device}'))
        import language_tool_python
        self.grammar_tool = language_tool_python.LanguageTool('en-US')

    def compute_use_similarity(self, text1: str, text2: str) -> float:
        embeddings = self.use_model([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)

    def compute_gpt_perplexity(self, text: str) -> float:
        inputs = self.gpt_tokenizer(text, return_tensors='pt')
        inputs = {k: v.to(self.gpt_model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.gpt_model(**inputs, labels=inputs['input_ids'])
            perplexity = torch.exp(outputs.loss).item()

        return perplexity

    def count_grammar_errors(self, text: str) -> int:
        matches = self.grammar_tool.check(text)
        return len(matches)


def main():
    cuda_device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained('model_name')

    surrogate_model = AutoModelForSequenceClassification.from_pretrained(
        'model_name'
    )

    victim_model = AutoModelForSequenceClassification.from_pretrained(
        'model_name'
    )

    mlm_model = AutoModelForMaskedLM.from_pretrained('model_name')
    mlm_head = mlm_model.cls

    class DummyScorer:
        def __call__(self, text1, text2):
            return 0.9

        def __call__(self, text):
            return 100.0

    attack_framework = TextJosherAttackFramework(
        surrogate_model=surrogate_model,
        victim_model=victim_model,
        tokenizer=tokenizer,
        mlm_head=mlm_head,
        use_scorer=DummyScorer(),
        gpt_scorer=DummyScorer(),
        grammar_scorer=DummyScorer(),
        cuda_device=cuda_device
    )
    result = attack_framework.attack

if __name__ == "__main__":
    main()