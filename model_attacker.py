import torch
from torch.special import logit
from tqdm import tqdm
from dataclasses import dataclass
import re
import copy
import os
from .model_metrics import USE, BERTScore, GPT2LM, GrammarChecker
import random
import numpy as np
import  model_make

class FlipDetector:

    def __init__(self):
        path = os.path.dirname(__file__) + '/antonyms_vocab.dict'
        self.antonyms_vocab = torch.load(path)

    def remove_punctuation(self, text):
        punctuation = '!,;:?"\'/'
        text = re.sub(r'[{}]+'.format(punctuation), '', text)
        return text.strip()

    def get_atonyms(self, text):
        word_list = self.remove_punctuation(text).lower().split(" ")
        antonyms_dict = {}
        for idx in range(len(word_list)):
            if word_list[idx] in self.antonyms_vocab.keys():
                antonyms_dict[word_list[idx]] = idx
        return antonyms_dict, word_list

    def __call__(self, sentence1, sentence2, trigger_mag=2):
        (d1, word_list1), (d2, word_list2) = self.get_atonyms(sentence1), self.get_atonyms(sentence2)
        for k1, i1 in d1.items():
            for k2, i2 in d2.items():
                if (k1 in self.antonyms_vocab[k2]) and (abs(i1 - i2) <= trigger_mag):
                    context_start = i1 - 3 if (i1 - 3) <= 0 else 0
                    context_end = i1 + 3 if (i1 + 3) < len(word_list1) else len(word_list1) - 1

                    word_list1[i1] = k1 + f'[{k2}]'
                    context = " ".join(word_list1[context_start: context_end])
                    return True, {i1: k1, i2: k2, 'context': context}
        return False, None


@dataclass
class AttackResult:
    succeed: bool = False
    flipped: bool = False
    flipped_content: dict = None
    query: int = 0
    bpstep: int = 0
    step_in_seg: int = 0
    ori_error: int = 0
    adv_error: int = 0
    use: float = 0.0
    bs: float = 0.0
    gpt: float = 0.0
    grammar: float = 0.0
    PPL: float = 0.0
    edit_distance: int = 0.0
    edit_distance_normalized: float = 0.0
    total_loss: float = 0.0
    ori_sentence: str = None
    adv_sentence: str = None
    adv_sentences: list = None
    keyword_positions: list = None
    iterations: int = 0

dataset, encoded_dataset_train, num_labels, text_key, stop_ids = model_make.make(task = task, tokenizer_checkpoint = tokenizer_checkpoint, local_path = data_local_path)
class GradientGuidedKeywordLocalization:

    def __init__(self, model, prober, model_cuda_device, stop_ids):
        self.model = model
        self.prober = prober
        self.model_cuda_device = model_cuda_device
        self.stop_ids = stop_ids
        self.decode_mode = prober.decode_mode

    def model_forward(self, input_ids, attention_mask, labels, decode_hidden_index=-1):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            output_hidden_states=True)
        loss, logits, hiddens = output['loss'], output['logits'], output['hidden_states']
        return loss, logits, hiddens[decode_hidden_index]

    def compute_cw_loss(self, logits, labels, kappa=0.05):
        correct_class_logits = logits.gather(1, labels.view(-1, 1)).squeeze(-1)

        wrong_class_logits = logits.clone()
        wrong_class_logits[torch.arange(logits.size(0)), labels] = float('-inf')
        max_wrong_class_logits = wrong_class_logits.max(dim=1)[0]

        cw_loss = torch.clamp(max_wrong_class_logits - correct_class_logits + kappa, min=0).mean()
        return cw_loss

    def compute_gradients(self, input_ids, attention_mask, labels,
                          perturb_mask, perturb_layer_index, decode_hidden_index):
        self.model.zero_grad()

        loss, logits, hidden = self.model_forward(
            input_ids, attention_mask, labels, decode_hidden_index
        )

        cw_loss = self.compute_cw_loss(logits, labels)

        cw_loss.backward()

        if self.decode_mode == 'Bert':
            attention_layer = self.model.bert.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Roberta':
            attention_layer = self.model.roberta.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Albert':
            attention_layer = self.model.albert.encoder

        if hasattr(attention_layer, 'perturb') and attention_layer.perturb is not None:
            gradients = attention_layer.perturb.grad
            gradients = torch.mul(perturb_mask.T, gradients).detach()
            return gradients, cw_loss

        return None, cw_loss

    def compute_sensitivity_scores(self, embeddings, gradients):
        if gradients is None:
            return None

        grad_norms = torch.norm(gradients, dim=2)

        embed_norms = torch.norm(embeddings, dim=2)

        sensitivity_scores = grad_norms * embed_norms
        return sensitivity_scores

    def select_keyword_positions(self, sensitivity_scores, perturb_positions, k=2):
        if sensitivity_scores is None or len(perturb_positions) == 0:
            return perturb_positions[:k] if len(perturb_positions) > 0 else []

        perturb_scores = sensitivity_scores[0, perturb_positions]

        if len(perturb_positions) <= k:
            return perturb_positions
        else:
            topk_indices = torch.topk(perturb_scores, k).indices
            keyword_positions = [perturb_positions[i] for i in topk_indices.tolist()]
            return keyword_positions

    def apply_perturbation(self, embeddings, gradients, keyword_positions, epsilon=1.5):
        if gradients is None:
            return embeddings

        perturbed_embeddings = embeddings.clone()

        for pos in keyword_positions:
            if pos < embeddings.size(1):
                grad_direction = gradients[0, pos]
                grad_norm = torch.norm(grad_direction)

                if grad_norm > 0:
                    unit_grad = grad_direction / grad_norm
                    perturbed_embeddings[0, pos] += epsilon * unit_grad

        return perturbed_embeddings

    def random_cover(self, ids, pos=None):
        if len(pos) == 0:
            return ids

        random.seed(114514)
        rindex = random.randint(0, len(pos) - 1)
        rindex = pos[rindex]

        if self.decode_mode == 'Bert':
            ids[:, rindex] = 104
        elif self.decode_mode == 'Roberta':
            ids[:, rindex] = 50264
        elif self.decode_mode == 'Albert':
            ids[:, rindex] = 4

        return ids

    def execute(self, index, dataset, encoded_dataset, task=None,
                k=2, epsilon=1.5, max_iterations=20,
                perturb_layer_index=0, decode_hidden_index=-1,
                init_mag=3.0, bs_lower_limit=0.0, bs_upper_limit=0.85,
                SEED=114514):

        ori_input_ids = torch.tensor(encoded_dataset['input_ids'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        attention_mask = torch.tensor(encoded_dataset['attention_mask'][index]).unsqueeze(0).cuda(
            self.model_cuda_device)
        label = torch.tensor(encoded_dataset['label'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        perturb_positions = encoded_dataset['perturb_positions'][index]
        perturb_mask = torch.tensor(encoded_dataset['mask'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        decode_mask = torch.tensor(encoded_dataset['mask'][index]).unsqueeze(0).cuda(self.model_cuda_device)
        adv_start, adv_end = encoded_dataset['adv_start'][index], encoded_dataset['adv_end'][index]

        torch.manual_seed(SEED)
        input_ids = self.random_cover(ori_input_ids.clone(), perturb_positions)

        if self.decode_mode == 'Bert':
            p_layer = self.model.bert.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Roberta':
            p_layer = self.model.roberta.encoder.layer[perturb_layer_index].attention.self
        elif self.decode_mode == 'Albert':
            p_layer = self.model.albert.encoder
            p_layer.set_pos(perturb_layer_index)

        p_layer.perturb = None

        input_length = len(ori_input_ids[0])
        p_layer.p_init(input_length, init_mag=init_mag,
                       cuda_device=self.model_cuda_device, perturb_mask=perturb_mask)

        best_keyword_positions = []
        best_adv_text = ""

        for iteration in range(max_iterations):
            gradients, cw_loss = self.compute_gradients(
                input_ids, attention_mask, label, perturb_mask,
                perturb_layer_index, decode_hidden_index
            )

            embeddings = self.model.get_input_embeddings()(input_ids)

            sensitivity_scores = self.compute_sensitivity_scores(embeddings, gradients)

            keyword_positions = self.select_keyword_positions(
                sensitivity_scores, perturb_positions, k
            )

            perturbed_embeddings = self.apply_perturbation(
                embeddings, gradients, keyword_positions, epsilon
            )

            _, reconstruct_ids = self.prober.decode(
                self.model_forward(input_ids, attention_mask, label, decode_hidden_index)[2],
                origin_ids=ori_input_ids,
                decode_mask=decode_mask,
                perturb_positions=perturb_positions,
                stop_ids=self.stop_ids
            )

            adv_text = self.prober.ids_to_sentence(
                reconstruct_ids[0], adv_start=adv_start, adv_end=adv_end
            )

            if len(keyword_positions) > 0:
                best_keyword_positions = keyword_positions
                best_adv_text = adv_text

            if len(keyword_positions) >= k:
                break

        return best_keyword_positions, best_adv_text, iteration + 1


class ContextAwareLexicalSubstitution:

    def __init__(self, victim_model, tokenizer, use_scorer, gpt_scorer,
                 grammar_scorer, flip_detector, victim_cuda_device):
        self.victim_model = victim_model
        self.tokenizer = tokenizer
        self.use_scorer = use_scorer
        self.gpt_scorer = gpt_scorer
        self.grammar_scorer = grammar_scorer
        self.flip_detector = flip_detector
        self.victim_cuda_device = victim_cuda_device

    def generate_candidates(self, text, position, num_candidates=30):
        tokens = self.tokenizer.tokenize(text)
        if position >= len(tokens):
            return []

        original_token = tokens[position]

        candidates = []

        candidate_tokens = [
            original_token,
            original_token + "s",
            original_token + "ed",
            original_token + "ing",
        ]

        for cand_token in candidate_tokens[:num_candidates]:
            new_tokens = tokens.copy()
            new_tokens[position] = cand_token
            candidate_sentence = self.tokenizer.convert_tokens_to_string(new_tokens)
            candidates.append(candidate_sentence)

        return candidates

    def compute_multi_constraint_loss(self, original_text, candidate_text,
                                      weights, bs_values=None, gpt_values=None,
                                      grammar_values=None, penalty_weight=10):
        bs = self.use_scorer(original_text, candidate_text)
        gpt = self.gpt_scorer(candidate_text) - self.gpt_scorer(original_text)
        grammar = self.grammar_scorer(candidate_text) - self.grammar_scorer(original_text)

        def dynamic_normalize(value, values_array):
            if values_array is not None and len(values_array) > 1:
                min_val = min(values_array)
                max_val = max(values_array)
                if max_val > min_val:
                    return (value - min_val) / (max_val - min_val)
            return value

        norm_bs = dynamic_normalize(bs, bs_values)
        norm_gpt = dynamic_normalize(gpt, gpt_values)
        norm_grammar = dynamic_normalize(grammar, grammar_values)

        loss_bs = -norm_bs * weights.get('bs', 1.0)
        loss_gpt = norm_gpt * weights.get('gpt', 1.0)
        loss_grammar = norm_grammar * weights.get('grammar', 1.0)

        penalty = 0
        if bs < 0.8:
            penalty += penalty_weight
        if not (gpt <= 100 and gpt >= -100):
            penalty += penalty_weight

        total_loss = loss_bs + loss_gpt + loss_grammar + penalty

        return total_loss, {'bs': bs, 'gpt': gpt, 'grammar': grammar}

    def select_optimal_substitution(self, original_text, candidates,
                                    position, weights, original_label,
                                    task=None, _ori_sentence=None):
        if not candidates:
            return original_text, None, 0

        best_candidate = original_text
        best_loss = float('inf')
        best_metrics = {}
        attack_success = False

        bs_values = []
        gpt_values = []
        grammar_values = []

        for candidate in candidates:
            bs = self.use_scorer(original_text, candidate)
            gpt = self.gpt_scorer(candidate) - self.gpt_scorer(original_text)
            grammar = self.grammar_scorer(candidate) - self.grammar_scorer(original_text)

            bs_values.append(bs)
            gpt_values.append(gpt)
            grammar_values.append(grammar)

        for candidate in candidates:
            total_loss, metrics = self.compute_multi_constraint_loss(
                original_text, candidate, weights,
                bs_values, gpt_values, grammar_values
            )

            label_tensor = torch.tensor([original_label]).cuda(self.victim_cuda_device)
            succeed, _ = self.check_attack_success(
                candidate, label_tensor, task, _ori_sentence
            )

            if succeed and total_loss < best_loss:
                best_loss = total_loss
                best_candidate = candidate
                best_metrics = metrics
                attack_success = True

        return best_candidate, best_metrics, attack_success

    def check_attack_success(self, text, label, task=None, _ori_sentence=None):
        return False, 0.0

    def execute(self, initial_adv_text, keyword_positions, original_text,
                original_label, num_candidates=30, constraint_weights=None,
                task=None, _ori_sentence=None):
        if constraint_weights is None:
            constraint_weights = {'bs': 1.0, 'gpt': 1.0, 'grammar': 1.0}

        current_text = initial_adv_text
        total_queries = 0
        all_metrics = []

        for position in keyword_positions:
            candidates = self.generate_candidates(current_text, position, num_candidates)

            if not candidates:
                continue

            best_candidate, metrics, attack_success = self.select_optimal_substitution(
                original_text, candidates, position, constraint_weights,
                original_label, task, _ori_sentence
            )

            current_text = best_candidate
            total_queries += len(candidates)

            if metrics:
                all_metrics.append(metrics)

        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = np.mean([m[key] for m in all_metrics])

        return {
            'adversarial_text': current_text,
            'query_count': total_queries,
            'semantic_similarity': avg_metrics.get('bs', 0.0),
            'perplexity_change': avg_metrics.get('gpt', 0.0),
            'grammar_error_change': avg_metrics.get('grammar', 0.0)
        }


class TextJosherAttacker:

    def __init__(self, model, dataset, encoded_dataset, text_key, prober,
                 model_cuda_device, stop_ids, target_metric='bs'):

        self.model = model
        self.model_cuda_device = model_cuda_device
        self.dataset = dataset
        self.encoded_dataset = encoded_dataset
        self.encoded_dataset.set_format("numpy")
        self.text_key = text_key
        self.prober = prober
        self.decode_mode = prober.decode_mode
        self.stop_ids = stop_ids

        self.use_scorer = USE(prober.victim_cuda_device)
        self.bert_scorer = BERTScore(prober.victim_cuda_device) if target_metric == 'bs' else None
        self.gpt_scorer = GPT2LM(cuda=prober.victim_cuda_device)
        self.grammar_scorer = GrammarChecker()
        self.flip_detector = FlipDetector()

        self.keyword_locator = GradientGuidedKeywordLocalization(
            model=model,
            prober=prober,
            model_cuda_device=model_cuda_device,
            stop_ids=stop_ids
        )

        self.lexical_substitutor = ContextAwareLexicalSubstitution(
            victim_model=prober.victim,
            tokenizer=prober.tokenizer,
            use_scorer=self.use_scorer,
            gpt_scorer=self.gpt_scorer,
            grammar_scorer=self.grammar_scorer,
            flip_detector=self.flip_detector,
            victim_cuda_device=prober.victim_cuda_device
        )

    def attack_step(self, index=0, bs_lower_limit=0.0, bs_upper_limit=0.85,
                    k=2, epsilon=1.5, max_iterations=20,
                    num_candidates=30, constraint_weights=None,
                    perturb_layer_index=0, decode_hidden_index=-1,
                    init_mag=3.0, adv_lr=5.0, num_seg_steps=100,
                    num_adv_steps=20, use_random_cover=True,
                    show_info=False, task=None, SEED=114514):

        result = AttackResult()
        result.ori_sentence = self.dataset[index][self.text_key]

        if task == 'mnli_hypothesis':
            _ori_sentence = self.dataset[index]['premise']
        elif task == 'mnli_premise':
            _ori_sentence = self.dataset[index]['hypothesis']
        else:
            _ori_sentence = None

        label = torch.tensor(self.encoded_dataset['label'][index]).unsqueeze(0)
        attack_with_origin_succeed, _ = self.prober.attack(
            result.ori_sentence, label,
            _ori_sentence=_ori_sentence,
            task=task,
            print_info=False
        )

        if attack_with_origin_succeed:
            return None

        keyword_positions, initial_adv_text, iterations = self.keyword_locator.execute(
            index=index,
            dataset=self.dataset,
            encoded_dataset=self.encoded_dataset,
            task=task,
            k=k,
            epsilon=epsilon,
            max_iterations=max_iterations,
            perturb_layer_index=perturb_layer_index,
            decode_hidden_index=decode_hidden_index,
            init_mag=init_mag,
            bs_lower_limit=bs_lower_limit,
            bs_upper_limit=bs_upper_limit,
            SEED=SEED
        )

        result.keyword_positions = keyword_positions
        result.iterations = iterations

        if not initial_adv_text or initial_adv_text == result.ori_sentence:
            return None

        initial_success, _ = self.prober.attack(
            initial_adv_text, label,
            _ori_sentence=_ori_sentence,
            task=task,
            print_info=False
        )

        if initial_success:
            result.succeed = True
            result.adv_sentence = initial_adv_text

            result.bs = self.use_scorer(result.ori_sentence, result.adv_sentence)
            result.gpt = self.gpt_scorer(result.adv_sentence) - self.gpt_scorer(result.ori_sentence)
            result.grammar = self.grammar_scorer(result.adv_sentence) - self.grammar_scorer(result.ori_sentence)
            result.flipped, result.flipped_content = self.flip_detector(
                result.ori_sentence, result.adv_sentence
            )

            return result

        if constraint_weights is None:
            constraint_weights = {'bs': 1.0, 'gpt': 1.0, 'grammar': 1.0}

        substitution_result = self.lexical_substitutor.execute(
            initial_adv_text=initial_adv_text,
            keyword_positions=keyword_positions,
            original_text=result.ori_sentence,
            original_label=self.encoded_dataset['label'][index],
            num_candidates=num_candidates,
            constraint_weights=constraint_weights,
            task=task,
            _ori_sentence=_ori_sentence
        )

        result.adv_sentence = substitution_result['adversarial_text']
        result.query = substitution_result['query_count']
        result.bs = substitution_result['semantic_similarity']
        result.gpt = substitution_result['perplexity_change']
        result.grammar = substitution_result['grammar_error_change']

        final_success, _ = self.prober.attack(
            result.adv_sentence, label,
            _ori_sentence=_ori_sentence,
            task=task,
            print_info=False
        )

        result.succeed = final_success
        result.flipped, result.flipped_content = self.flip_detector(
            result.ori_sentence, result.adv_sentence
        )

        return result