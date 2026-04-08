import math

from datamodels import Product
from utils import get_real_purchases, make_basket

MAXQ = 5
W1 = 0.2
W2 = 0.5
W3 = 0.3

def calculate_persona_reward(true_personas_list, pred_actions):
    """
    Calculates a magnitude-aware cosine similarity reward.
    
    Args:
        true_personas_list: List of dicts e.g. [{'persona': 'A', 'confidence': 0.8}]
        pred_actions: List of objects with .persona and .confidence attributes
    """
    TruePersonas = {p['persona']: p['confidence'] for p in true_personas_list}
    PredPersonas = {p.persona: p.confidence for p in pred_actions}

    active_keys = set(TruePersonas.keys()) | set(PredPersonas.keys())
    
    if not active_keys:
        return 1.0

    dot_product = 0.0
    true_mag_sq = 0.0
    pred_mag_sq = 0.0

    for p in active_keys:
        t = TruePersonas.get(p, 0.0)
        p_val = PredPersonas.get(p, 0.0)
        
        dot_product += t * p_val
        true_mag_sq += t ** 2
        pred_mag_sq += p_val ** 2

    if true_mag_sq == 0 or pred_mag_sq == 0:
        return 1.0 if true_mag_sq == pred_mag_sq else 0.0

    true_mag = math.sqrt(true_mag_sq)
    pred_mag = math.sqrt(pred_mag_sq)

    cosine_sim = dot_product / (true_mag * pred_mag)

    magnitude_ratio = min(true_mag, pred_mag) / max(true_mag, pred_mag)

    reward = cosine_sim * magnitude_ratio

    return max(0.0, min(1.0, reward))

def calculate_product_ranking_reward(purchase_history: list[Product], ranked_products: list[str]) -> float:
    """
    Calculates the Mean Average Precision (MAP) for the ranked list.
    Returns a float between 0.0 and 1.0.
    """
    true_titles = {p.title for p in purchase_history}
    
    if not true_titles or not ranked_products:
        return 0.0

    relevant_found = 0
    running_precision_sum = 0.0

    for i, product in enumerate(ranked_products):
        rank = i + 1
        
        if product in true_titles:
            relevant_found += 1
            precision_at_k = relevant_found / rank
            running_precision_sum += precision_at_k

    avg_precision = running_precision_sum / len(true_titles)

    return float(avg_precision)

def task3_evaluator(pred_actions, ranked_products, true_personas_list, purchase_history, numq):
    e1 = calculate_persona_reward(true_personas_list, pred_actions)
    e2 = calculate_product_ranking_reward(purchase_history, ranked_products)
    e3 = (MAXQ - numq) / MAXQ

    return e1 * W1 + e2 * W2 + e3 * W3

if __name__ == "__main__":
    import json
    from pprint import pprint

    DATA = json.load(open("server/user_personas.json"))
    user = DATA[0]

    real_purchases = get_real_purchases(DATA, DATA[0]['user_id'])
    basket = make_basket(DATA, DATA[0]['user_id'])

    pprint(real_purchases)
    pprint(basket)

    basket = [p.title for p in basket]

    print(calculate_product_ranking_reward(real_purchases, basket))