from datamodels import Product, Persona, PersonaPrediction
from typing import List
import random

def get_all_personas(PDATA) -> List[Persona]:
    personas = []
    for p in PDATA:
        personas.append(Persona(name = p['name'], description=p['description'], signals=p['signals']))
    return personas

def get_personas(DATA, user_id: str) -> List[PersonaPrediction]:
    target_user = None
    for u in DATA:
        if u['user_id'] == user_id:
            target_user = u
    
    retval = []
    for p in target_user['persona']['all']:
        retval.append(PersonaPrediction(persona = p['persona'], confidence=p['confidence']))
    
    return retval

def get_real_purchases(DATA, user_id: str) -> List[Product]:
    target_user = None
    for u in DATA:
        if u['user_id'] == user_id:
            target_user = u
    
    real_purchases = [
        Product(title=item['title'],
                price = item['price'],
                description=item['description'])
        for item in target_user['purchase_history']
    ][:5]

    return real_purchases

def make_basket(DATA, user_id: str) -> List[Product]:
    """
        Takes DATA as json and userid as string... Returns a list of products
    """

    target_user = None
    other_users = []

    for u in DATA:
        if u['user_id'] == user_id:
            target_user = u
        else:
            other_users.append(u)
    
    real_purchases = get_real_purchases(DATA, user_id)

    potential_decoys: List[Product] = []
    for user in other_users:
        potential_decoys.extend([
            Product(title=item['title'],
                    price = item['price'],
                    description=item['description'])
            for item in user['purchase_history']
    ])
    
    real_titles = {p.title for p in real_purchases}
    added = set()
    filtered_decoys = []

    for d in potential_decoys:
        if d.title in real_titles: continue
        if d.title in added: continue
        filtered_decoys.append(d)
        added.add(d.title)


    num_decoys = min(len(filtered_decoys), 15)
    decoys = random.sample(filtered_decoys, num_decoys)

    basket = real_purchases + decoys
    random.shuffle(basket)

    return basket


if __name__ == "__main__":
    import json
    from pprint import pprint

    DATA = json.load(open("server/user_personas.json"))
    retval = make_basket(DATA, DATA[0]['user_id'])
    pprint(retval)

    retval = get_personas(DATA, DATA[0]['user_id'])
    pprint(retval)

    # DATA = json.load(open("server/persona_catalogue.json"))
    # retval = get_all_personas(DATA)
    # pprint(retval)