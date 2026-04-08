from datamodels import Product
from typing import List
import random

from pprint import pprint

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
    
    real_purchases = [
        Product(title=item['title'],
                price = item['price'],
                description=item['description'])
        for item in target_user['purchase_history']
    ][:5]

    potential_decoys: List[Product] = []
    for user in other_users:
        potential_decoys.extend([
            Product(title=item['title'],
                    price = item['price'],
                    description=item['description'])
            for item in target_user['purchase_history']
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
    DATA = json.load(open("server/user_personas.json"))
    retval = make_basket(DATA, DATA[0]['user_id'])
    pprint(retval)