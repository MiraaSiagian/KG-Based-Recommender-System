import logging
import time
from copy import deepcopy
import math
import numpy as np
import torch
from pytorch_lightning import seed_everything
from sklearn.metrics import roc_auc_score
import os
import sys
#os.path.join(..., '..') -> go up one level to (.../KG-BASED_RECOMM...)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
 '..')))
from demo.kgrs import KGRS

def nDCG(sorted_items, pos_item, train_pos_item, k=5):
    dcg = 0
    train_pos_item = set(train_pos_item)
    filter_item = set(filter(lambda item: item not in train_pos_item,
        pos_item))
    max_correct = min(len(filter_item), k)
    train_hit_num = 0
    valid_num = 0
    recommended_items = set()
    for index in range(len(sorted_items)):
        if sorted_items [index] in train_pos_item:
            train_hit_num += 1
        else:
            valid_num += 1
            if sorted_items[index] in filter_item and \
                sorted_items[index] not in recommended_items:
                # Rank starts from 0
                dcg += 1 / math.log2(index - train_hit_num + 2)
                recommended_items.add(sorted_items[index])
        if valid_num >= k:
            break
    idcg = sum([1/math.log2(i + 2) for i in range(max_correct)])
    return dcg / idcg if idcg > 0 else 0 # Handle idcg = 0 case

def load_data():
    # Path sudah diperbaiki untuk dijalankan dari root folder
    train_pos, train_neg = np.load("data/train_pos.npy"), np.load(
        "data/train_neg.npy")
    
    np.random.shuffle(train_pos)
    np.random.shuffle(train_neg)
    all_users = set(train_pos[:, 0]) | set (train_neg[:, 0])
    all_items = set(train_pos[:, 1]) | set (train_neg[:, 1])
    n_user = max(all_users) + 1
    n_item = max(all_items) + 1
    train_pos_len, train_neg_len = int(len(train_pos)*0.8), \
        int(len(train_neg)*0.8)
    test_pos, test_neg = train_pos [train_pos_len:], \
        train_neg[train_neg_len:]
    train_pos, train_neg = train_pos[:train_pos_len], \
        train_neg[:train_neg_len]
    return train_pos, train_neg, test_pos, test_neg, n_user, n_item

def get_user_pos_items (train_pos, test_pos):
    user_pos_items, user_train_pos_items = {}, {}
    for record in train_pos:
        user, item = record [0], record [1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        user_train_pos_items [user].add(item)
    for record in test_pos:
        user, item = record [0], record [1]
        if user not in user_train_pos_items:
            user_train_pos_items[user] = set()
        if user not in user_pos_items:
            user_pos_items [user] = set()
        user_pos_items[user].add(item)
    return user_pos_items, user_train_pos_items

def evaluate(config_run: dict):
    train_pos, train_neg, test_pos, test_neg, n_user, n_item = \
        load_data()
    user_pos_items, user_train_pos_items = \
        get_user_pos_items (train_pos=train_pos, test_pos=test_pos)
    logging.disable (logging.INFO)
    torch.set_num_threads (8)
    auc, ndcg5 = 0, 0
    init_timeout, train_timeout, ctr_timeout, topk_timeout = False, \
        False, False, False
    start_time, init_time, train_time, ctr_time, topk_time = \
        time.time(), 0, 0, 0, 0
    
    # Path sudah diperbaiki untuk dijalankan dari root folder
    kgrs = KGRS(train_pos=deepcopy (train_pos),
        train_neg=deepcopy (train_neg),
        kg_lines=open('data/kg.txt',
            encoding='utf-8').readlines(),
        n_user=n_user,
        n_item=n_item,
        config=config_run)
    
    init_time = time.time() - start_time
    kgrs.training()
    train_time = time.time() - start_time - init_time
    test_data = np.concatenate ((deepcopy (test_neg), deepcopy(test_pos)),
        axis=0)
    np.random.shuffle(test_data)
    test_label = test_data[:, 2]
    test_data = test_data[:,:2]
    scores = kgrs.eval_ctr(test_data=test_data)
    auc = roc_auc_score(y_true=test_label, y_score=scores)
    ctr_time = time.time() - start_time - init_time - train_time
    users = list(user_pos_items.keys())
    user_item_lists = kgrs.eval_topk(users=users)
    ndcg5 = np.mean([nDCG(user_item_lists [index], user_pos_items [user],
        user_train_pos_items[user]) for index, user in
        enumerate (users)])
    topk_time = time.time() - start_time - ctr_time - init_time - train_time
    return auc, ndcg5, init_timeout, train_timeout, ctr_timeout, \
        topk_timeout, init_time, train_time, ctr_time, topk_time

if __name__ == '__main__':
    seed_everything(1088, workers=True)
    
    # --- INI BAGIAN GRID SEARCH ---
    
    # 1. Tentukan parameter yang akan diuji
    #    (Anda bisa menambah/mengubah daftar ini)
    param_grid_emb_dim = [16, 32]
    param_grid_lr = [2e-3, 1e-3]
    param_grid_margin = [10, 30]

    best_score_auc = 0
    best_score_ndcg = 0
    best_params_auc = {}
    best_params_ndcg = {}
    
    print("--- MEMULAI GRID SEARCH ---")
    print(f"Akan menjalankan {len(param_grid_emb_dim) * len(param_grid_lr) * len(param_grid_margin)} percobaan...")

    # 2. Buat loop
    for emb_dim in param_grid_emb_dim:
        for lr in param_grid_lr:
            for margin in param_grid_margin:
                
                print(f"\n=============================================")
                print(f"MENJALANKAN: emb_dim={emb_dim}, lr={lr}, margin={margin}")
                print(f"=============================================")

                # 3. Buat config untuk percobaan ini
                current_config = {
                    "batch_size": 256, "eval_batch_size": 1024,
                    "emb_dim": emb_dim,      # <-- Gunakan variabel loop
                    "l1": True, 
                    "margin": margin,      # <-- Gunakan variabel loop
                    "learning_rate": lr,   # <-- Gunakan variabel loop
                    "weight_decay": 5e-4,
                    "neg_rate": 1.5,
                    "epoch_num": 35      # Kita jaga epoch tetap 35 untuk perbandingan
                }
                
                # 4. Jalankan evaluasi dengan config ini
                start_run_time = time.time()
                auc, ndcg5, init_timeout, train_timeout, ctr_timeout, topk_timeout, \
                init_time, train_time, ctr_time, topk_time = evaluate(config_run=current_config)
                
                # 5. Tampilkan hasil untuk percobaan ini
                print(f"--- Result (Percobaan Ini) ---")
                print(f"AUC Score : {auc:.4f}")
                print(f"nDCG@5 Score : {ndcg5:.4f}")
                print(f"Waktu Eksekusi: {(time.time() - start_run_time):.2f} detik")
                
                # 6. Simpan hasil terbaik (berdasarkan nDCG@5)
                if ndcg5 > best_score_ndcg:
                    best_score_ndcg = ndcg5
                    best_params_ndcg = {"emb_dim": emb_dim, "lr": lr, "margin": margin, "auc": auc}
                
                # 7. Simpan hasil terbaik (berdasarkan AUC)
                if auc > best_score_auc:
                    best_score_auc = auc
                    best_params_auc = {"emb_dim": emb_dim, "lr": lr, "margin": margin, "ndcg": ndcg5}

    # 8. Tampilkan hasil terbaik setelah semua loop selesai
    print("\n\n--- GRID SEARCH SELESAI ---")
    print(f"Skor AUC Terbaik: {best_score_auc:.4f}")
    print(f"Parameter Terbaik (untuk AUC): {best_params_auc}")
    print("----")
    print(f"Skor nDCG@5 Terbaik: {best_score_ndcg:.4f}")
    print(f"Parameter Terbaik (untuk nDCG@5): {best_params_ndcg}")