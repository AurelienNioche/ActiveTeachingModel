def to_key_val_list(dic):
    lab = list(dic.keys())
    val = [dic[k] for k in lab]
    return lab, val
