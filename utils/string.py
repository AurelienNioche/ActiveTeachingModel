def dic2string(dic):
    return str(dic).replace(' ', '_').replace('{', '')\
        .replace('}', '').replace("'", '').replace(':', '').replace(',', '')\
        .replace('(', '').replace(')', '').replace('.', '-')


def param_string(param_labels, param):
    return '_'.join(f'{k}={v:.2f}'
                    for (k, v) in zip(param_labels, param))