def dic2string(dic):
    return str(dic).replace(' ', '_').replace('{', '')\
        .replace('}', '').replace("'", '').replace(':', '').replace(',', '')\
        .replace('(', '').replace(')', '').replace('.', '-')


def param_string(param_labels, param, first_letter_only=False):
    return '_'.join(f'{k[0] if first_letter_only else k}={v:.2f}'
                    for (k, v) in zip(param_labels, param))