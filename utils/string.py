def dic2string(dic):
    return str(dic).replace(' ', '_').replace('{', '')\
        .replace('}', '').replace("'", '').replace(':', '').replace(',', '')\
        .replace('(', '').replace(')', '').replace('.', '-')
