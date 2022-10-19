from ast import literal_eval

def Convert2dict(config, option_name):
    config_dict = {}
    for k,v in config.items(option_name):
        try:
            config_dict[k] = literal_eval(v)
        except:
            config_dict[k] = v
    return config_dict

class Convert2object(object):
    def __init__(self, data):
        for name, value in data.items():
            setattr(self, name, self._wrap(value))

    def _wrap(self, value):
        if isinstance(value, (tuple, list, set, frozenset)): 
            return type(value)([self._wrap(v) for v in value])
        else:
            return Convert2object(value) if isinstance(value, dict) else value