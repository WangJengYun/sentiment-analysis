import copy 

class MetricRecord(object):
    def __init__(self, *keys):
        
        self._empty_dict = {'epoch':0, 'counts':0, 'historical_mertic':[], 'final_mertic':0}
        self.current_metric = {}
        self._metrtic_data = dict(zip(keys,[copy.deepcopy(self._empty_dict)]*len(keys)))

    def reset(self, key = None):
        self.current_metric = {}
        if key is not None:
            self._metrtic_data[key] = copy.deepcopy(self._empty_dict)
        else:
            for key in self._metrtic_data.keys():
                self._metrtic_data[key] = copy.deepcopy(self._empty_dict)
    
    def update(self,epoch, key, value, iscusum = True):
        self.current_metric[key] = value
        self._metrtic_data[key]['epoch'] = epoch

        if iscusum:
            self._metrtic_data[key]['counts'] += 1
            self._metrtic_data[key]['historical_mertic'].append(value)            
            self._metrtic_data[key]['final_mertic'] = sum(self._metrtic_data[key]['historical_mertic']) / self._metrtic_data[key]['counts']
        else:
            self._metrtic_data[key]['counts'] = 1
            self._metrtic_data[key]['historical_mertic'].append(value)
            self._metrtic_data[key]['final_mertic'] = value

    def get_final_mertic(self, key):
        return self._metrtic_data[key]['final_mertic']
    
    def current_performance(self):
        return self.current_metric

    def performance(self):
        _performance = {}
        for key in self._metrtic_data.keys():
            _performance[key] =  self._metrtic_data[key]['final_mertic']
        return _performance

if __name__ == '__main__':
    pass