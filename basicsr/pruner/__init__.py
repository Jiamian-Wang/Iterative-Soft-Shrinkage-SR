from . import l1_pruner, greg_pruner, ist_pruner, fast_ist_pruner, asslSR_pruner

# when new pruner implementation is added in the 'pruner' dir, update this dict to maintain minimal code change.
# key: pruning method name, value: the corresponding pruner
pruner_dict = {
    'L1': l1_pruner,
    'ASSL': asslSR_pruner,
    'GReg-1': greg_pruner,
    'IHT': ist_pruner,
    'IST': ist_pruner,
    'IHT-fast': fast_ist_pruner,
    'IST-fast':fast_ist_pruner,
    'IST-GReg': ist_pruner,
    'IST-GReg-fast': fast_ist_pruner,
}