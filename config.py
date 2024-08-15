

# real ordinal classification data
# ["datasets/tae.data", "tae", [[1,2]]] , ["datasets/balance-scale.data", "balance-scale", [[1,2]]],
# ["datasets/new-thyroid.csv", "thyroid", [[1,2]]], ["datasets/SWD.csv", "SWD", [[2,3,4]]],
# ["datasets/dataset_194_eucalyptus.csv", "eucalyptus", [[1,2,3,4]]], ["datasets/LEV.csv", "LEV", [[0,1,2,3]]],
# ["datasets/winequality-red.csv","wine_full",[[3,4,5,6,7]]]
# ["datasets/ERA.csv", "ERA", [[1,2,3,4,5,6,7,8]]]
# ["datasets/ESL.csv","ESL", [[3,4,5,6,7]]]
# ["datasets/car-evaluation.csv","car", [[0,1,2,3]]]

# regression data
# ["datasets/abalone.data", "abalone5", [[5, 8, 11, 14]]],
# ["datasets/ailerons.csv", "ailerons5", [[-0.0020, -0.0015, -0.0010, -0.0005]]],
# ["datasets/bank.csv", "bank5", [[0.1, 0.2, 0.3, 0.5]]],
# ["datasets/dataset_2202_elevators.csv", "elevators5", [[0.02, 0.025, 0.03, 0.04]]],
# ["datasets/pumadyn.csv", "pumadyn5", [[-0.05, -0.025, 0.025, 0.05]]],
# ["datasets/census.csv", "census5", [[25000, 50000, 75000, 100000]]],
# ["datasets/california.csv", "california5", [[100000, 200000, 300000, 400000]]],
#
# ["datasets/abalone.data", "abalone10", [[5, 6, 7, 8, 9, 10, 11, 12, 13]]],
# ["datasets/ailerons.csv", "ailerons10",
#  [[-0.0020, -0.00175, -0.0015, -0.00125, -0.00115, -0.0010, -0.00075, -0.0006, -0.0005]]],
# ["datasets/bank.csv", "bank10", [[0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]]],
# ["datasets/dataset_2202_elevators.csv", "elevators10",
#  [[0.017, 0.019, 0.021, 0.023, 0.025, 0.0275, 0.03, 0.035, 0.04]]],
# ["datasets/pumadyn.csv", "pumadyn10", [[-0.05, -0.0375, -0.025, -0.0175, 0, 0.0175, 0.025, 0.0375, 0.05]]],
# ["datasets/census.csv", "census10", [[25000, 37500, 50000, 62500, 75000, 87500, 100000, 150000, 200000]]],
# ["datasets/california.csv", "california10",
#  [[70000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 450000]]],
# ["datasets/HousingData.csv", "boston5", [[10, 20, 30, 40]]],
# ["datasets/HousingData.csv", "boston10", [[10, 15, 20, 23, 25, 30, 35, 40, 45]]],
# ["datasets/dataset_2209_stock.csv", "stock5", [[40, 45, 50, 55]]],
# ["datasets/dataset_2209_stock.csv", "stock10", [[37.5, 40, 42.5, 45, 47.5, 50, 52.5, 55, 57.5]]]
    # ["datasets/abalone.data", "abalone7", [[5, 7, 9, 11, 12, 13]]],
    # ["datasets/ailerons.csv", "ailerons7",
    #  [[-0.0020, -0.0016, -0.0013, -0.0010, -0.00075, -0.0005]]],
    # ["datasets/bank.csv", "bank7", [[0.02, 0.08, 0.15, 0.2, 0.3, 0.5]]],
    # ["datasets/dataset_2202_elevators.csv", "elevators7",
    #  [[0.017, 0.021, 0.025, 0.03, 0.035, 0.04]]],
    # ["datasets/pumadyn.csv", "pumadyn7", [[-0.05, -0.025, 0, 0.025, 0.0375, 0.05]]],
    # ["datasets/census.csv", "census7", [[25000, 50000, 75000, 100000, 150000, 200000]]],
    # ["datasets/california.csv", "california7",
    #  [[70000, 150000, 200000, 300000, 400000, 450000]]],
    # ["datasets/HousingData.csv", "boston7", [[10, 20, 25, 30, 40, 45]]],
    # ["datasets/dataset_2209_stock.csv", "stock7", [[37.5, 42.5, 47.5, 50, 55, 57.5]]]

DATA = [
    ["datasets/abalone.data", "abalone7", [[5, 7, 9, 11, 12, 13]]],
    ["datasets/ailerons.csv", "ailerons7",
     [[-0.0020, -0.0016, -0.0013, -0.0010, -0.00075, -0.0005]]],
    ["datasets/bank.csv", "bank7", [[0.02, 0.08, 0.15, 0.2, 0.3, 0.5]]],
    ["datasets/dataset_2202_elevators.csv", "elevators7",
     [[0.017, 0.021, 0.025, 0.03, 0.035, 0.04]]],
    ["datasets/pumadyn.csv", "pumadyn7", [[-0.05, -0.025, 0, 0.025, 0.0375, 0.05]]],
    ["datasets/census.csv", "census7", [[25000, 50000, 75000, 100000, 150000, 200000]]],
    ["datasets/california.csv", "california7",
     [[70000, 150000, 200000, 300000, 400000, 450000]]],
    ["datasets/HousingData.csv", "boston7", [[10, 20, 25, 30, 40, 45]]],
    ["datasets/dataset_2209_stock.csv", "stock7", [[37.5, 42.5, 47.5, 50, 55, 57.5]]]
]

# "Cross_Entropy","SORD","OLL","Wocel","Accumulating"
# ALPHA = [0.3, 0.5, 0.7, 0.8, 0.9, 1, 2, 3, 4]
#  7, 10, 15, 20, 25
# types = ["max", "norm_max", "norm_log", "log", "norm_division", "division"]
# base = ["Cross_Entropy", "SORD", "OLL", "Wocel", "Accumulating", "Focal_loss", "Accumulating_SORD"]
# LOSS = [["SORD_" + type, "OLL_" + type, "Accumulating_SORD_prox_" + type] for type in types]
# LOSS = [item for sublist in LOSS for item in sublist]
# LOSS = [item for sublist in [LOSS, base] for item in sublist]
LOSS = ["SORD","OLL","Cross_Entropy","Accumulating_SORD_prox_max"]

ALPHA = [0.3, 0.5, 0.8, 1, 2, 3, 4, 7, 10, 15, 20, 25]

RANGE_OF_SEEDS = [0, 10]
PARAMS = {'colsample_bytree': 0.5, 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 100}
