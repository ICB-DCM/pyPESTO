MODEL_ID = 'modelId'
COMPARED_MODEL_ID = 'compared_'+MODEL_ID
YAML_FILENAME = 'YAML'
NOT_PARAMETERS = [MODEL_ID, YAML_FILENAME]

# Zero-indexed column/row indices
MODEL_ID_COLUMN = 0
YAML_FILENAME_COLUMN = 1
# It is assumed that all columns after PARAMETER_DEFINITIONS_START contain
# parameter IDs.
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0

PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE_SYMBOL_UI = '-'
# Here, 'nan' is a string as it will be written to a (temporary) file. The
# actual internal symbol is float('nan'). Equality to this symbol should be
# checked with a function like `math.isnan()` (not ` == float('nan')`).
ESTIMATE_SYMBOL_INTERNAL = 'nan'
INITIAL_VIRTUAL_MODEL = 'PYPESTO_INITIAL_MODEL'
