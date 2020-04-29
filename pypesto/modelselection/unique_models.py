import itertools
import tempfile
from typing import List, Tuple

# Zero-indexed column indices
MODEL_NAME_COLUMN = 0
SBML_FILENAME_COLUMN = 1
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0
PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE_SYMBOL_UI = '-'
# here, 'nan' is a string as it will be written to a file. The actual internal
# symbol is float('nan')
ESTIMATE_SYMBOL_INTERNAL = 'nan'

#def split_file_definition(line: str) -> Tuple:
#    '''
#    Returns a 3-tuple that contains the model name, SBML file name, and all
#    possible parameter values as a list (parameters) of tuples (parameter
#    values).
#    '''
#    columns = line.strip().split('\t')
#    return (columns[MODEL_NAME_COLUMN],
#            columns[SBML_FILENAME_COLUMN],
#            [definition.split(PARAMETER_VALUE_DELIMITER)
#                for definition in columns[PARAMETER_DEFINITIONS_START:]])
#
#def generate_model_definition_lines(
#        model_name: str,
#        sbml_filename: str,
#        parameter_definitions: List[str]
#) -> str:
#    '''Yields all expanded model definitions.'''
#    for index, selection in enumerate(
#            itertools.product(*parameter_definitions)):
#        yield model_name+f'_{index}' + '\t' + sbml_filename + '\t' + \
#                '\t'.join(selection) + '\n'
#
#def unpack_file(file_name: str):
#    '''
#    Converts model definitions from the compressed form to the expanded form.
#    '''
#    expanded_models_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
#    with open(file_name) as fh:
#        for line_index, line in enumerate(fh):
#            if line_index != HEADER_ROW:
#                for definition_line in generate_model_definition_lines(
#                        *split_file_definition(line)):
#                    expanded_models_file.write(definition_line)
#            else:
#                expanded_models_file.write(line)
#    return expanded_models_file

def _replace_estimate_symbol(parameter_definition) -> List:
    return [ESTIMATE_SYMBOL_INTERNAL if p == ESTIMATE_SYMBOL_UI else p
            for p in parameter_definition]

def unpack_file(file_name: str):
    '''
    Unpacks a model definition file into a new temporary file that is returned.

    TODO
        - Consider alternatives to `_{n}` suffix for model `modelId`
        - How should the selected model be reported to the user? Remove the
          `_{n}` suffix and report the original `modelId` alongside the
          selected parameters? Generate a set of PEtab files with the chosen
          SBML file and the parameters specified in a parameter or condition
          file?
    '''
    expanded_models_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
    with open(file_name) as fh:
        for line_index, line in enumerate(fh):
            if line_index != HEADER_ROW:
                columns = line.strip().split('\t')
                parameter_definitions = [
                    _replace_estimate_symbol(
                        definition.split(PARAMETER_VALUE_DELIMITER)
                    ) for definition in columns[PARAMETER_DEFINITIONS_START:]
                ]
                for index, selection in enumerate(itertools.product(
                        *parameter_definitions
                )):
                    expanded_models_file.write(
                        '\t'.join([
                            columns[MODEL_NAME_COLUMN]+f'_{index}',
                            columns[SBML_FILENAME_COLUMN],
                            *selection
                        ]) + '\n'
                    )
            else:
                expanded_models_file.write(line)
    return expanded_models_file

# write function to return unpacked file line by line, for use in a model
# selector class (e.g. ForwardSelector). The line should be parsed, and a
# dictionary returned with keys 'model_name', 'sbml_filename',
# 'parameter_values' (list of parameter values), 'parameter_ids' (list of
# parameter ids). The parameter values are returned as a list so they can be
# easily compared with the current selected model e.g. to find models that
# represent the "next complexity" in forward selection, you could check that
# the old list of parameter values, and the new list, are zero at the same
# indices except one (and that indice is only zero on the old parameter values
# list.


original_file = 'example_model_selection_definitions.tsv'
unpacked_file = unpack_file(original_file)

# The file at expanded_models_file.name now contains the expanded model.
print('The expanded model space definitions have been stored in: ' + \
        unpacked_file.name)
