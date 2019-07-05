import os
import pickle


def var_to_pickle(var, filename):
    '''
    Writes the given variable to a pickle file

    Args:
        var (any): variable to be written to pickle file
        filename (str): path and filename of pickle file

    Returns:
        None
    '''
    try:
        with open(filename, 'wb') as f:
            pickle.dump(var, f)
    except:
        print(f'Failed to save pickle to \'{filename}\'')
    return

def read_pickle(filename):
    '''
    Reads the given pickle file

    Args:
        filename (str): path and filename of pickle file

    Returns:
        any: contents of pickle file if it exists, None if not
    '''
    output = None
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                output = pickle.load(f)
        except:
            print(f'Failed to load pickle from \'{filename}\'')
    return output
