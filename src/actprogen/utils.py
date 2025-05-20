import pandas as pd

def dict_to_row(input_dict):
    expanded_dict = {}
    for key, values in input_dict.items():
        for i, value in enumerate(values):
            expanded_dict[key + str(i)] = [value]
    # Convert the new dictionary into a DataFrame
    df = pd.DataFrame.from_dict(expanded_dict)
    return df


def get_action_and_effect_dims(df):
    list_of_prefix = []
    effect_dims = []
    motion_dims = []
    counter_dims = -1
    for dim in df.columns:
        if dim.startswith('action'):
            motion_dims.append(dim)
            continue
        if dim[:3] not in list_of_prefix:
            list_of_prefix.append(dim[:3])
            effect_dims.append([dim])
            counter_dims += 1
        else:
            effect_dims[counter_dims].append(dim)
    
    return effect_dims, motion_dims