import pickle


def extract_data_from_pkl(file_path):
    """
    Deserializes (extracts) the Python object(s) from a .pkl file.
    """
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while extracting data: {e}")
        return None


file_to_extract = 'c041/c041_dets_feat.pkl'
extracted_object = extract_data_from_pkl(file_to_extract)

if extracted_object is not None:
    print("Data extracted successfully:")
    # print(extracted_object)
    if isinstance(extracted_object, dict):
        # print first element of the dictionary
        first_key = next(iter(extracted_object))
        print(f"First key in the dictionary: {first_key}")
        print(f"First value in the dictionary: {extracted_object[first_key]}")
        print(f"The feat element of the first value: {extracted_object[first_key]['feat']}")
else:
    print("Failed to extract data.")
