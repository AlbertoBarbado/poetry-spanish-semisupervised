import glob
import pickle
import xmltodict
import json


def get_files(PATH, extension):
    """
    # TODO
    """
    return glob.glob(PATH + "/" + "*." + extension)

def file_presistance(file_path, file_type, doc, mode):
    """
    # TODO
    """
    
    if mode == "load":
        doc = None
        if file_type == "xml":
            with open(file_path, encoding="utf-8") as fd:
                doc = xmltodict.parse(fd.read())
        elif file_type=='generic':
            with open(file_path, "rb") as fd:
                doc = pickle.load(fd)
        elif file_type=='json':
            with open(file_path, "r") as fd:
                doc = json.load(fd)
        return doc
    
    elif mode == "save":
        if file_type == "xml":
            with open(file_path, 'w', encoding="utf-8") as result_file:
                result_file.write(xmltodict.unparse(doc, pretty=True))
        elif file_type=='generic':
            with open(file_path, "wb") as fd:
                pickle.dump(doc, fd)
        elif file_type=='json':
            with open(file_path, "w") as fd:
                json.dump(doc, fd)
        return None