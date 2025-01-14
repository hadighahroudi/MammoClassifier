import pydicom
path = r"268bbbd1932753cedf8846fe716c669b/ceb58e034ec7b2856bad9b73c98fba74.dicom"
print(pydicom.dcmread(path))
