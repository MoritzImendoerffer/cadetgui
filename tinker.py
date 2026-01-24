from CADETProcess import processModel
from CADETProcess.processModel.unitOperation import ChromatographicColumnBase

bm_dict = {}
for name in processModel.binding.__all__:
    pm = getattr(processModel, name)
    bm_dict[name] = pm.__module__ + "." + name
    
col_dict = {}
for name in processModel.unitOperation.__all__:
    uo = getattr(processModel.unitOperation, name)
    if issubclass(uo, processModel.unitOperation.ChromatographicColumnBase):
        if uo is not processModel.unitOperation.ChromatographicColumnBase:
            col_dict[name] = uo.__module__ + "." + name
            
    
print("Done")