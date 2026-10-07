"""Explicit JSON encoding for nonfinite scheduler metadata only."""
import math

def encode_nonfinite(value, path='$'):
    """Return a JSON-safe metadata tree with explicit non-finite markers."""
    changes=[]
    if isinstance(value,float) and not math.isfinite(value):
        label='NaN' if math.isnan(value) else ('+Infinity' if value>0 else '-Infinity')
        return {'__nonfinite_float__':label},[{'path':path,'value':label}]
    if isinstance(value,dict):
        out={}
        for k,v in value.items():
            out[k],c=encode_nonfinite(v,path+'.'+str(k));changes.extend(c)
        return out,changes
    if isinstance(value,(tuple,list)):
        out=[]
        for i,v in enumerate(value):
            item,c=encode_nonfinite(v,path+'['+str(i)+']');out.append(item);changes.extend(c)
        return out,changes
    return value,changes
