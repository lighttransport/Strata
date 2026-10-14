"""Optional binding to the worker's C11 admission rule; no Python rule copy."""
import ctypes as c


class CacheAdmissionPolicy:
    def __init__(self, library):
        self.library=c.CDLL(str(library))
        self.select=self.library.strata_mixed_select_cache
        self.select.argtypes=[c.c_uint32,c.c_uint32,c.POINTER(c.c_int),c.POINTER(c.c_double),
                              c.POINTER(c.c_float),c.POINTER(c.c_uint8),c.POINTER(c.c_uint8),
                              c.c_uint32,c.c_float,c.c_double,c.POINTER(c.c_uint8)]
        self.select.restype=c.c_int

    def apply(self, scores, counts, previous, primary, capacity,
              minimum_routes=0, minimum_gain_percent=0, max_replacements=4, order=None):
        n=len(scores)
        if not 1<=n<=288 or len(counts)!=n or len(previous)!=n or len(primary)!=n:
            raise ValueError('inconsistent cache policy geometry')
        if not 0<=capacity<=n or not 0<=max_replacements<=288:
            raise ValueError('invalid cache capacity/replacement count')
        if any(x not in (0,1) for x in [*previous,*primary]):
            raise ValueError('ownership masks must be binary')
        order=sorted(range(n),key=lambda e:(-scores[e],e)) if order is None else order
        if len(order)!=n:raise ValueError('inconsistent ranking length')
        order=(c.c_int*n)(*order)
        output=(c.c_uint8*n)()
        ok=self.select(n,capacity,order,(c.c_double*n)(*scores),(c.c_float*n)(*counts),
                       (c.c_uint8*n)(*previous),(c.c_uint8*n)(*primary),max_replacements,
                       minimum_routes,minimum_gain_percent,output)
        if not ok:
            raise ValueError('C11 cache admission rejected invalid input')
        return list(output)
