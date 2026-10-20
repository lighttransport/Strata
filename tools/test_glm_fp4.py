"""CPU tests for the shared CSF contract; pass the compiled shared library."""
import ctypes as C
import random
import struct
import sys
import unittest

lib = C.CDLL(sys.argv.pop(1))
u8 = C.POINTER(C.c_uint8)
lib.strata_glm_csf32_decode.argtypes = [C.c_size_t, C.c_size_t] + [x for _ in range(6) for x in (u8,C.c_size_t)]
lib.strata_glm_csf32_decode.restype = C.c_int

def fixture(rows, cols):
    rng=random.Random(rows*1000+cols)
    bases=bytes(rng.randrange(100) for _ in range(rows))
    fields=[[rng.randrange(16) for _ in range(cols)] for _ in range(rows)]
    expected=bytearray(bases[r]+fields[r][c] for r in range(rows) for c in range(cols))
    exceptions={}
    for r in range(rows):
        for c in range(cols):
            if rng.randrange(17)==0:
                exceptions[r,c]=rng.randrange(127);expected[r*cols+c]=exceptions[r,c]
    # Sort canonical 32x2 logical blocks by their native tile and column-pair order.
    blocks=sorted([(r,c) for r in range(0,rows,32) for c in range(0,cols,2)],
                  key=lambda rc:(rc[0]//128,rc[1]//4,rc[1]%4,rc[0]%128))
    codes=bytearray();records=[];pages=[];offsets=[];last_page=-1;page_start=0
    for r,c in blocks:
        tile=(r//128)*((cols+3)//4)+c//4;page=tile//32
        if page!=last_page:
            if last_page>=0:offsets.append(len(records)-page_start)
            pages.append(len(records)*2);page_start=len(records);last_page=page
        offsets.append(len(records)-page_start)
        values=[fields[rr][cc] for rr in range(r,r+32) for cc in range(c,min(c+2,cols))]
        codes.extend(values[i]|values[i+1]<<4 for i in range(0,len(values),2))
        records.extend((rr-r)*2+(cc-c)+(exceptions[rr,cc]<<8)
                       for rr in range(r,r+32) for cc in range(c,min(c+2,cols)) if (rr,cc) in exceptions)
    offsets.append(len(records)-page_start)
    pack=lambda code,x:struct.pack('<'+code*len(x),*x)
    return [bases,bytes(codes),pack('I',pages),pack('H',offsets),pack('H',records)],bytes(expected)

def decode(rows,cols,parts):
    arrays=[(C.c_uint8*len(p)).from_buffer_copy(p) for p in parts]
    out=(C.c_uint8*(rows*cols))(*([0xcd]*(rows*cols)))
    args=[]
    for a in arrays:args.extend([a,len(a)])
    return lib.strata_glm_csf32_decode(rows,cols,*args,out,len(out)),bytes(out)

class CsfTests(unittest.TestCase):
    def test_geometries(self):
        for rows,cols in [(32,1),(32,3),(96,5),(128,4),(160,9),(256,129),(2048,256),(4096,128)]:
            with self.subTest(rows=rows,cols=cols):
                parts,want=fixture(rows,cols);rc,got=decode(rows,cols,parts)
                self.assertEqual(rc,0);self.assertEqual(got,want)
    def test_truncation_no_write(self):
        parts,_=fixture(128,4)
        for i in range(5):
            bad=parts.copy();bad[i]=bad[i][:-1];rc,got=decode(128,4,bad)
            self.assertNotEqual(rc,0);self.assertEqual(got,bytes([0xcd])*512)
    def test_invalid_base(self):
        parts,_=fixture(128,4);parts[0]=b'\xff'+parts[0][1:]
        self.assertNotEqual(decode(128,4,parts)[0],0)
    def test_invalid_page(self):
        parts,_=fixture(128,4);parts[2]=struct.pack('<I',2)
        self.assertNotEqual(decode(128,4,parts)[0],0)
    def test_invalid_exception(self):
        parts,_=fixture(128,4);parts[4]=b'\xff'+parts[4][1:]
        self.assertNotEqual(decode(128,4,parts)[0],0)
    def test_bad_terminal(self):
        parts,_=fixture(128,4);parts[3]=parts[3][:-2]+b'\xff\xff'
        self.assertNotEqual(decode(128,4,parts)[0],0)

lib.strata_glm_fp4_repack_k32.argtypes=[u8,u8,C.c_size_t,C.c_int]
lib.strata_glm_fp4_repack_k32.restype=C.c_int

class PackK32(unittest.TestCase):
    def test_roundtrip_and_layout(self):
        rng=random.Random(37)
        for n in (16,32,4096):
            src=bytes(rng.randrange(256) for _ in range(n))
            a=(C.c_uint8*n).from_buffer_copy(src);b=(C.c_uint8*n)();c=(C.c_uint8*n)()
            self.assertEqual(lib.strata_glm_fp4_repack_k32(a,b,n,0),0)
            for offset in range(0,n,16):
                values=[(src[offset+j//2]>>((j%2)*4))&15 for j in range(32)]
                self.assertEqual(bytes(b)[offset:offset+16],bytes(values[j]|values[j+16]<<4 for j in range(16)))
            self.assertEqual(lib.strata_glm_fp4_repack_k32(b,c,n,1),0)
            self.assertEqual(bytes(c),src)
            self.assertEqual(lib.strata_glm_fp4_repack_k32(a,a,n,0),0)
            self.assertEqual(bytes(a),bytes(b))
            self.assertEqual(lib.strata_glm_fp4_repack_k32(a,a,n,1),0)
            self.assertEqual(bytes(a),src)
    def test_rejection_before_write(self):
        a=(C.c_uint8*64)(*range(64));before=bytes(a)
        for n,inverse in ((15,0),(16,2)):
            self.assertNotEqual(lib.strata_glm_fp4_repack_k32(a,a,n,inverse),0)
            self.assertEqual(bytes(a),before)
        ptr=C.cast(C.byref(a,1),u8)
        self.assertNotEqual(lib.strata_glm_fp4_repack_k32(a,ptr,16,0),0)
        self.assertNotEqual(lib.strata_glm_fp4_repack_k32(ptr,a,16,1),0)
        self.assertEqual(bytes(a),before)
        self.assertNotEqual(lib.strata_glm_fp4_repack_k32(None,a,16,0),0)
        self.assertEqual(lib.strata_glm_fp4_repack_k32(None,None,0,0),0)

if __name__=='__main__':unittest.main()
