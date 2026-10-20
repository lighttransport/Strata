#!/usr/bin/env python3
"""Bounded native-Q8 batch commands: shared C11 validation, no GPU or sockets."""
import ctypes
import itertools
import sys
import unittest

lib=ctypes.CDLL(sys.argv.pop(1))
check=lib.strata_glm_fp4_batch_command_valid
check.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32]
check.restype=ctypes.c_int
old=lib.strata_glm_fp4_command_valid
old.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32,ctypes.c_uint,ctypes.c_int]
old.restype=ctypes.c_int
MAGIC=0x42504653

class BatchProtocol(unittest.TestCase):
    def valid(self,h,seq=7):
        return bool(check((ctypes.c_uint32*9)(*h),seq))
    def test_all_legal_routes(self):
        for layer in (0,1):
            for count in range(4):
                for ids in itertools.product(range(3),repeat=count):
                    h=[MAGIC,1,1,layer,count,7,*ids,*([0]*(3-count))]
                    self.assertTrue(self.valid(h))
                    self.assertLessEqual(count*4096*4,49152)
    def test_bad_fields_and_unused_ids(self):
        h=[MAGIC,1,1,1,3,7,2,0,1]
        for index,value in [(0,0),(1,2),(2,2),(3,2),(4,4),(5,0),(5,8),(6,3),(7,3),(8,3)]:
            bad=h.copy();bad[index]=value;self.assertFalse(self.valid(bad))
        for index in range(9):
            bad=h.copy();bad[index]=0xffffffff;self.assertFalse(self.valid(bad))
        for count in range(3):
            for slot in range(count,3):
                bad=[MAGIC,1,1,0,count,7,0,0,0];bad[6+slot]=1
                self.assertFalse(self.valid(bad))
    def test_idle_stop_null_and_sequence(self):
        stop=[MAGIC,1,0,0,0,7,0,0,0]
        self.assertTrue(self.valid(stop))
        for index in (3,4,6,7,8):
            bad=stop.copy();bad[index]=1;self.assertFalse(self.valid(bad))
        self.assertTrue(self.valid([MAGIC,1,1,1,0,7,0,0,0]))
        self.assertFalse(self.valid(stop,0));self.assertEqual(check(None,7),0)
        stop[5]=0xffffffff;self.assertTrue(self.valid(stop,0xffffffff))
        stop[5]=0;self.assertFalse(self.valid(stop,0))
    def test_protocols_do_not_cross_accept(self):
        tree=[0x33504653,1,1,0,1,1,7]
        self.assertTrue(old((ctypes.c_uint32*7)(*tree),7,1,0))
        self.assertFalse(self.valid(tree+[0,0]))
        batch=[MAGIC,1,1,0,3,7,0,1,2]
        self.assertFalse(old((ctypes.c_uint32*9)(*batch),7,1,0))

if __name__=='__main__':unittest.main()
