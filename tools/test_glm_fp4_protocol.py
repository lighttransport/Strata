#!/usr/bin/env python3
"""CPU-only malformed wire-command checks against the shared C11 validator."""
import ctypes
import sys
import unittest
lib=ctypes.CDLL(sys.argv.pop(1))
check=lib.strata_glm_fp4_command_valid
check.argtypes=[ctypes.POINTER(ctypes.c_uint32),ctypes.c_uint32,ctypes.c_uint,ctypes.c_int]
check.restype=ctypes.c_int
class Protocol(unittest.TestCase):
    def valid(self,h,seq=7,role=1,direct=0):
        return bool(check((ctypes.c_uint32*7)(*h),seq,role,direct))
    def test_exhaustive_legal_geometry(self):
        for layer in (0,1):
            for m1 in range(17):
                for m2 in range(17):
                    h=[0x33504653,1,1,layer,m1,m2,7]
                    self.assertTrue(self.valid(h))
                    self.assertEqual(self.valid(h,direct=1),m2==0)
                    self.assertEqual(self.valid(h,role=2),m1==0)
    def test_invalid_words(self):
        h=[0x33504653,1,1,0,1,1,7]
        for index,value in [(0,0),(1,2),(2,2),(3,2),(4,17),(5,17),(6,6),(6,8)]:
            bad=h.copy();bad[index]=value;self.assertFalse(self.valid(bad))
        for index in range(7):
            bad=h.copy();bad[index]=0xffffffff;self.assertFalse(self.valid(bad))
    def test_stop_and_context(self):
        stop=[0x33504653,1,0,0,0,0,7]
        self.assertTrue(self.valid(stop))
        for i in (3,4,5):
            bad=stop.copy();bad[i]=1;self.assertFalse(self.valid(bad))
        for role in (0,3,0xffffffff):self.assertFalse(self.valid(stop,role=role))
        for direct in (-1,2):self.assertFalse(self.valid(stop,direct=direct))
        self.assertFalse(self.valid(stop,seq=0))
        self.assertEqual(check(None,7,1,0),0)
unittest.main()
