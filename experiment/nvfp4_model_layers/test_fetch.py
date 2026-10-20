"""The downloader must reject ignored ranges before reading a shard body."""
import unittest
from unittest.mock import patch
import fetch

class Response:
    def __init__(self, status, content_range, data):
        self.status=status; self.headers={'Content-Range':content_range}; self.data=data; self.reads=[]
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def read(self,n): self.reads.append(n); return self.data[:n]

class RangeTests(unittest.TestCase):
    def check_response(self,r,ok=False):
        with patch('urllib.request.urlopen',return_value=r):
            if ok: self.assertEqual(fetch.get('fake',8,4),b'abcd')
            else:
                with self.assertRaises(RuntimeError): fetch.get('fake',8,4)
    def test_exact_range(self): self.check_response(Response(206,'bytes 8-11/100',b'abcd'),True)
    def test_full_shard_rejected_without_read(self):
        r=Response(200,'',b'abcd');self.check_response(r);self.assertEqual(r.reads,[])
    def test_wrong_offset_rejected_without_read(self):
        r=Response(206,'bytes 0-3/100',b'abcd');self.check_response(r);self.assertEqual(r.reads,[])
    def test_truncated(self): self.check_response(Response(206,'bytes 8-11/100',b'abc'))
    def test_oversized_body(self): self.check_response(Response(206,'bytes 8-11/100',b'abcde'))

if __name__=='__main__': unittest.main()
