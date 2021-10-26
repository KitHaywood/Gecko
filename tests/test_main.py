from re import L
import unittest
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
print(len(sys.path))
sys.path.append(parentdir)
print(len(sys.path))

from main import *
# Prelim Unit Testing

class TestDataType(unittest.TestCase):
    
    def test_sysArgv(self):
        self.assertIsInstance(sys.argv[1:],list)
    
    def test_sysArgv_type(self):
        self.assertTrue(all([x for x in sys.argv[1:]]))
        
    def test_get_all_coins(self):
        self.assertIsInstance(Gecko().get_all_coins(),list)


if __name__=="__main__":
    unittest.main()
