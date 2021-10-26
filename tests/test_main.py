import unittest
import Gecko.main
# Prelim Unit Testing

class TestDataType(unittest.TestCase):
    
    def test_sysArgv(self):
        self.assertIsInstance(sys.argv[1:],list)
    
    def test_sysArgv_type(self):
        self.assertTrue(all([x for x in sys.argv[1:]]))
        
if __name__=="__main__":
    unittest.main()
