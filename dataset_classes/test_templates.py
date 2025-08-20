import unittest
from templates import (
    get_list_llama_without_answer,
    get_list_qwen_without_answer,
    get_list_phi_without_answer,
    get_list_without_answer
)

class TestTemplates(unittest.TestCase):
    
    def setUp(self):
        # Define some sample questions for testing
        self.questions = [
            "What is the capital of France?",
            "Explain quantum computing."
        ]
    
    def test_llama_equivalence(self):
        """Test if get_list_without_answer matches get_list_llama_without_answer output"""
        # Test standard prompts
        direct_output = get_list_llama_without_answer(self.questions, cot=False)
        generic_output = get_list_without_answer(self.questions, cot=False, model_short='llama')
        self.assertEqual(direct_output, generic_output)
        
        # Test CoT prompts
        direct_output_cot = get_list_llama_without_answer(self.questions, cot=True)
        generic_output_cot = get_list_without_answer(self.questions, cot=True, model_short='llama')
        self.assertEqual(direct_output_cot, generic_output_cot)
    
    def test_qwen_equivalence(self):
        """Test if get_list_without_answer matches get_list_qwen_without_answer output"""
        # Test standard prompts
        direct_output = get_list_qwen_without_answer(self.questions, cot=False)
        generic_output = get_list_without_answer(self.questions, cot=False, model_short='qwen')
        self.assertEqual(direct_output, generic_output)
        
        # Test CoT prompts
        direct_output_cot = get_list_qwen_without_answer(self.questions, cot=True)
        generic_output_cot = get_list_without_answer(self.questions, cot=True, model_short='qwen')
        self.assertEqual(direct_output_cot, generic_output_cot)
    
    def test_phi_equivalence(self):
        """Test if get_list_without_answer matches get_list_phi_without_answer output"""
        # Test standard prompts
        direct_output = get_list_phi_without_answer(self.questions, cot=False)
        generic_output = get_list_without_answer(self.questions, cot=False, model_short='phi')
        self.assertEqual(direct_output, generic_output)
        
        # Test CoT prompts
        direct_output_cot = get_list_phi_without_answer(self.questions, cot=True)
        generic_output_cot = get_list_without_answer(self.questions, cot=True, model_short='phi')
        self.assertEqual(direct_output_cot, generic_output_cot)
    
    def test_invalid_model(self):
        """Test if get_list_without_answer raises an assertion error for invalid models"""
        with self.assertRaises(AssertionError):
            get_list_without_answer(self.questions, cot=False, model_short='invalid_model')

if __name__ == '__main__':
    unittest.main()