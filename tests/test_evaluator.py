import pytest
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from evaluator import DynamicEvaluator

@pytest.fixture
def evaluator():
    return DynamicEvaluator()

def test_min_words(evaluator):
    text = "Hello world this is a test"
    # 6 words
    constraints_pass = {"min_words": 5}
    constraints_fail = {"min_words": 10}
    
    assert evaluator.evaluate(text, constraints_pass)["details"]["min_words"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["min_words"] == False

def test_max_words(evaluator):
    text = "Hello world this is a test"
    # 6 words
    constraints_pass = {"max_words": 10}
    constraints_fail = {"max_words": 5}
    
    assert evaluator.evaluate(text, constraints_pass)["details"]["max_words"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["max_words"] == False

def test_exact_lines(evaluator):
    text = "Line 1\nLine 2\n\nLine 3"
    # 3 non-empty lines
    constraints_pass = {"exact_lines": 3}
    constraints_fail = {"exact_lines": 2}

    assert evaluator.evaluate(text, constraints_pass)["details"]["exact_lines"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["exact_lines"] == False

def test_mandatory_words(evaluator):
    text = "The quick brown fox jumps over the lazy dog"
    
    constraints_pass = {"mandatory_words": ["quick", "fox", "dog"]}
    constraints_fail = {"mandatory_words": ["cat"]}

    assert evaluator.evaluate(text, constraints_pass)["details"]["mandatory_words"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["mandatory_words"] == False

def test_forbidden_words(evaluator):
    text = "Perfect quality and amazing speed"
    
    constraints_pass = {"forbidden_words": ["bad", "terrible", "slow"]}
    constraints_fail = {"forbidden_words": ["amazing"]}

    assert evaluator.evaluate(text, constraints_pass)["details"]["forbidden_words"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["forbidden_words"] == False

def test_json_format(evaluator):
    text_with_markdown = "```json\n{\"key\": \"value\"}\n```"
    text_invalid_json = "{\"key\": \"value\", }"
    
    constraints = {"format": "json"}

    assert evaluator.evaluate(text_with_markdown, constraints)["details"]["format_json"] == True
    assert evaluator.evaluate(text_invalid_json, constraints)["details"]["format_json"] == False

def test_required_json_keys(evaluator):
    text = "{\"name\": \"AI\", \"version\": 3}"
    
    constraints_pass = {"format": "json", "required_json_keys": ["name", "version"]}
    constraints_fail = {"format": "json", "required_json_keys": ["author"]}

    assert evaluator.evaluate(text, constraints_pass)["details"]["required_json_keys"] == True
    assert evaluator.evaluate(text, constraints_fail)["details"]["required_json_keys"] == False
