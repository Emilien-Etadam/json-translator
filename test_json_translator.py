#!/usr/bin/env python3
"""
Unit tests for JSON Translator

Run with: python3 -m pytest test_json_translator.py -v
Or: python3 test_json_translator.py
"""

import json
import unittest
from json_translator import (
    TranslationCache,
    VariablePreserver,
    CaseStyleHelper,
    TranslationQualityValidator,
    JSONKeySelector
)


class TestTranslationCache(unittest.TestCase):
    """Test TranslationCache functionality"""

    def test_cache_set_and_get(self):
        cache = TranslationCache(max_size=3)
        cache.set("hello", "fr", "mistral", "bonjour")
        result = cache.get("hello", "fr", "mistral")
        self.assertEqual(result, "bonjour")
        self.assertEqual(cache.stats["hits"], 1)
        self.assertEqual(cache.stats["misses"], 0)

    def test_cache_miss(self):
        cache = TranslationCache()
        result = cache.get("nonexistent", "fr", "mistral")
        self.assertIsNone(result)
        self.assertEqual(cache.stats["misses"], 1)

    def test_cache_eviction(self):
        cache = TranslationCache(max_size=2)
        cache.set("one", "fr", "mistral", "un")
        cache.set("two", "fr", "mistral", "deux")
        cache.set("three", "fr", "mistral", "trois")  # Should evict "one"

        self.assertEqual(len(cache.cache), 2)
        self.assertIsNone(cache.get("one", "fr", "mistral"))
        self.assertEqual(cache.get("two", "fr", "mistral"), "deux")
        self.assertEqual(cache.stats["evictions"], 1)

    def test_cache_clear(self):
        cache = TranslationCache()
        cache.set("hello", "fr", "mistral", "bonjour")
        cache.clear()
        self.assertEqual(len(cache.cache), 0)
        self.assertEqual(cache.stats["hits"], 0)


class TestVariablePreserver(unittest.TestCase):
    """Test VariablePreserver functionality"""

    def test_find_double_brace_variables(self):
        preserver = VariablePreserver()
        text = "Hello {{name}}, welcome to {{place}}!"
        variables = preserver.find_variables(text)
        self.assertEqual(len(variables), 2)
        self.assertEqual(variables[0][0], "{{name}}")
        self.assertEqual(variables[1][0], "{{place}}")

    def test_protect_and_restore(self):
        preserver = VariablePreserver()
        text = "Hello {{name}}, you have {count} messages"
        protected, vars_list = preserver.protect(text)

        # Check protection
        self.assertIn("__VAR_0__", protected)
        self.assertIn("__VAR_1__", protected)
        self.assertNotIn("{{name}}", protected)
        self.assertEqual(len(vars_list), 2)

        # Check restoration
        restored = preserver.restore(protected, vars_list)
        self.assertEqual(restored, text)

    def test_protect_empty_text(self):
        preserver = VariablePreserver()
        protected, vars_list = preserver.protect("")
        self.assertEqual(protected, "")
        self.assertEqual(vars_list, [])


class TestCaseStyleHelper(unittest.TestCase):
    """Test CaseStyleHelper functionality"""

    def test_detect_upper_case(self):
        self.assertEqual(CaseStyleHelper.detect_case("HELLO WORLD"), "upper")
        self.assertEqual(CaseStyleHelper.detect_case("hello world"), "mixed")
        self.assertEqual(CaseStyleHelper.detect_case("Hello World"), "mixed")

    def test_apply_case(self):
        self.assertEqual(CaseStyleHelper.apply_case("hello", "upper"), "HELLO")
        self.assertEqual(CaseStyleHelper.apply_case("HELLO", "mixed"), "HELLO")

    def test_uppercase_line_indices(self):
        text = "TITLE\nHello world\nSUBTITLE\nnormal text"
        indices = CaseStyleHelper.uppercase_line_indices(text)
        self.assertEqual(indices, [0, 2])


class TestTranslationQualityValidator(unittest.TestCase):
    """Test TranslationQualityValidator functionality"""

    def test_valid_translation(self):
        validator = TranslationQualityValidator()
        issues = validator.validate(
            source="Hello world",
            target="Bonjour le monde",
            source_lang="english",
            target_lang="français"
        )
        self.assertEqual(issues, [])

    def test_length_ratio_issue(self):
        validator = TranslationQualityValidator()
        issues = validator.validate(
            source="Hi",
            target="This is a very long translation that is way too verbose",
            source_lang="english",
            target_lang="français"
        )
        self.assertTrue(any("longer" in issue for issue in issues))

    def test_missing_variables(self):
        validator = TranslationQualityValidator()
        issues = validator.validate(
            source="Hello {{name}}, you have {count} items",
            target="Bonjour, vous avez des items",
            source_lang="english",
            target_lang="français"
        )
        self.assertTrue(any("Missing variables" in issue for issue in issues))

    def test_untranslated_text(self):
        validator = TranslationQualityValidator()
        issues = validator.validate(
            source="Programming",
            target="Programming",
            source_lang="english",
            target_lang="français"
        )
        self.assertTrue(any("untranslated" in issue for issue in issues))

    def test_placeholder_corruption(self):
        validator = TranslationQualityValidator()
        issues = validator.validate(
            source="Hello world",
            target="Bonjour __VAR_0__",
            source_lang="english",
            target_lang="français"
        )
        self.assertTrue(any("placeholder" in issue for issue in issues))

    def test_format_issues(self):
        validator = TranslationQualityValidator()
        formatted = validator.format_issues([])
        self.assertIn("No issues", formatted)

        formatted = validator.format_issues(["Issue 1", "Issue 2"])
        self.assertIn("Issue 1", formatted)
        self.assertIn("Issue 2", formatted)


class TestJSONKeySelector(unittest.TestCase):
    """Test JSONKeySelector functionality"""

    def test_flatten_json(self):
        data = {
            "title": "Hello",
            "user": {
                "name": "John",
                "age": 30
            },
            "items": [{"id": 1}, {"id": 2}]
        }
        flat = JSONKeySelector.flatten_json(data)
        self.assertIn("title", flat)
        self.assertIn("user.name", flat)
        self.assertIn("items[0].id", flat)
        self.assertEqual(flat["title"], "Hello")
        self.assertEqual(flat["user.name"], "John")

    def test_match_pattern_direct(self):
        self.assertTrue(JSONKeySelector.match_pattern("title", "title"))
        self.assertFalse(JSONKeySelector.match_pattern("title", "description"))

    def test_match_pattern_wildcard(self):
        self.assertTrue(JSONKeySelector.match_pattern("user.name", "*.name"))
        self.assertTrue(JSONKeySelector.match_pattern("items[0].title", "items[*].title"))
        self.assertFalse(JSONKeySelector.match_pattern("items[0].id", "items[*].title"))

    def test_select_keys(self):
        data = {
            "title": "Hello",
            "description": "World",
            "user": {"name": "John", "bio": "Developer"}
        }
        selected = JSONKeySelector.select_keys(data, ["title", "user.name"])
        self.assertIn("title", selected)
        self.assertIn("user.name", selected)
        self.assertNotIn("description", selected)

    def test_set_nested_value(self):
        data = {"user": {"name": "John"}}
        JSONKeySelector.set_nested_value(data, "user.name", "Jane")
        self.assertEqual(data["user"]["name"], "Jane")

    def test_set_nested_value_array(self):
        data = {"items": [{"id": 1}, {"id": 2}]}
        JSONKeySelector.set_nested_value(data, "items[0].id", 10)
        self.assertEqual(data["items"][0]["id"], 10)

    def test_set_nested_value_invalid_index(self):
        data = {"items": [{"id": 1}]}
        with self.assertRaises(IndexError):
            JSONKeySelector.set_nested_value(data, "items[10].id", 5)

    def test_get_nested_value(self):
        data = {"user": {"name": "John", "age": 30}}
        value = JSONKeySelector.get_nested_value(data, "user.name")
        self.assertEqual(value, "John")

    def test_get_nested_value_missing_key(self):
        data = {"user": {"name": "John"}}
        with self.assertRaises(KeyError):
            JSONKeySelector.get_nested_value(data, "user.missing")

    def test_extract_smart_patterns(self):
        keys = ["items[0].title", "items[1].title", "items[2].title"]
        patterns = JSONKeySelector.extract_smart_patterns(keys)
        self.assertIn("items[*].title", patterns)


def run_tests():
    """Run all tests"""
    unittest.main(verbosity=2)


if __name__ == "__main__":
    run_tests()
