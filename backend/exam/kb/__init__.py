"""
BeanHealth CLR — Clinical Knowledge Base.

Versioned YAML in exam/kb/data, validated at load. Every clinical value the
gaze examination uses lives here with its source and review status.
"""

from exam.kb.loader import DEFAULT_KB_DIR, KBValidationError, KnowledgeBase, load_kb

__all__ = ["DEFAULT_KB_DIR", "KBValidationError", "KnowledgeBase", "load_kb"]
