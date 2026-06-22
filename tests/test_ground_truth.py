from sift.data.ground_truth import locations_from_diff

# A diff that modifies lines 10-11 of the pre-fix file.
MODIFY_DIFF = """\
--- a/src/calc.py
+++ b/src/calc.py
@@ -9,4 +9,3 @@ def divide(a, b):
     if b == 0:
-        return None
-        raise ValueError
+        raise ZeroDivisionError
     return a / b
"""

# A pure-addition hunk (no removed lines).
ADD_DIFF = """\
--- a/src/util.py
+++ b/src/util.py
@@ -5,2 +5,3 @@ def f():
     x = 1
+    y = 2
     return x
"""


def test_modify_blames_removed_line_span():
    locs = locations_from_diff(MODIFY_DIFF)
    assert len(locs) == 1
    loc = locs[0]
    assert loc.file == "src/calc.py"
    assert loc.start_line == 10
    assert loc.end_line == 11


def test_addition_blames_insertion_point():
    locs = locations_from_diff(ADD_DIFF)
    assert len(locs) == 1
    assert locs[0].file == "src/util.py"
    assert locs[0].start_line == locs[0].end_line  # collapsed to a single line


def test_new_file_has_no_prefix_location():
    new_file = """\
--- /dev/null
+++ b/src/new.py
@@ -0,0 +1,2 @@
+def g():
+    return 1
"""
    assert locations_from_diff(new_file) == []
