import ast
import sys

try:
    with open('web_ui.py', 'r') as f:
        source = f.read()
    ast.parse(source)
    print("AST parse successful")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
