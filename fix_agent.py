"""Comprehensive fix for core/agent.py - fix all syntax issues"""
import re

filepath = 'core/agent.py'
with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

fixed = 0
i = 0
result = []

while i < len(lines):
    line = lines[i]
    stripped = line.rstrip()

    # Pattern: self._write('key', value  (missing closing paren)
    # where value is a simple expression (no dict literal, no nested calls)
    if 'self._write(' in stripped and not stripped.rstrip().endswith(')'):
        # Check if this is a simple one-liner missing )
        # Count open/close parens
        opens = stripped.count('(')
        closes = stripped.count(')')

        if opens > closes:
            # Is it a multi-line dict? Check if line ends with {
            if stripped.rstrip().endswith('{'):
                result.append(line)
                i += 1
                continue

            # Is it a multi-line call? Check if line ends with ,
            if stripped.rstrip().endswith(','):
                result.append(line)
                i += 1
                continue

            # Simple fix: add missing ) at end
            deficit = opens - closes
            result.append(stripped + ')' * deficit + '\n')
            fixed += 1
            i += 1
            continue

    result.append(line)
    i += 1

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(result)

print(f'Fixed {fixed} lines')

# Verify
import ast
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
try:
    ast.parse(content)
    print('Syntax OK!')
except SyntaxError as e:
    print(f'Remaining error at line {e.lineno}: {e.msg}')
    lines2 = content.split('\n')
    for j in range(max(0,e.lineno-3), min(len(lines2), e.lineno+2)):
        m = '>>>' if j+1==e.lineno else '   '
        print(f'{m} {j+1}: {lines2[j][:120]}')
