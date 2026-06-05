"""Fix all missing closing parentheses in core/agent.py self._write() calls"""
import re

filepath = 'core/agent.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

lines = content.split('\n')
fixed_count = 0

i = 0
while i < len(lines):
    line = lines[i]
    stripped = line.rstrip()

    if 'self._write(' in stripped:
        # Count unmatched parens on this line
        opens = stripped.count('(')
        closes = stripped.count(')')
        deficit = opens - closes

        if deficit > 0:
            # Check if this is a multi-line dict literal (ends with {)
            if stripped.rstrip().endswith('{'):
                # Skip - it's a multi-line dict, will close later
                i += 1
                continue

            # Check if next line is a continuation
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                # If next line looks like a dict key or continuation
                if next_stripped.startswith("'") or next_stripped.startswith('"'):
                    if next_stripped.endswith(',') or next_stripped.endswith('}'):
                        i += 1
                        continue

            # Check if this is a multi-line expression (like max(...))
            if stripped.rstrip().endswith(','):
                i += 1
                continue

            # Add missing closing parens
            lines[i] = stripped + ')' * deficit
            fixed_count += 1

    i += 1

content = '\n'.join(lines)
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print(f'Fixed {fixed_count} lines with missing closing parentheses')
