#!/usr/bin/env python
"""Fix SystemSpecification to be treated as POD (plain old data) type."""

import os

print("Fixing SystemSpecification to be POD type...")
print("=" * 80)

minimizer_h_path = "/mnt/c/users/scott/Documents/pycalphad/pycalphad/gpu/minimizer.h"

# Read the file
with open(minimizer_h_path, 'r') as f:
    content = f.read()

# Add a pragma to ensure C compilation mode for this struct
old_typedef = "typedef struct SystemSpecification {"

new_typedef = """#ifdef __cplusplus
extern "C" {
#endif

typedef struct SystemSpecification {"""

if old_typedef in content:
    content = content.replace(old_typedef, new_typedef)
    print("Added extern C wrapper for SystemSpecification")
    
    # Find the end of the struct and close the extern C
    # Look for the closing brace and SystemSpecification;
    import re
    pattern = r'(} SystemSpecification;)'
    replacement = r'\1\n\n#ifdef __cplusplus\n}\n#endif'
    content = re.sub(pattern, replacement, content)
    print("Closed extern C wrapper")
else:
    print("WARNING: Could not find typedef struct SystemSpecification")

# Write the fixed content
with open(minimizer_h_path, 'w') as f:
    f.write(content)

print("\nFile updated successfully!")
print("SystemSpecification should now be treated as a POD type.")