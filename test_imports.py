"""
Test script to verify imports from the navigation module.
"""

import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    print("Attempting to import from hypersynceddt.frontend.components.navigation...")
    from hypersynceddt.frontend.components.navigation import (
        get_role_specific_pages,
        Page,
        Category,
        render_integration_options,
        render_top_navigation,
        render_sidebar_navigation
    )
    print("Successfully imported all required components!")
    
    # Test the get_role_specific_pages function
    print("\nTesting get_role_specific_pages function:")
    pages = get_role_specific_pages("Operator")
    print(f"Found {sum(len(pages[cat]) for cat in pages)} pages for Operator role")
    for category, page_list in pages.items():
        print(f"  Category: {category}")
        for page in page_list:
            print(f"    - {page.name} ({page.id})")
    
    print("\nAll imports and functions are working correctly!")
except ImportError as e:
    print(f"Import Error: {e}")
    
    # Check if the module exists
    print("\nChecking module structure:")
    try:
        import hypersynceddt
        print("✓ hypersynceddt module exists")
        
        import hypersynceddt.frontend
        print("✓ hypersynceddt.frontend module exists")
        
        import hypersynceddt.frontend.components
        print("✓ hypersynceddt.frontend.components module exists")
        
        import hypersynceddt.frontend.components.navigation
        print("✓ hypersynceddt.frontend.components.navigation module exists")
        
        # Check what's in the navigation module
        print("\nContents of navigation module:")
        print(dir(hypersynceddt.frontend.components.navigation))
    except ImportError as e2:
        print(f"Module structure error: {e2}")

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hypersynceddt.frontend.advanced_ml_models import SynchronizedDigitalTwin
    print("Successfully imported SynchronizedDigitalTwin")
except ImportError as e:
    print(f"Error importing SynchronizedDigitalTwin: {e}")

try:
    from hypersynceddt.frontend.integrated_dashboard import IntegratedDashboard
    print("Successfully imported IntegratedDashboard")
except ImportError as e:
    print(f"Error importing IntegratedDashboard: {e}") 