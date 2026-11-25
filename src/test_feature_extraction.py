"""
PhishGuard – Feature Extraction Unit & Validation Tests
========================================================
Runs tests across required URL scenarios:
1. Normal HTTPS URL
2. HTTP URL
3. URL containing multiple subdomains
4. URL containing suspicious keywords
5. URL containing an IP address
6. Invalid URL
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(__file__))
from feature_extraction import (
    FEATURE_COLUMNS,
    extract_features_single,
    is_valid_url
)

def run_tests():
    test_cases = [
        ("Normal HTTPS URL", "https://www.google.com/search?q=machine+learning"),
        ("HTTP URL", "http://example-blog.org/articles/post-1.html"),
        ("Multiple Subdomains URL", "https://portal.secure.banking.verify.paypal.com.account-update.xyz/login"),
        ("Suspicious Keywords URL", "http://chase-security-update.com/verify/account/login.php?user=alert"),
        ("IP Address URL", "http://192.168.1.105/admin/auth/signin.html"),
        ("Invalid URL", "not_a_valid_url_string!@#$$%^^&&*"),
    ]
    
    results = []
    print("=" * 80)
    print("PHISHGUARD FEATURE EXTRACTION VERIFICATION TEST SUITE")
    print("=" * 80)
    
    for category, url in test_cases:
        df_feat = extract_features_single(url)
        # Add metadata for reporting
        df_feat.insert(0, "Category", category)
        df_feat.insert(1, "Test_URL", url)
        results.append(df_feat)
        
        print(f"\n[TEST CASE: {category.upper()}]")
        print(f"URL: {url}")
        for col in FEATURE_COLUMNS:
            print(f"  • {col:25s}: {df_feat[col].iloc[0]}")
            
    summary_df = pd.concat(results, ignore_index=True)

    validation_cases = [
        ("HTTPS URL", "https://www.google.com/", True),
        ("Bare hostname", "example.com", True),
        ("IP URL", "http://192.168.1.105/admin", True),
        ("Blank input", "  ", False),
        ("Malformed input", "not_a_valid_url_string!@#$$%^^&&*", False),
        ("Unsupported scheme", "javascript:alert(1)", False),
        ("Invalid port", "https://example.com:bad/", False),
    ]
    for category, url, expected in validation_cases:
        actual = is_valid_url(url)
        if actual != expected:
            raise AssertionError(f"{category}: expected {expected}, got {actual}")
        print(f"[URL VALIDATION: {category}] {'PASS' if actual == expected else 'FAIL'}")

    return summary_df

if __name__ == "__main__":
    summary_df = run_tests()
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 80)
