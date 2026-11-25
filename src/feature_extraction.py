"""
PhishGuard – URL Feature Extraction Module
===========================================
This module provides reusable, purely offline lexical and structural feature 
extraction from raw website URLs. It powers both the offline batch training 
pipeline and real-time live inference in the Streamlit application.

Key design principles:
- Safe URL component parsing via urllib.parse.
- Reliable IP address detection using the standard ipaddress library.
- Zero network or socket calls (ensures high speed, privacy, and security).
- Robust error handling for malformed, missing, or irregular URLs.
- Consistent feature naming and strict column ordering for machine learning.
"""

from typing import Dict, List, Tuple, Union
import urllib.parse
import ipaddress
import pandas as pd
import numpy as np


# ------------------------------------------------------------------------------
# Constants & Configuration
# ------------------------------------------------------------------------------

# Strict, immutable feature column ordering for training and inference
FEATURE_COLUMNS: List[str] = [
    "url_length",
    "domain_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "is_https",
    "has_ip_address",
    "num_suspicious_keywords",
    "num_subdomains",
    "has_at_symbol",
    "num_special_chars",
    "has_redirection",
]

# Sensitive keywords commonly found in credential-harvesting phishing URLs
SUSPICIOUS_KEYWORDS: Tuple[str, ...] = (
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "banking",
    "signin",
    "confirm",
    "security",
    "authenticate",
    "password",
)

# Common special characters used for obfuscation and parameter manipulation
SPECIAL_CHARS: Tuple[str, ...] = ("_", "?", "=", "&", "%", "~", "+")


# ------------------------------------------------------------------------------
# URL Parsing Helper
# ------------------------------------------------------------------------------

def parse_url_components(url: Union[str, float, None]) -> Tuple[str, str, str, str]:
    """
    Parses a raw URL string into (scheme, domain, path, query).
    
    Handles URLs with or without explicit schemes (e.g. 'http://' or bare domains)
    gracefully without modifying the original input.
    """
    if not isinstance(url, str):
        return "", "", "", ""
    
    cleaned = url.strip()
    if not cleaned:
        return "", "", "", ""
    
    try:
        if "://" in cleaned:
            parsed = urllib.parse.urlsplit(cleaned)
            scheme = parsed.scheme.lower()
            netloc = parsed.netloc
            path = parsed.path
            query = parsed.query
        else:
            # Prepend '//' so urlsplit correctly resolves netloc for bare URLs
            parsed = urllib.parse.urlsplit("//" + cleaned)
            scheme = ""
            netloc = parsed.netloc
            path = parsed.path
            query = parsed.query
        
        # Clean domain: remove userinfo (before '@') and port (after ':')
        domain = netloc
        if "@" in domain:
            domain = domain.split("@")[-1]
        if ":" in domain:
            domain = domain.split(":")[0]
            
        return scheme, domain.lower(), path, query
    except Exception:
        # Fallback for severely corrupted strings
        return "", "", "", ""


def is_valid_url(url: Union[str, None]) -> bool:
    """Checks for a syntactically valid HTTP(S) URL or bare hostname."""
    if not isinstance(url, str) or not url.strip():
        return False

    cleaned = url.strip()
    if any(character.isspace() for character in cleaned):
        return False

    has_scheme = "://" in cleaned
    try:
        parsed = urllib.parse.urlsplit(cleaned if has_scheme else "//" + cleaned)
        parsed.port
    except ValueError:
        return False

    if has_scheme and parsed.scheme.lower() not in ("http", "https"):
        return False

    hostname = parsed.hostname
    if not hostname:
        return False

    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        pass

    try:
        ascii_hostname = hostname.rstrip(".").encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if not ascii_hostname or len(ascii_hostname) > 253:
        return False

    labels = ascii_hostname.split(".")
    return all(
        0 < len(label) <= 63
        and label[0].isalnum()
        and label[-1].isalnum()
        and all(character.isalnum() or character == "-" for character in label)
        for label in labels
    )


# ------------------------------------------------------------------------------
# Individual Feature Extraction Functions
# ------------------------------------------------------------------------------

def get_url_length(url: str) -> int:
    """Feature 1: Total character length of the raw URL."""
    return len(url.strip()) if isinstance(url, str) else 0


def get_domain_length(domain: str) -> int:
    """Feature 2: Character length of the fully qualified domain name (FQDN)."""
    return len(domain.strip()) if isinstance(domain, str) else 0


def count_dots(url: str) -> int:
    """Feature 3: Number of dot ('.') characters in the URL."""
    return url.count(".") if isinstance(url, str) else 0


def count_hyphens(url: str) -> int:
    """Feature 4: Number of hyphen ('-') characters in the URL."""
    return url.count("-") if isinstance(url, str) else 0


def count_digits(url: str) -> int:
    """Feature 5: Number of numerical digits (0-9) in the URL."""
    return sum(c.isdigit() for c in url) if isinstance(url, str) else 0


def check_is_https(scheme: str, url: str) -> int:
    """
    Feature 6: Binary flag indicating if the URL explicitly uses HTTPS.
    Returns 1 if HTTPS, 0 otherwise.
    """
    if scheme == "https":
        return 1
    if isinstance(url, str) and url.strip().lower().startswith("https://"):
        return 1
    return 0


def check_has_ip_address(domain: str) -> int:
    """
    Feature 7: Binary flag indicating if the domain is a raw IPv4 or IPv6 address.
    Phishers often host attack pages on raw IP addresses to bypass domain reputation.
    Returns 1 if IP address, 0 otherwise.
    """
    if not domain or not isinstance(domain, str):
        return 0
    try:
        ipaddress.ip_address(domain.strip())
        return 1
    except ValueError:
        return 0


def count_suspicious_keywords(url: str) -> int:
    """
    Feature 8: Total count of deceptive security and credential-harvesting tokens.
    Searches for terms like 'login', 'verify', 'update', 'secure', 'account', etc.
    """
    if not isinstance(url, str):
        return 0
    url_lower = url.lower()
    return sum(url_lower.count(kw) for kw in SUSPICIOUS_KEYWORDS)


def count_subdomains(domain: str) -> int:
    """
    Feature 9: Number of subdomain levels preceding the primary domain and TLD.
    Example: 'sub1.sub2.example.com' has 2 subdomains; 'example.com' has 0.
    Standard 'www.' is stripped so as not to penalize benign websites.
    """
    if not domain or not isinstance(domain, str):
        return 0
    if check_has_ip_address(domain):
        return 0
    
    clean_domain = domain.strip().lower()
    if clean_domain.startswith("www."):
        clean_domain = clean_domain[4:]
        
    parts = [p for p in clean_domain.split(".") if p]
    # Standard domain (name + tld) has 2 parts -> 0 subdomains
    if len(parts) > 2:
        return len(parts) - 2
    return 0


def check_has_at_symbol(url: str) -> int:
    """
    Feature 10: Binary flag indicating presence of '@' in the URL.
    Browsers treat text before '@' as user credentials and navigate to what follows,
    which is a classic phishing deception technique.
    """
    return 1 if (isinstance(url, str) and "@" in url) else 0


def count_special_characters(url: str) -> int:
    """
    Feature 11: Total count of selected delimiter characters ('_', '?', '=', '&', '%', '~', '+').
    """
    if not isinstance(url, str):
        return 0
    return sum(url.count(c) for c in SPECIAL_CHARS)


def check_has_redirection(url: str, path: str, query: str) -> int:
    """
    Feature 12: Binary flag indicating URL redirection patterns.
    Triggers if '//' appears outside the protocol prefix, or if redirection
    tokens like 'redirect', 'url=', or 'rdir' exist in path/query parameters.
    """
    if not isinstance(url, str):
        return 0
    
    # Check for embedded '//' beyond initial protocol
    first_double_slash = url.find("//")
    if first_double_slash != -1:
        second_double_slash = url.find("//", first_double_slash + 2)
        if second_double_slash != -1:
            return 1
            
    # Check for open redirection keywords in path and query
    combined = (path + "?" + query).lower()
    for token in ("redirect", "url=", "rdir", "dest=", "target="):
        if token in combined:
            return 1
            
    return 0


# ------------------------------------------------------------------------------
# Core Feature Extraction Engine
# ------------------------------------------------------------------------------

def extract_features_from_url(url: Union[str, float, None]) -> Dict[str, Union[int, float]]:
    """
    Extracts all 12 numerical features from a single raw URL string.
    Returns a dictionary mapping feature names to numerical values.
    """
    if not isinstance(url, str) or not url.strip():
        # Clean default zeros for empty or null input
        return {col: 0 for col in FEATURE_COLUMNS}
    
    raw_url = url.strip()
    scheme, domain, path, query = parse_url_components(raw_url)
    
    return {
        "url_length": get_url_length(raw_url),
        "domain_length": get_domain_length(domain),
        "num_dots": count_dots(raw_url),
        "num_hyphens": count_hyphens(raw_url),
        "num_digits": count_digits(raw_url),
        "is_https": check_is_https(scheme, raw_url),
        "has_ip_address": check_has_ip_address(domain),
        "num_suspicious_keywords": count_suspicious_keywords(raw_url),
        "num_subdomains": count_subdomains(domain),
        "has_at_symbol": check_has_at_symbol(raw_url),
        "num_special_chars": count_special_characters(raw_url),
        "has_redirection": check_has_redirection(raw_url, path, query),
    }


def extract_features_single(url: str) -> pd.DataFrame:
    """
    Accepts a single raw URL from a user or application and returns a 1-row
    Pandas DataFrame formatted with identical feature column names and order
    as used during machine learning model training.
    """
    features_dict = extract_features_from_url(url)
    df = pd.DataFrame([features_dict], columns=FEATURE_COLUMNS)
    return df


def extract_features_batch(urls: List[str]) -> pd.DataFrame:
    """
    Batch feature extraction across an iterable list of URL strings.
    Returns a DataFrame with rows corresponding to URLs and columns matching FEATURE_COLUMNS.
    """
    records = [extract_features_from_url(u) for u in urls]
    return pd.DataFrame(records, columns=FEATURE_COLUMNS)
