import hashlib
import random
import re
from typing import Dict, Optional


class FormatPreservingEmailAnonymizer:
    def __init__(self, preserve_length: bool = True, preserve_structure: bool = True):
        """
        Initialize the email anonymizer.

        Args:
            preserve_length: Whether to preserve the length of original parts
            preserve_structure: Whether to preserve dots, hyphens in the structure
        """
        self.preserve_length = preserve_length
        self.preserve_structure = preserve_structure
        self.email_cache: Dict[str, str] = {}
        self.part_cache: Dict[str, str] = {}  # Cache for individual parts

        # Character sets for replacement
        self.alphanumeric = "abcdefghijklmnopqrstuvwxyz0123456789"
        self.letters = "abcdefghijklmnopqrstuvwxyz"

    def _get_deterministic_random(self, seed: str) -> random.Random:
        """Create a deterministic random generator from a seed."""
        # Use hash of the seed as random seed for consistency
        hash_int = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
        return random.Random(hash_int)

    def _preserve_structure_replace(self, text: str, seed: str) -> str:
        """
        Replace text while preserving structure (length, special chars, case pattern).
        """
        if text in self.part_cache:
            return self.part_cache[text]

        rng = self._get_deterministic_random(seed)
        result = []

        for char in text:
            if char.isalpha():
                # Preserve case pattern
                new_char = rng.choice(self.letters)
                result.append(new_char.upper() if char.isupper() else new_char)
            elif char.isdigit():
                result.append(str(rng.randint(0, 9)))
            else:
                # Keep special characters (dots, hyphens, etc.)
                result.append(char)

        anonymized = "".join(result)
        self.part_cache[text] = anonymized
        return anonymized

    def _simple_hash_replace(self, text: str, target_length: Optional[int] = None) -> str:
        """Simple hash replacement with optional length preservation."""
        if target_length is None:
            target_length = len(text)

        hash_val = hashlib.md5(text.encode()).hexdigest()

        # Create a mix of letters and numbers that looks more natural
        result = []
        for i in range(target_length):
            if i < len(hash_val):
                char = hash_val[i]
                if char.isdigit():
                    result.append(char)
                else:
                    # Convert hex chars to letters
                    result.append(chr(ord("a") + (ord(char) - ord("a")) % 26))
            else:
                result.append("x")

        return "".join(result)

    def _anonymize_email(self, email: str) -> str:
        """
        Anonymize a single email while preserving format and structure.
        """
        if email in self.email_cache:
            return self.email_cache[email]

        # Split email into local part and domain
        local_part, domain = email.split("@", 1)

        if self.preserve_structure:
            # Preserve the structure (dots, hyphens, length, case pattern)
            local_anonymized = self._preserve_structure_replace(local_part, f"local_{local_part}")
            domain_anonymized = self._preserve_structure_replace(domain, f"domain_{domain}")
        else:
            # Simple length-preserving hash
            local_length = len(local_part) if self.preserve_length else 8
            domain_length = len(domain) if self.preserve_length else 8

            local_anonymized = self._simple_hash_replace(local_part, local_length)
            domain_anonymized = self._simple_hash_replace(domain, domain_length)

        anonymized_email = f"{local_anonymized}@{domain_anonymized}"
        self.email_cache[email] = anonymized_email

        return anonymized_email

    def anonymize_text(self, text: str) -> str:
        """
        Anonymize all emails in the given text while preserving format.
        """
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"

        def replace_email(match: re.Match[str]) -> str:
            email = match.group(0)
            return self._anonymize_email(email)

        return re.sub(email_pattern, replace_email, text)

    def get_mapping(self) -> Dict[str, str]:
        """Return the mapping of original emails to anonymized versions."""
        return self.email_cache.copy()


# Example usage and comparison
if __name__ == "__main__":
    print("Format-Preserving Email Anonymization - Structure Preserving:\n")

    # Structure-preserving anonymizer
    anonymizer1 = FormatPreservingEmailAnonymizer(preserve_structure=True)

    test_emails = [
        "john@gmail.com",
        "john@gmail.com",
        "john@outlook.com",
        "joe@outlook.com",
        "user.name@company.co.uk",
        "test-email@sub.example.org",
        "Admin123@BigCorp.net",
        "a@b.co",
    ]

    print("Structure-Preserving Mode:")
    print("=" * 50)
    for email in test_emails:
        anonymized = anonymizer1._anonymize_email(email)
        print(f"{email:25} -> {anonymized}")

    print("\n" + "=" * 50)
    print("Length-Preserving Mode:")
    print("=" * 50)

    # Length-preserving but simpler anonymizer
    anonymizer2 = FormatPreservingEmailAnonymizer(preserve_length=True, preserve_structure=False)

    for email in test_emails:
        anonymized = anonymizer2._anonymize_email(email)
        print(f"{email:25} -> {anonymized}")

    # Test with full text
    print("\n" + "=" * 70)
    print("Full Text Anonymization Examples:")
    print("=" * 70)

    test_texts = [
        "Hi, my name is John and my email is john@gmail.com",
        "Contact: support@company.com or admin@BigCorp.net",
        "Emails: user.name@test.co.uk, simple@domain.org",
    ]

    for text in test_texts:
        anonymized = anonymizer1.anonymize_text(text)
        print(f"Original:   {text}")
        print(f"Anonymized: {anonymized}")
        print()

    # Consistency test
    print("Consistency Test:")
    print("-" * 30)
    email = "john@gmail.com"
    result1 = anonymizer1._anonymize_email(email)
    result2 = anonymizer1._anonymize_email(email)
    print(f"First call:  {email} -> {result1}")
    print(f"Second call: {email} -> {result2}")
    print(f"Consistent: {'✓' if result1 == result2 else '✗'}")
