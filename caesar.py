import typing as tp


def encrypt_caesar(plaintext: str, shift: int = 3) -> str:
    """
    Encrypts plaintext using a Caesar cipher.

    >>> encrypt_caesar("PYTHON")
    'SBWKRQ'
    >>> encrypt_caesar("python")
    'sbwkrq'
    >>> encrypt_caesar("Python3.6")
    'Sbwkrq3.6'
    >>> encrypt_caesar("")
    ''
    """
    ciphertext = ""
    for char in plaintext:
        if char.isupper():
            # Shift uppercase letters (A-Z)
            shifted = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
            ciphertext += shifted
        elif char.islower():
            # Shift lowercase letters (a-z)
            shifted = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
            ciphertext += shifted
        else:
            # Keep non-letter characters unchanged (numbers, punctuation, etc.)
            ciphertext += char
    return ciphertext


def decrypt_caesar(ciphertext: str, shift: int = 3) -> str:
    """
    Decrypts a ciphertext using a Caesar cipher.

    >>> decrypt_caesar("SBWKRQ")
    'PYTHON'
    >>> decrypt_caesar("sbwkrq")
    'python'
    >>> decrypt_caesar("Sbwkrq3.6")
    'Python3.6'
    >>> decrypt_caesar("")
    ''
    """
    plaintext = ""
    for char in ciphertext:
        if char.isupper():
            # Shift uppercase letters backwards (A-Z)
            shifted = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
            plaintext += shifted
        elif char.islower():
            # Shift lowercase letters backwards (a-z)
            shifted = chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
            plaintext += shifted
        else:
            # Keep non-letter characters unchanged
            plaintext += char
    return plaintext


def caesar_breaker_brute_force(ciphertext: str, dictionary: tp.Set[str]) -> int:
    """
    Brute force breaking a Caesar cipher.
    """
    best_shift = 0
     best_count = 0
    
    # Try all possible shifts (0-25)
    for shift in range(26):
        # Decrypt with current shift
        decrypted = decrypt_caesar(ciphertext, shift)
        
        # Split into words and count matches in dictionary
        words = decrypted.lower().split()
        match_count = sum(1 for word in words if word in dictionary)
        
        # Track the shift with the most dictionary matches
        if match_count > best_count:
            best_count = match_count
            best_shift = shift
    return best_shift
