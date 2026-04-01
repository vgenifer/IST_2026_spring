def encrypt_vigenere(plaintext: str, keyword: str) -> str:
    """
    Encrypts plaintext using a Vigenere cipher.

    >>> encrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> encrypt_vigenere("python", "a")
    'python'
    >>> encrypt_vigenere("ATTACKATDAWN", "LEMON")
    'LXFOPVEFRNHR'
    """
    ciphertext = ""
     keyword_index = 0
    
    for char in plaintext:
        if char.isalpha():
            # Get the shift amount from the keyword letter
            shift = ord(keyword[keyword_index % len(keyword)].upper()) - ord('A')
            
            if char.isupper():
                # Shift uppercase letters
                shifted = chr((ord(char) - ord('A') + shift) % 26 + ord('A'))
                ciphertext += shifted
            else:
                # Shift lowercase letters
                shifted = chr((ord(char) - ord('a') + shift) % 26 + ord('a'))
                ciphertext += shifted
            
            # Move to the next keyword letter (only for alphabetic characters)
            keyword_index += 1
        else:
            # Keep non-letter characters unchanged
            ciphertext += char
    return ciphertext


def decrypt_vigenere(ciphertext: str, keyword: str) -> str:
    """
    Decrypts a ciphertext using a Vigenere cipher.

    >>> decrypt_vigenere("PYTHON", "A")
    'PYTHON'
    >>> decrypt_vigenere("python", "a")
    'python'
    >>> decrypt_vigenere("LXFOPVEFRNHR", "LEMON")
    'ATTACKATDAWN'
    """
    plaintext = ""
    keyword_index = 0
    
    for char in ciphertext:
        if char.isalpha():
            # Get the shift amount from the keyword letter
            shift = ord(keyword[keyword_index % len(keyword)].upper()) - ord('A')
            
            if char.isupper():
                # Shift uppercase letters backwards
                shifted = chr((ord(char) - ord('A') - shift) % 26 + ord('A'))
                plaintext += shifted
            else:
                # Shift lowercase letters backwards
                shifted = chr((ord(char) - ord('a') - shift) % 26 + ord('a'))
                plaintext += shifted
            
            # Move to the next keyword letter (only for alphabetic characters)
            keyword_index += 1
        else:
            # Keep non-letter characters unchanged
            plaintext += char
    
    return plaintext
