from autoscrap_800 import clean, extract_int, extract_power_real


def test_clean_text():
    """
    Tester la fonction clean :
    - Supprime les espaces superflus
    - Retourne None pour chaînes vides ou None
    """
    assert clean("  Peugeot   208 ") == "Peugeot 208"
    assert clean("   Renault Mégane") == "Renault Mégane"
    assert clean("") is None
    assert clean(None) is None


def test_extract_int():
    """
    Tester la fonction extract_int :
    - Extrait un entier depuis une chaîne avec espaces et unités
    - Retourne None si extraction impossible
    """
    assert extract_int("12 345 €") == 12345
    assert extract_int("90 000 km") == 90000
    assert extract_int("abc") is None
    assert extract_int(None) is None
    assert extract_int("0") == 0


def test_extract_power_real():
    """
    Tester la fonction extract_power_real :
    - Extrait la puissance en ch depuis une chaîne
    - Fonctionne même si la valeur est dans une phrase
    - Retourne None si extraction impossible
    """
    assert extract_power_real("Puissance : 130 ch") == 130
    assert extract_power_real("85 ch") == 85
    assert extract_power_real("La voiture a 220ch") == 220
    assert extract_power_real("puissance inconnue") is None
    assert extract_power_real(None) is None
