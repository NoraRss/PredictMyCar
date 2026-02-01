from playwright.sync_api import sync_playwright
import json
import re
import time
import os

BASE_URL = "https://www.autosphere.fr/recherche"
# 35 pages
PAGES = [i*23 for i in range(0, 35)]



annonces = []


def clean(text):
    """Nettoie une chaîne de caractères en supprimant les espaces superflus et les retours à la ligne."""
    return re.sub(r"\s+", " ", text.strip()) if text else None


def extract_int(text):
    """Extrait un entier depuis une chaîne de caractères, ou retourne None si impossible."""
    if not text:
        return None
    nums = re.findall(r"\d+", text.replace(" ", ""))
    return int("".join(nums)) if nums else None


def extract_power_real(text):
    """Extrait la puissance réelle en chevaux (ex: '130 ch') depuis une chaîne de texte."""
    if not text:
        return None
    match = re.search(r"(\d{2,4})\s*ch", text)
    return int(match.group(1)) if match else None


def run_scraping():
    """
    Lance le scraping complet des annonces sur Autosphere.

    - Parcourt toutes les pages définies
    - Récupère les informations principales et détails de chaque annonce
    - Nettoie et transforme certaines données
    - Sauvegarde le résultat dans 'data/autoscrap_800.json'
    """
    global annonces
    annonces = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for offset in PAGES:
            url = BASE_URL if offset == 0 else f"{BASE_URL}?from={offset}"
            print(f"Scraping {url}")

            page.goto(url, timeout=60000)
            page.wait_for_timeout(5000)
 
            for _ in range(6):
                page.mouse.wheel(0, 4000)
                page.wait_for_timeout(1000)

            cards = page.query_selector_all("article")

            for card in cards:
                a_tag = card.query_selector("a")
                annonce_url = a_tag.get_attribute("href") if a_tag else None
                if not annonce_url:
                   continue

                annonce_url = "https://www.autosphere.fr" + annonce_url

                titre_tag = card.query_selector("h2") or card.query_selector("h3")
                nom_annonce = clean(titre_tag.inner_text()) if titre_tag else None

                prix = None
                price_tags = card.query_selector_all("span, div")
                for pt in price_tags:
                    txt = pt.inner_text().lower()
                    if "€" in txt and "mois" not in txt and "mensuel" not in txt:
                        prix_match = re.search(r"\d[\d\s]*€", txt)
                        if prix_match:
                            prix = extract_int(prix_match.group())
                            break

                text_card = card.inner_text().lower()

                annee = None
                year_match = re.search(r"(19|20)\d{2}", text_card)
                if year_match:
                   annee = int(year_match.group())

                kilometrage = None
                km_match = re.search(r"(\d[\d\s]*)\s*km", text_card)
                if km_match:
                   kilometrage = extract_int(km_match.group(1))

                carburant = None
                for c in ["diesel", "essence", "hybride", "electrique","e85","gpl"]:
                    if c in text_card:
                       carburant = c.capitalize()
                       break

                if "automatique" in text_card:
                    boite = "Automatique"
                elif "manuelle" in text_card:
                    boite = "Manuelle"
                else:
                    boite = None

                puissance_fiscale = None
                puissance_reelle = None
                code_postal = None

                try:
                    detail_page = browser.new_page()
                    detail_page.goto(annonce_url, timeout=60000)
                    detail_page.wait_for_selector("body", timeout=10000)

                    page_text = detail_page.inner_text("body").lower()

                    cv_match = re.search(r"puissance fiscale\s*:\s*(\d{1,3})\s*cv", page_text)
                    if cv_match:
                        puissance_fiscale = int(cv_match.group(1))

                    puissance_reelle = extract_power_real(page_text)

                    cp_match = re.search(r"\((\d{5})\)", page_text)
                    if cp_match:
                        code_postal = cp_match.group(1)

            
                    detail_page.close()

                except Exception as e:
                    print(f"Erreur sur {annonce_url} : {e}")

                annonces.append({
                       "nom_annonce": nom_annonce,
                       "url_annonce": annonce_url,
                       "prix": prix,
                       "annee": annee,
                       "kilometrage": kilometrage,
                       "puissance_fiscale_cv": puissance_fiscale,
                       "puissance_reelle_ch": puissance_reelle,
                       "carburant": carburant,
                       "boite_vitesse": boite,
                       "code_postal": code_postal,
                
            })

            time.sleep(5)

        browser.close()


    os.makedirs("data", exist_ok=True)

    with open("data/autoscrap_800.json", "w", encoding="utf-8") as f:
       json.dump(annonces, f, ensure_ascii=False, indent=2)

    print(f"\n✅ {len(annonces)} annonces enregistrées dans autoscrap_800.json")

if __name__ == "__main__":
    run_scraping()