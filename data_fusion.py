import json
import os

base_path = r"C:\Users\nrous\Documents\M2\Web Scraping & Machine Learning\Projet\PredictMyCar"

input_files = [
    "autoscrap_800.json",
    "autoscrap_1600.json",
    "autoscrap_2400.json",
    "autoscrap_3200.json",
    "autoscrap_4000.json",
    "autoscrap_4800.json",
    "autoscrap_5600.json",
]

output_file = os.path.join(base_path, "autoscrap_FIN.json")

missing_values = {None, "", "NA", "N/A", "null"}


merged_data = []

for file in input_files:
    file_path = os.path.join(base_path, file)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        if isinstance(data, list):
            merged_data.extend(data)
        else:
            raise ValueError(f"❌ {file} ne contient pas une liste JSON")

print(f"✅ Fusion terminée : {len(merged_data)} lignes chargées")


print("\n🔍 Valeurs manquantes détectées :\n")
missing_count = 0

for index, row in enumerate(merged_data):
    for key, value in row.items():
        if value in missing_values:
            print(f"Ligne {index} | Colonne '{key}' | Valeur = {value}")
            missing_count += 1

print(f"\n📊 Total de valeurs manquantes : {missing_count}")


cp_modified = 0

for row in merged_data:
    if "code_postal" in row and row["code_postal"] is not None:
        cp = str(row["code_postal"])
        row["code_postal"] = cp[:2]
        cp_modified += 1

print(f"✅ Codes postaux modifiés : {cp_modified}")


km_modified = 0

for row in merged_data:
    if "kilometrage" in row and row["kilometrage"] is not None:
        km_str = str(row["kilometrage"])
        if len(km_str) > 4:
            row["kilometrage"] = int(km_str[4:])
            km_modified += 1

print(f"✅ Kilométrages modifiés : {km_modified}")


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4)

print(f"\n🎉 Fichier final créé avec succès : {output_file}")
 