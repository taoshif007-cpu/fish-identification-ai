"""Creates species_info.csv. You can also just edit species_info.csv in Excel/Sheets.
Run:  python make_species_csv.py
The values below are a DRAFT written from general ichthyology knowledge.
YOU MUST verify each row against FishBase (https://www.fishbase.se) and
Bangladesh sources (e.g. IUCN Bangladesh, DoF) before publishing the app.
"""
import csv

STATUS = "DRAFT - verify against FishBase / DoF Bangladesh before publishing"

rows = [
 dict(class_id="rohu", common_name="Rohu", local_name="Rui (রুই)", scientific_name="Labeo rohita",
  family="Cyprinidae", order="Cypriniformes",
  habitat="Freshwater: rivers, floodplains, ponds; major cultured carp.",
  distribution="Native to South Asia (Indo-Gangetic rivers incl. Bangladesh); widely introduced.",
  identification_characteristics="Elongated, moderately deep body; small sub-terminal mouth with fringed lower lip; no prominent barbels; reddish tint on eyes and scales; bluish-brown back, silvery flanks.",
  similar_species="katla;mrigal"),
 dict(class_id="katla", common_name="Catla", local_name="Katla (কাতলা)", scientific_name="Labeo catla (also listed as Gibelion catla / Catla catla)",
  family="Cyprinidae", order="Cypriniformes",
  habitat="Freshwater: rivers and ponds; surface and column feeder.",
  distribution="Native to South Asia (Indo-Gangetic rivers incl. Bangladesh); widely introduced.",
  identification_characteristics="Very large broad head (about a quarter or more of body length); wide upturned mouth with protruding lower jaw; deep body; no barbels; greyish back, silvery sides.",
  similar_species="rohu;silver_carp"),
 dict(class_id="mrigal", common_name="Mrigal", local_name="Mrigel (মৃগেল)", scientific_name="Cirrhinus cirrhosus",
  family="Cyprinidae", order="Cypriniformes",
  habitat="Freshwater: rivers and ponds; bottom feeder.",
  distribution="Native to South Asia (Indo-Gangetic rivers incl. Bangladesh); widely introduced.",
  identification_characteristics="Slender, elongated, fairly cylindrical body; small head with blunt snout; sub-terminal mouth; silvery body, often with a reddish-orange tinge on fins.",
  similar_species="rohu;katla"),
 dict(class_id="grass_carp", common_name="Grass carp", local_name="Grass carp (গ্রাস কার্প)", scientific_name="Ctenopharyngodon idella",
  family="Cyprinidae", order="Cypriniformes",
  habitat="Freshwater: ponds, lakes, rivers; herbivore eating aquatic plants.",
  distribution="Native to eastern Asia (China/Russia Far East); introduced to Bangladesh and many countries.",
  identification_characteristics="Torpedo-shaped, almost cylindrical body; broad head and terminal mouth; no barbels; large scales with dark edges; olive-brown back, pale belly.",
  similar_species="rohu;common_carp"),
 dict(class_id="silver_carp", common_name="Silver carp", local_name="Silver carp (সিলভার কার্প)", scientific_name="Hypophthalmichthys molitrix",
  family="Xenocyprididae (often listed under Cyprinidae in older sources)", order="Cypriniformes",
  habitat="Freshwater: ponds, rivers; filter-feeds on phytoplankton.",
  distribution="Native to eastern Asia; introduced to Bangladesh and many countries.",
  identification_characteristics="Laterally compressed silvery body; very small scales; large head with low-set eyes; upturned mouth; keel along the belly.",
  similar_species="katla;hilsa"),
 dict(class_id="common_carp", common_name="Common carp", local_name="Common carp (কমন কার্প)", scientific_name="Cyprinus carpio",
  family="Cyprinidae", order="Cypriniformes",
  habitat="Freshwater: ponds, lakes, slow rivers; bottom-feeding omnivore.",
  distribution="Native to Eurasia; introduced to Bangladesh and worldwide.",
  identification_characteristics="Two pairs of barbels at the upper jaw; protrusible mouth; long dorsal fin with a strong serrated spine; golden-olive body with large scales.",
  similar_species="grass_carp;mrigal"),
 dict(class_id="nile_tilapia", common_name="Nile tilapia", local_name="Tilapia (তেলাপিয়া)", scientific_name="Oreochromis niloticus",
  family="Cichlidae", order="Cichliformes (Perciformes in older sources)",
  habitat="Freshwater ponds, lakes, rivers; tolerates a wide range of conditions.",
  distribution="Native to Africa and the Levant; introduced to Bangladesh and worldwide for aquaculture.",
  identification_characteristics="Perch-like deep body; one long dorsal fin with spines then soft rays; regular vertical dark stripes on the tail fin; faint vertical bars on the body.",
  similar_species=""),
 dict(class_id="hilsa", common_name="Hilsa shad", local_name="Ilish (ইলিশ)", scientific_name="Tenualosa ilisha",
  family="Clupeidae", order="Clupeiformes",
  habitat="Anadromous: lives in coastal marine/estuarine waters and migrates up large rivers to breed.",
  distribution="Northern Indian Ocean: Bay of Bengal and rivers such as the Padma-Meghna system in Bangladesh; also Myanmar, India and the Persian Gulf region.",
  identification_characteristics="Laterally compressed silvery body with sharp belly scutes; notch in the middle of the upper jaw; dark spot behind the gill cover; no lateral line; large scales.",
  similar_species="silver_carp"),
]

fields = ["class_id","common_name","local_name","scientific_name","family","order","habitat",
          "distribution","identification_characteristics","similar_species","info_status"]
with open("species_info.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        r["info_status"] = STATUS
        w.writerow(r)
print("Wrote species_info.csv with", len(rows), "species")
