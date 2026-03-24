# WB2_Project
# Projekt 2.A: Impact of Data Ranking on Training Dynamics

Projekt realizowany w ramach kursu Warsztaty Badawcze 2 (2026) dla Grupy 3. Celem jest zbadanie wpływu rankingu ważności danych na efektywność trenowania modeli.

## Zespół Projektowy
* Kacper Rzeźniczak
* Liliana Sirko
* Syrkiewicz Michał

## Metodologia 
* Badany mechanizm: Ranking oparty na reprezentacjach (Representation-Based Ranking).
* Logika rankingu: Analiza gęstości i odległości w przestrzeni cech przy użyciu modelu DINOv2-ViT-L/14.
* Publikacja bazowa: "What makes for a 'good' Data Augmentation?".
* Zbiór danych: frgfm/imagenette.

## Stos Techniczny
* Narzędzie do zarządzania projektem: uv.
* Linting i formatowanie: ruff.
* Typowanie statyczne: mypy lub pyright.
* Konfiguracja: pyproject.toml.

## Plan Projektu
* Faza 1: Konfiguracja repozytorium oraz analiza matematyczna logiki rankingu.
* Faza 2: Budowa pipeline'u i porównanie treningu na pełnym zbiorze oraz podzbiorach dla modelu ResNet50 (Weights: IMAGENET1K_V2).
* Faza 2 (Warianty): Testy dla modeli zamrożonych (Linear Probe) oraz odblokowanych (Fine-tuning).
* Faza 3: Weryfikacja uniwersalności rankingu na architekturze ConvNeXt_Base (Weights: IMAGENET1K_V1).
