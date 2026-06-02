# Warsztaty Badawcze 2
## Projekt 2.A: Impact of Data Ranking on Training Dynamics

Projekt realizowany w ramach kursu Warsztaty Badawcze 2 (2026) dla Grupy 3. Celem projektu jest badanie wpływu rankingu oraz strategii próbkowania danych (w tym przy użyciu hierarchicznego K-means) na dynamikę trenowania modeli. Eksperymenty przeprowadzane są z wykorzystaniem zbiorów danych Imagenette, ImageNet-1K oraz Places365.

## Zespół Projektowy
* Kacper Rzeźniczak
* Liliana Sirko
* Michał Syrkiewicz

## Struktura Repozytorium

* `clustering_helpers/` - moduły i funkcje pomocnicze do przeprowadzania klasteryzacji (m.in. implementacja hierarchicznego K-means).
* `dataprep/` - skrypty odpowiedzialne za pobieranie i generowanie ostatecznych datasetów gotowych do uczenia.
* `experiment_notebooks/` - notatniki Jupyter zawierające główne eksperymenty, pętle uczące oraz kod do ewaluacji i porównywania modeli.
