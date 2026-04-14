#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Przetwarza pliki DICOM z katalogu i zapisuje metadane do plików CSV.

Domyślne ścieżki można dostosować za pomocą zmiennych środowiskowych
  INPUT_DICOM_DIR  – katalog wejściowy z plikami .dcm
  OUTPUT_CSV_DIR   – katalog wyjściowy na wygenerowane .csv
  DCM_TO_CSV_ROOT  – katalog bazowy (jeśli nie ustawione, skrypt wybierze ~/Documents/dcm-to-csv lub ./dcm-to-csv)
"""

import pydicom                                                     # biblioteka do odczytu plików DICOM
import csv                                                         # do zapisywania wyników w formacie CSV
import logging                                                     # do logowania komunikatów o postępie i błędach
import os                                                          # do obsługi zmiennych środowiskowych
from pathlib import Path                                           # wygodna obsługa ścieżek plików/katalogów
from typing import Dict, Tuple                                     # adnotacje typów: słownik i krotka
from pydicom.pixel_data_handlers.util import convert_color_space   # do konwersji tablicy pikseli
from PIL import Image                                              # 

# 1) USTAWIENIE KATALOGU BAZOWEGO
#    Można go nadpisać przez DCM_TO_CSV_ROOT;
#    jeśli nie ma, wybiera ~/Documents/dcm-to-csv (jeśli istnieje)
#    lub ./dcm-to-csv
root_env = os.environ.get("DCM_TO_CSV_ROOT", "")
if root_env:
    # Jeśli użytkownik podał ścieżkę przez zmienną, zamień na Path
    DEFAULT_ROOT = Path(root_env)
else:
    # Jeśli nie podał – ustaw na domyślny katalog w Documents lub bieżącym
    home = Path.home()
    base = home / "Documents" if (home / "Documents").exists() else Path.cwd()
    DEFAULT_ROOT = base / "dcm-to-csv"

# 2) ŚCIEŻKI WEJŚCIOWE i WYJŚCIOWE
#    Domyślnie raw-dcm do meta-csv w katalogu DEFAULT_ROOT,
#    ale można je nadpisać przez INPUT_DICOM_DIR / OUTPUT_CSV_DIR.
DEFAULT_INPUT_DIR = Path(
    os.environ.get("INPUT_DICOM_DIR", str(DEFAULT_ROOT / "raw-dcm"))
)
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_CSV_DIR", str(DEFAULT_ROOT / "meta-csv"))
)


def parse_dicom_file(path: Path) -> Tuple[Dict[str, str], pydicom.FileDataset]:
    """
    Odczytuje pojedynczy plik DICOM i zwraca:
      - słownik metadanych {tag_keyword: wartość}
      - obiekt FileDataset z pełnymi danymi (na wypadek potrzeby dalszej analizy)
    """
    # 1) wczytaj plik jako obiekt DICOM
    dcm = pydicom.dcmread(str(path))
    output: Dict[str, str] = {}

    # 2) iteruj po wszystkich elementach metadanych
    for elem in dcm:
        key = elem.keyword                  # unikalne nazwy dla keyword'u (np. "PatientID", "StudyDate" itd.)
        if not key or key == "PixelData":
            # pomijamy pola bez keyword'u oraz surowe dane obrazu
            continue
        try:
            output[key] = str(elem.value)   # wartość zamieniamy na string
        except Exception:
            # niektóre typy (obrazy, sekwencje) mogą nie być serializowalne -
            # wartości niekonwertowalne pomijamy
            continue

    # 3) zwróć słownik i oryginalny obiekt DICOM
    return output, dcm


def save_first_frame(ds: pydicom.FileDataset, output_path: Path) -> None:
    """Zapisuje pierwszą klatkę danych obrazowych DICOM jako JPEG."""
    try:
        px = ds.pixel_array
    except Exception as exc:
        logging.error("Nie można odczytać danych pikselowych: %s", exc)
        return

    # Obsługa wieloklatkowych tablic - wybierz pierwszą klatkę
    if px.ndim == 4:  # np. (klatki, wiersze, kolumny, kanały)
        px = px[0]
    elif px.ndim == 3 and px.shape[0] > 1 and px.shape[-1] != 3:
        px = px[0]

    # Konwersja przestrzeni barw na RGB, jeżeli potrzebne
    try:
        interp = ds.get("PhotometricInterpretation", "")
        if interp.startswith("YBR"):
            px = convert_color_space(px, interp, "RGB")
        elif interp == "MONOCHROME1":
            px = px.max() - px
    except Exception as exc:  # pragma: no cover - ostrożnościowo
        logging.debug("Błąd konwersji kolorów: %s", exc)

    if px.ndim == 2:
        img = Image.fromarray(px)
    elif px.ndim == 3 and px.shape[-1] in {1, 3}:
        if px.shape[-1] == 1:
            img = Image.fromarray(px[:, :, 0])
        else:
            img = Image.fromarray(px)
    else:
        logging.error("Nieobsługiwany kształt obrazu: %s", px.shape)
        return

    if img.mode not in {"RGB", "L"}:
        img = img.convert("L")

    img.save(output_path, format="JPEG")


def process_directory(
    input_dir: Path,
    output_dir: Path,
    photo_dir: Path | None = None,
    extension: str = "dcm",
    recursive: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Przeszukuje `input_dir` w poszukiwaniu plików z rozszerzeniem `extension`
    i dla każdego generuje CSV z metadanymi w `output_dir`.
    """
    # utwórz katalog wyjściowy (rekurencyjnie, jeśli trzeba)
    output_dir.mkdir(parents=True, exist_ok=True)
    if photo_dir is not None:
        photo_dir.mkdir(parents=True, exist_ok=True)
    # przygotuj wzorzec wyszukiwania
    pattern = f"**/*.{extension}" if recursive else f"*.{extension}"

    # dla każdego pliku DICOM
    for dcm_file in input_dir.glob(pattern):
        if not dcm_file.is_file():
            # np. katalogi, dowiązania itp. - pomijamy
            continue

        # docelowy plik .csv o tej samej nazwie
        output_file = output_dir / f"{dcm_file.stem}.csv"
        photo_file: Path | None = None
        if photo_dir is not None:
            photo_file = photo_dir / f"{dcm_file.stem}.jpg"

        if output_file.exists() and not overwrite:
            logging.info("Pominięto %s (już przetworzone)", dcm_file)
            if photo_file is not None and not photo_file.exists():
                try:
                    _, ds = parse_dicom_file(dcm_file)
                    save_first_frame(ds, photo_file)
                except Exception as exc:
                    logging.error(
                        "Błąd podczas tworzenia zdjęcia %s: %s", photo_file, exc
                    )
            continue

        try:
            # ekstrahuj metadane
            data, ds = parse_dicom_file(dcm_file)

            # zapisz do pliku CSV
            with open(output_file, "w", newline="") as fh:
                writer = csv.writer(fh)
                # nagłówek
                writer.writerow(["Tag", "Value"])
                # każdy wiersz: klucz na wartość
                for key, value in data.items():
                    writer.writerow([key, value])

            logging.info("Utworzono %s", output_file)

            if photo_dir is not None:
                photo_file = photo_dir / f"{dcm_file.stem}.jpg"
                if not photo_file.exists() or overwrite:
                    save_first_frame(ds, photo_file)
                else:
                    logging.info("Pominięto %s (zdjęcie już istnieje)", photo_file)

        except Exception as exc:
            # złap i zaloguj każdy błąd, by nie przerywać całej pętli
            logging.error("Błąd podczas przetwarzania %s: %s", dcm_file, exc)


def main() -> None:
    """
    Punkt wejścia skryptu:
      - statyczne ścieżki do katalogów w systemie Linux
      - wywołuje process_directory z domyślnymi ustawieniami
    """
    # Ścieżki do katalogów (Linux)
    input_dir = Path("/home/azureuser/Documents/dcm-to-csv/raw-dcm")
    output_dir = Path("/home/azureuser/Documents/dcm-to-csv/meta-csv")
    photo_dir = Path("/home/azureuser/Documents/dcm-to-csv/photo")

    # Inicjalizacja prostego loggera
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s"
    )

    # Uruchomienie przetwarzania (domyślnie extension="dcm", recursive=False, overwrite=False)
    process_directory(
        input_dir=input_dir,
        output_dir=output_dir,
        photo_dir=photo_dir,
    )


if __name__ == "__main__":
    # uruchomienie skryptu
    main()
