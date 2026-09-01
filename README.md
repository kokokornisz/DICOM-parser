# System Przetwarzania Danych Medycznych (DICOM) w Microsoft Azure

## Opis projektu
Projekt wdraża zautomatyzowany mechanizm przetwarzania danych medycznych zgodnych ze standardem DICOM w środowisku chmury obliczeniowej Microsoft Azure. Skrypty działające na maszynie wirtualnej z systemem Linux wyodrębniają metadane z plików DICOM do formatu CSV oraz generują podgląd zdjęć (JPEG), a następnie automatycznie przesyłają je do usługi Azure Blob Storage.

## Architektura i działanie
Rozwiązanie opiera się na modelu IaaS (Infrastruktura jako usługa) z wykorzystaniem maszyny wirtualnej (Ubuntu) oraz konta magazynu (Azure Storage Account) w chmurze Azure.
Proces działania jest zautomatyzowany dzięki narzędziu systemowemu `crontab`, które w równych odstępach czasu uruchamia odpowiednie skrypty w środowisku Python.

## Struktura plików i katalogów
Aby system działał poprawnie, wymagana jest następująca struktura katalogów roboczych (domyślnie `/home/azureuser/Documents/dcm-to-csv/`):
* `raw-dcm/` - katalog wejściowy przechowujący surowe pliki medyczne `.dcm`.
* `meta-csv/` - katalog wyjściowy na wygenerowane pliki `.csv`.
* `photo/` - katalog wyjściowy na wygenerowane pliki obrazów `.jpg`.
* `logs/` - katalog przechowujący logi działania skryptów.

## Opis plików źródłowych
1. **`parse_dicom.py`**
   * Przeszukuje katalog wejściowy w poszukiwaniu nowych plików `.dcm`.
   * Ekstrahuje metadane (tagi DICOM) do formatu płaskiego pliku tabelarycznego `.csv`.
   * Wyodrębnia pierwszą klatkę obrazu z pliku DICOM i konwertuje ją do przestrzeni barw RGB lub skali szarości, zapisując jako plik `.jpg`.
   * Zapisuje wyniki do odpowiednich katalogów wyjściowych.

2. **`csv_upload.py`**
   * Przeszukuje katalog `meta-csv` w poszukiwaniu gotowych plików z metadanymi.
   * Nawiązuje połączenie z Azure Blob Storage za pomocą ciągu połączenia (Connection String).
   * Weryfikuje, czy plik już istnieje w chmurze (zapobiega to ponownemu przesyłaniu).
   * Wgrywa brakujące pliki do kontenera `meta-csv` w Azure.

3. **`jpg_upload.py`**
   * Działa analogicznie do skryptu przesyłającego pliki CSV.
   * Pobiera przekonwertowane zdjęcia medyczne `.jpg` z katalogu `photo/`.
   * Przesyła nowe pliki do dedykowanego kontenera `photo` w usłudze Azure Blob Storage.

4. **`crontab.txt`**
   * Zawiera wpisy harmonogramu zadań systemu Linux (`crontab`).
   * Uruchamia `parse_dicom.py` co 5 minut.
   * Uruchamia skrypty przesyłające `csv_upload.py` i `jpg_upload.py` co 6 minut, zapewniając minimalne opóźnienie potrzebne na ukończenie procesu ekstrakcji danych.
   * Przekierowuje wyjście programów oraz błędy (strumienie standardowe) do plików logów w folderze `logs/`.

## Wymagania i instalacja
* Zainstalowany Python 3.
* Wymagane pakiety Python: `pydicom`, `azure-storage-blob`, `Pillow`, `numpy`.
* Skonfigurowana zmienna środowiskowa `AZURE_STORAGE_CONNECTION_STRING` z ciągiem połączeniowym do konta magazynu Azure, posiadającym dostęp do kontenerów `meta-csv` i `photo`.

## Bezpieczeństwo i regulacje
Projekt opierał się o rygorystyczne wymagania przetwarzania wrażliwych danych medycznych, wykorzystując mechanizmy zgodne z normą ISO 27001 oraz RODO:
* **SSE (Server-Side Encryption):** Szyfrowanie po stronie serwera w Azure Storage gwarantujące ochronę w chmurze.
* **Azure Storage Firewall:** Ograniczenie połączeń przychodzących do kontenera tylko dla zaufanych adresów sieciowych.
* **Prawa dostępu Linux:** Ścisła kontrola uprawnień odczytu i zapisu na maszynie wirtualnej (chmod).
* **Tokeny SAS:** Umożliwiają wygenerowanie tymczasowego, bezpiecznego linku z określonymi prawami dostępu do danych dla personelu medycznego.
