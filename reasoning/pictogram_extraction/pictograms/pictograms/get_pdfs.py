import requests
import re
import os

from calendar import monthrange

BASE_URL = "https://www.osmer.fvg.it/ajax/getPrevisArchive.php"
PDF_BASE = "https://www.osmer.fvg.it/"

def fetch_previs_and_pdf(day, month, year):
    params = {
        "a": year,
        "m": month,
        "g": day,
        "z": "regione",
        "l": "it",
        "ln": ""
    }

    #print("[*] Requesting:", BASE_URL, params)
    r = requests.get(BASE_URL, params=params)

    if r.status_code != 200:
        #print("[!] Error:", r.status_code)
        return

    html = r.text
    #print("\n=== Raw server response (trimmed) ===")
    #print(html[:300] + "...\n")

    # -------------------------
    # Remove escaped quotes and slashes for easier regex
    # -------------------------
    clean_html = html.replace("\\/", "/").replace('\\"', '"')

    # Regex: find the last PDF timestamp in the HTML
    # pattern: /pdf/2025/20250503/202505031118/pdf/regione-20250503-it.pdf
    matches = re.findall(r"pdf/\d{4}/\d{8}/(\d{12})/pdf/regione-\d{8}-it\.pdf", clean_html)
    print(clean_html)
    if not matches:
        print("[!] Could not extract timestamp.")
        return

    # Take the last one (the active/latest link)
    timestamp = matches[-1]
    #print("[+] Extracted timestamp:", timestamp)

    yyyymmdd = f"{year:04d}{month:02d}{day:02d}"

    yyyy = f"{year:04d}"
    mm = f"{month:02d}"
    dd = f"{day:02d}"
    pdf_url = f"{PDF_BASE}pdf/{year}/{yyyymmdd}/{timestamp}/pdf/regione-{yyyymmdd}-it.pdf"
    #print("[*] PDF URL:", pdf_url)

    # -------------------------------
    # Create folder if it doesn't exist
    # -------------------------------
    if not os.path.exists(f"../pdfs/{yyyy}/{mm}"):
        os.makedirs(f"../pdfs/{yyyy}/{mm}")
    
    # Download PDF

    pdf_filename = f"../pdfs/{yyyy}/{mm}/{yyyy}_{mm}_{dd}.pdf"
    pdf_data = requests.get(pdf_url)

    if pdf_data.status_code == 200:
        with open(pdf_filename, "wb") as f:
            f.write(pdf_data.content)
        print("[+] PDF saved successfully as", pdf_filename)
    else:
        print("[!] PDF download failed, status:", pdf_data.status_code)



def download_month(year, month):
    """
    Download all available PDFs for a given month
    """
    num_days = monthrange(year, month)[1]
    for day in range(1, num_days + 1):
        try:
            fetch_previs_and_pdf(day, month, year)
        except Exception as e:
            print(f"[!] Error on {day:02d}/{month:02d}/{year}: {e}")

def donwnload_year(yyyy):
    for mm in range (1,13):
        print(f"downloading month {mm}")
        download_month(yyyy,mm)

# Example usage
if __name__ == "__main__":
    donwnload_year(2019)
