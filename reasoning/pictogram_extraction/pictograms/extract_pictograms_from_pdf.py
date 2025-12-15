import fitz  # PyMuPDF
import os
from datetime import datetime, timedelta
from glob import glob

def get_next_date_string(date_str: str):
    """
    Takes a date string in "YYYY_MM_DD" format and returns the next date 
    in the same format.
    """
    try:
        date_format = "%Y_%m_%d"
        current_date = datetime.strptime(date_str, date_format)
    except ValueError as e:
        print(f"Error: Input date format must be YYYY_MM_DD. Error: {e}")
        return ""
    
    next_date = current_date + timedelta(days=1)
    return next_date.strftime(date_format)

def extract_third_image_from_folder(pdf_folder: str):
    """
    Extracts the third image from the first page of all PDFs in a folder,
    saves it using the next date in the filename, and prints year and month.
    """

    pdf_files = glob(os.path.join(pdf_folder, "*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {pdf_folder}")

    for pdf_path in pdf_files:
        base_name = os.path.basename(pdf_path)
        date_str = os.path.splitext(base_name)[0]  # "YYYY_MM_DD"
        
        # Extract year and month

        yyyy, mm, dd = date_str.split("_")
        print(f"Processing {base_name}: Year={yyyy}, Month={mm}")
        
        next_date_str = get_next_date_string(date_str)
        if not next_date_str:
            continue

        doc = fitz.open(pdf_path)
        page = doc[0]  # first page
        image_list = page.get_images(full=True)
        print(f"  Found {len(image_list)} images on page 1")

        if len(image_list) >= 3:
            xref = image_list[2][0]  # third image (index 2)
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]

            if not os.path.exists(f"../extracted_pictograms/{yyyy}/{mm}/"):
                os.makedirs(f"../extracted_pictograms/{yyyy}/{mm}/")
            
            filename = f"../extracted_pictograms/{yyyy}/{mm}/{next_date_str}.{ext}"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"  Saved third image as {filename}")
        else:
            print(f"  Less than 3 images on page 1 of {base_name}")

# Example usage
def extract_year():
    for mm in ["01","02","03","04","05","06","07","08","09", "10", "11", "12"]:
        extract_third_image_from_folder(pdf_folder=f"../pdfs/2019/{mm}")


# Example usage
if __name__ == "__main__":
    extract_year()