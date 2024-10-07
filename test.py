import glob
from pypdf import PdfReader

files = glob.glob("*.pdf")

for file in files:
    reader = PdfReader(file)

    for i, page in enumerate(reader.pages):
        for image in page.images:
            with open(image.name, "wb") as fp:
                fp.write(image.data)
