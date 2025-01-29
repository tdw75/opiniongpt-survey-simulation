import os
import pickle

import PyPDF2


def pdf_to_strings(directory: str, page_nums: list[int] = None) -> dict[int, str]:
    file_name = "WVS7_Codebook_Variables_report_V6.0.pdf"

    with open(os.path.join(directory, file_name), 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        page_nums = page_nums or range(1, len(reader.pages))
        pages = {i: reader.pages[i-1].extract_text() for i in page_nums}

    return pages


def pickle_pages(directory: str, pages: dict[int, str]) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory)
    for n, text in pages.items():
        with open(os.path.join(directory, f"page{n}.pkl"), "wb") as f:
            pickle.dump(text, f)


def unpickle_pages(directory: str, page_nums: list[int]) -> dict[int, str]:
    pages = {}
    file_names = {n: f"page{n}.pkl" for n in page_nums}
    for n, name in file_names.items():
        with open(os.path.join(directory, name), "rb") as f:
            pages[n] = pickle.load(f)
    return pages
