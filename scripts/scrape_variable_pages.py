import os

from src.data.read import pdf_to_strings, pickle_pages


def main(base_directory: str, page_nums: list[int]):
    """
    Scrape variables (questions, responses and metadata) from the WV7 PDF codebook and pickle each page.
    """
    pages = pdf_to_strings(base_directory, page_nums)
    page_directory = os.path.join(base_directory, "variables/pages_raw")
    pickle_pages(page_directory, pages)


if __name__ == "__main__":
    all_page_nums = [*range(10, 84)]
    wd = os.getcwd()  # change as needed
    os.chdir(wd)
    main("../data_files/WV7", all_page_nums)
