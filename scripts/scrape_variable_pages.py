import os

from src.data.read import pdf_to_strings, pickle_pages


def main(base_directory: str, page_nums: list[int]):
    pages = pdf_to_strings(base_directory, page_nums)
    page_directory = os.path.join(base_directory, "variables/pages_raw")
    pickle_pages(page_directory, pages)


if __name__ == "__main__":
    all_page_nums = [*range(10, 84)]
    wd = os.getcwd()  # change as needed
    os.chdir(wd)
    main("../data_files/WV7", all_page_nums)
