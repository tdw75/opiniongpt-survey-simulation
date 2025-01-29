from src.data.read import unpickle_pages


def test_unpickle_pages():
    page_nums = [10, 11, 12]
    pages = unpickle_pages("test_data_files/pages", page_nums)

    assert list(pages.keys()) == page_nums
    assert "Q2 Important in life: Friends" in pages[10]
    assert "Q2 Important in life: Friends" not in pages[11]
    assert "Q2 Important in life: Friends" not in pages[12]

    assert "Q6 Important in life: Religion" not in pages[10]
    assert "Q6 Important in life: Religion" in pages[11]
    assert "Q6 Important in life: Religion" not in pages[12]

    assert "Q13 Important child qualities: Thrift saving money and things" not in pages[10]
    assert "Q13 Important child qualities: Thrift saving money and things" not in pages[11]
    assert "Q13 Important child qualities: Thrift saving money and things" in pages[12]

    assert pages[10].endswith("-5-.- Missing; Not available  \n ")
    assert pages[11].endswith("-2-.- No answer  ")
    assert pages[12].endswith("-5-.- Missing; Not available  \n \n ")