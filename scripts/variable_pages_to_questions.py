import os

import pandas as pd

from src.data.read import unpickle_pages
from src.data.variables import pipeline, split_question_into_parts


def main(base_directory: str, page_nums: list[int]):
    page_directory = os.path.join(base_directory, "pages_raw")
    pages = unpickle_pages(page_directory, page_nums)
    question_strings = pipeline(pages)
    questions = [split_question_into_parts(q) for q in remove_problem_questions(question_strings)]

    questions_df = pd.DataFrame(questions)
    questions_df.to_csv(os.path.join(base_directory, "variables.csv"))


def remove_problem_questions(all_questions: list[str]) -> list[str]:
    # todo: handle problem questions
    # format deviates from that expected in the variables pipeline
    problem_questions = ["Q33", "Q34", "Q35", "Q82", "Q172", "Q223", "Q234"] + [f"Q{i}" for i in range(94, 106)]
    return [q for q in all_questions if not any(x in problem_questions for x in [q[:3], q[:4]])]


if __name__ == "__main__":
    all_page_nums = [*range(10, 84)]
    wd = os.getcwd()  # change as needed
    os.chdir(wd)
    main("../data_files/WV7/variables", all_page_nums)
