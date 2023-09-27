import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# from utils.FileUtil import read_builtin_names, divide_list


# from SourceScanner.utils.FileUtil import normal_file_filter, get_all_files, read_file, divide_list, read_builtin_names


class BuiltinParser:

    def __init__(self, repo_path):
        # from utils.FileUtil import normal_file_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import normal_file_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(normal_file_filter, get_all_files(repo_path)))
        from SourceScanner.utils.FileUtil import read_builtin_names
        self.x86_builtin_names = read_builtin_names()

    def find_builtin_functions(self, codes: str):
        from SourceScanner.utils.FileUtil import divide_list
        builtin_list = divide_list(self.x86_builtin_names, 100)
        final_matches = []
        for builtins in builtin_list:
            pattern = r"\b(" + "|".join(builtins) + r")\b"
            builtin_regex = re.compile(pattern, re.IGNORECASE)
            matches = re.findall(builtin_regex, codes)
            for match in matches:
                codes = re.sub(r'\b' + match + r'\b', "", codes, 1)
            final_matches += matches
        return final_matches

    def process_file(self, file_path):
        # print(f"[BuiltinParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 找Intrinsic函数
        matched = list(self.find_builtin_functions(file_content))

        return matched

    def run(self):
        matched_builtins = []

        with ThreadPoolExecutor() as executor:
            futures = []
            # submit a task for each file
            for file_path in self.file_paths:
                future = executor.submit(self.process_file, file_path)
                futures.append(future)

            # collect the results as they become available
            for future in as_completed(futures):
                matched = future.result()
                matched_builtins.extend(matched)

        return matched_builtins


if __name__ == '__main__':
    repo_path = "/Users/jimto/PycharmProjects/repos/gcc"

    parser = BuiltinParser(repo_path)
    res = parser.run()
    print(res)
