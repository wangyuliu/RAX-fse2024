import re

from concurrent.futures import ThreadPoolExecutor, as_completed

# from SourceScanner.utils.FileUtil import build_script_filter, get_all_files, read_file


class BuildScriptParser:

    def __init__(self, repo_path):
        # from utils.FileUtil import build_script_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import build_script_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(build_script_filter, get_all_files(repo_path)))

    def find_script(self, codes):
        build_script_pattern = r"\b(x86_[0-9]{0,2}|i[2345678]86|amd64|x86-[0-9]{0,2}|i\[3456789\]86)"
        build_script_regex = re.compile(build_script_pattern, re.IGNORECASE)
        matches = re.findall(build_script_regex, codes)
        for match in matches:
            codes = re.sub(r'\b' + match + r'\b', "", codes, 1)

        return matches

    def process_file(self, file_path):
        # print(f"[BuildScriptParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 找Intrinsic函数
        matched = list(self.find_script(file_content))

        return matched

    def run(self):
        matched_syscalls = []

        with ThreadPoolExecutor() as executor:
            futures = []
            # submit a task for each file
            for file_path in self.file_paths:
                future = executor.submit(self.process_file, file_path)
                futures.append(future)

            # collect the results as they become available
            for future in as_completed(futures):
                matched = future.result()
                matched_syscalls.extend(matched)

        return matched_syscalls


if __name__ == '__main__':
    repo_path = "/Users/jimto/PycharmProjects/repos/gcc"
    parser = BuildScriptParser(repo_path)
    res = parser.run()
    print(res)
