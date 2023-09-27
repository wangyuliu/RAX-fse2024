import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# from SourceScanner.settings import SYSCALL_PATH
# from SourceScanner.utils.FileUtil import normal_file_filter, get_all_files, read_file


class SyscallParser:

    def __init__(self, repo_path):
        # from utils.FileUtil import normal_file_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import normal_file_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(normal_file_filter, get_all_files(repo_path)))
        self.x86_syscall_names = self.read_syscall_names()

    def read_syscall_names(self):
        # from settings import SYSCALL_PATH
        from SourceScanner.settings import SYSCALL_PATH
        with open(SYSCALL_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        lines = [l.strip() for l in lines]

        return lines

    def find_syscalls(self, codes):
        pattern = r"\b(" + "|".join(self.x86_syscall_names) + r")\s*\("

        syscall_regex = re.compile(pattern, re.IGNORECASE)
        matches = re.findall(syscall_regex, codes)
        for match in matches:
            codes = re.sub(r'\b' + match + r'\b', "", codes, 1)

        return matches

    def process_file(self, file_path):
        # print(f"[SyscallParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 找Intrinsic函数
        matched = list(self.find_syscalls(file_content))

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

    parser = SyscallParser(repo_path)
    match_syscalls = parser.run()
    print(match_syscalls)
