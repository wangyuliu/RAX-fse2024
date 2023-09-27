from concurrent.futures import ThreadPoolExecutor, as_completed

# from SourceScanner.utils.FileUtil import get_all_files, normal_file_filter, asm_file_filter, extract_asm_blocks, \
#     extract_asm_lines, read_file


class AssemblyParser:

    def __init__(self, repo_path):
        # from utils.FileUtil import normal_file_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import normal_file_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(normal_file_filter, get_all_files(repo_path)))
        # from utils.FileUtil import asm_file_filter
        from SourceScanner.utils.FileUtil import asm_file_filter
        self.asm_file_paths = list(filter(asm_file_filter, get_all_files(repo_path)))

    def extract_asm_block(self, content):
        asm_blocks = ""
        regex = r"[__]*?asm[__]*?\s*[__]*[volatile]*[__]*\s*[(|{]"
        # from utils.FileUtil import extract_asm_blocks
        from SourceScanner.utils.FileUtil import extract_asm_blocks
        asms, content = extract_asm_blocks(regex, content)

        asm_blocks += asms.strip() + "\n"
        return asm_blocks, content

    def extract_asm_line(self, content):
        regex = r"[__]*?asm[__]*?\s*[__]*[volatile]*[__]*\s[{|(]"
        # from utils.FileUtil import extract_asm_lines
        from SourceScanner.utils.FileUtil import extract_asm_lines
        asm_lines = extract_asm_lines(regex, content)

        return asm_lines

    def scan_asm_file(self):
        total_content = ""
        # from utils.FileUtil import read_file
        for file in self.asm_file_paths:
            from SourceScanner.utils.FileUtil import read_file
            content = read_file(file)
            total_content += content + "\n"

        return total_content

    def process_file(self, file_path):
        # print(f"[AsmParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 扫描ASM块
        asm_blocks, file_content = self.extract_asm_block(file_content)

        # 扫描ASM行
        asm_lines = self.extract_asm_line(file_content)

        return asm_blocks, asm_lines

    def scan_asm_inline(self):
        asm_blocks = ""
        asm_lines = ""

        with ThreadPoolExecutor() as executor:
            futures = []
            # submit a task for each file
            for file_path in self.file_paths:
                future = executor.submit(self.process_file, file_path)
                futures.append(future)

            # collect the results as they become available
            for future in as_completed(futures):
                blocks, lines = future.result()
                asm_blocks += blocks.strip() + "\n"
                asm_lines += lines.strip() + "\n"

        # 删除空行
        new_content = ""
        for line in (asm_blocks + asm_lines).split("\n"):
            if len(line.strip()) == 0:
                continue
            new_content += line + "\n"

        return new_content

    def scan_asm_file(self):
        asm_file_content = ""
        # from utils.FileUtil import read_file
        for file in self.asm_file_paths:
            from SourceScanner.utils.FileUtil import read_file
            content = read_file(file)
            asm_file_content += content + "\n"

        return asm_file_content
#     scan 所有后缀为.asm的文件获取他们的所有内容
#       此项统计的是.asm文件的数量？


if __name__ == '__main__':
    repo_path = "/Users/jimto/PycharmProjects/repos/train_repos/mpc/"
    parser = AssemblyParser(repo_path)
    res = parser.scan_asm_inline()
    print(res)


    res = parser.scan_asm_file()
    print(res)
