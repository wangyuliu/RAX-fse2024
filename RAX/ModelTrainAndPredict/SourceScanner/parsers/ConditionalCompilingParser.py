import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# from SourceScanner.utils.FileUtil import normal_file_filter, get_all_files, read_file, read_macros


class ConditionalCompilingParser:

    def __init__(self, repo_path):
        # from utils.FileUtil import normal_file_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import normal_file_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(normal_file_filter, get_all_files(repo_path)))
        # from utils.FileUtil import read_macros
        from SourceScanner.utils.FileUtil import read_macros
        self.x86_macros = read_macros()

    def replace_keyword_str(self, text: str, start_index: int, end_index: int):
        text = text.replace(text[start_index:end_index], "*" * (end_index - start_index), 1)
        return text

    def is_start_pattern(self, text: str):
        start_pattern = re.compile(r'#\s*(?:if defined|ifdef|ifndef|if)\b', re.IGNORECASE)
        if re.search(start_pattern, text) is not None:
            return True
        return False

    def replace_keyword_str(self, text: str, start_index: int, end_index: int):
        text = text.replace(text[start_index:end_index], "*" * (end_index - start_index), 1)
        return text

    def is_end_pattern(self, text: str):
        end_pattern = re.compile(r'#\s*endif', re.IGNORECASE)
        if re.search(end_pattern, text) is not None:
            return True
        return False

    def find_conditions(self, text):
        codes = []
        text_for_locate = text
        start_pattern = re.compile(r'#\s*(?:if defined|ifdef|ifndef|if)\b', re.IGNORECASE)
        all_pattern = re.compile(r'#\s*(?:if defined|ifdef|ifndef|if|endif)\b', re.IGNORECASE)
        while re.search(start_pattern, text_for_locate) != None:
            # 搜索起始关键词
            match = re.search(start_pattern, text_for_locate)
            # 关键词起始、终止位置
            start_index = int(match.start())
            end_index = int(match.end())

            # 创建代码段栈
            location_stack = [start_index]
            # 把前面定位到的代码段过滤
            text_for_locate = self.replace_keyword_str(text_for_locate, start_index, end_index)

            # 按理来说，这是一个括号匹配问题，栈为空之前必然会继续出现右括号，但是可能出现错误，这时直接返回之前扫描的结果
            try:
                while len(location_stack) != 0:
                    # 判断是否有关键词
                    match = re.search(all_pattern, text_for_locate)
                    match_str = text_for_locate[int(match.start()): int(match.end())]
                    # 若关键词为start关键词，继续压栈，替换文本
                    if self.is_start_pattern(match_str):
                        location_stack.append(int(match.start()))
                        text_for_locate = self.replace_keyword_str(text_for_locate, int(match.start()),
                                                                   int(match.end()))
                    elif self.is_end_pattern(match_str):
                        # 若为end关键词，弹栈，从原始文本里提取出来
                        match_start_index = location_stack.pop()
                        match_end_index = int(match.end())
                        text_for_locate = self.replace_keyword_str(text_for_locate, match_start_index, match_end_index)
                        code = text[match_start_index: match_end_index]
                        codes.append(code)
            except AttributeError as e:
                # print(e)
                return codes
        return codes

    def process_file(self, file_path):
        # print(f"[ConditionalCompilingParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 找Intrinsic函数
        matched = list(self.find_conditions(file_content))

        return matched

    def remove_duplicate_conditions(self, conditions: list):
        sorted(conditions, key=len, reverse=True)
        # 用set去重一遍
        conditions = list(set(conditions))
        filtered_conditions = []
        # 先把小的从大的里去掉
        for big in conditions:
            for small in conditions:
                if big == small:
                    filtered_conditions.append(big)
                elif small in big:
                    big = big.replace(small, "")

            if big not in filtered_conditions:
                filtered_conditions.append(big)

        filtered_conditions = list(set(filtered_conditions))

        return filtered_conditions

    def is_x86_relate(self, conditional_content):
        if_elif_pattern = re.compile(r'#\s*(?:if defined|ifdef|ifndef|if|elif|elif defined)\b', re.IGNORECASE)
        macro_pattern = re.compile("(" + '|'.join(self.x86_macros) + ")", re.IGNORECASE)

        # 按行切分
        lines = conditional_content.split("\n")
        # 过滤出if和elif
        for line in lines:
            if re.search(if_elif_pattern, line) != None:
                if re.search(macro_pattern, line) != None:
                    return True

        return False

    def filter_x86_relate_conditions(self, conditions: list):
        new_conditions = []
        for con in conditions:
            if self.is_x86_relate(con):
                new_conditions.append(con)

        return new_conditions

    def run(self):
        # 摘取所有条件编译语句
        matched_conditions = []
        with ThreadPoolExecutor() as executor:
            futures = []
            # submit a task for each file
            for file_path in self.file_paths:
                future = executor.submit(self.process_file, file_path)
                futures.append(future)

            # collect the results as they become available
            for future in as_completed(futures):
                matched = future.result()
                matched_conditions.extend(matched)

        # 先按照架构宏过滤一遍，减少搜索空间
        conditions = self.filter_x86_relate_conditions(matched_conditions)

        # 按重复内容过滤嵌套问题
        conditions = self.remove_duplicate_conditions(conditions)

        # 按架构宏过滤条件编译语句
        conditions = self.filter_x86_relate_conditions(conditions)

        return conditions


if __name__ == '__main__':
    repo_path = "/Users/jimto/PycharmProjects/repos/gcc"

    parser = ConditionalCompilingParser(repo_path)
    res = parser.run()

    # 保存一份结果，写论文时作为材料
    # with open("../data/other/ConditionalCompilingCases.txt", 'w', encoding='utf-8') as f:
    #     for i in res:
    #         f.write(i + '\n')
    #         f.write("*" * 30 + '\n')
