import sys

from parsers.BuiltinParser import BuiltinParser


# from SourceScanner.missions import sub_dirs, root_dir
# from SourceScanner.parsers.AssemblyParser import AssemblyParser
# from SourceScanner.parsers.BuildScriptParser import BuildScriptParser
# from SourceScanner.parsers.BuiltinParser import BuiltinParser
# from SourceScanner.parsers.ConditionalCompilingParser import ConditionalCompilingParser
# from SourceScanner.parsers.IntrinsicParser import IntrinsicParser
# from SourceScanner.parsers.SyscallParser import SyscallParser
# from SourceScanner.settings import SAVE_PATH_1, SAVE_PATH_2


class Scanner:

    def __init__(self, repo_path, repo_name):
        self.repo_path = repo_path
        self.repo_name = repo_name
        # 初始化各类Parser
        from parsers.AssemblyParser import AssemblyParser
        self.assemblyParser = AssemblyParser(self.repo_path)
        from parsers.BuildScriptParser import BuildScriptParser
        self.buildScriptParser = BuildScriptParser(self.repo_path)
        self.builtinParser = BuiltinParser(self.repo_path)
        from parsers.ConditionalCompilingParser import ConditionalCompilingParser
        self.conditionalCompilingParser = ConditionalCompilingParser(self.repo_path)
        from parsers.IntrinsicParser import IntrinsicParser
        self.intrinsicParser = IntrinsicParser(self.repo_path)
        from parsers.SyscallParser import SyscallParser
        self.syscallParser = SyscallParser(self.repo_path)

    def scan(self):
        # 扫描工程的汇编
        print("[扫描工程汇编]")
        assembly_inline_result = self.assemblyParser.scan_asm_inline()  # 返回字符串
        assembly_in_file_result = self.assemblyParser.scan_asm_file()  # 返回字符串

        # 扫描构建脚本
        print("[扫描构建脚本]")
        build_script_result = self.buildScriptParser.run()  # 返回数组

        # 扫描工程的Builtin
        print("[扫描工程Builtin]")
        builtin_result = self.builtinParser.run()  # 返回数组

        # 扫描条件编译语句
        print("[扫描条件编译语句]")
        conditional_result = self.conditionalCompilingParser.run()  # 返回数组

        # 扫描工程的Intrinsic
        print("[扫描工程的Intrinsic]")
        intrinsic_result = self.intrinsicParser.run()  # 返回数组

        # 扫描工程的Syscall
        print("[扫描工程的Syscall]")
        syscall_result = self.syscallParser.run()  # 返回数组

        # 保存结果
        self.save_as_vector(assembly_inline_result,
                            assembly_in_file_result,
                            build_script_result,
                            builtin_result,
                            conditional_result,
                            intrinsic_result,
                            syscall_result)

        self.save_as_text(assembly_inline_result,
                          assembly_in_file_result,
                          build_script_result,
                          builtin_result,
                          conditional_result,
                          intrinsic_result,
                          syscall_result)

    def filter_empty_in_list(self, item: str):
        if len(item) == 0:
            return False
        return True

    def save_as_vector(self, assembly_inline_result: str,
                       assembly_in_file_result: str,
                       build_script_result: list,
                       builtin_result: list,
                       conditional_result: list,
                       intrinsic_result: list,
                       syscall_result: list):
        # 把汇编处理成list
        assembly_inline_result = list(filter(self.filter_empty_in_list, assembly_inline_result.split("\n")))
        assembly_in_file_result = list(filter(self.filter_empty_in_list, assembly_in_file_result.split("\n")))

        # 所有内容计算list长度
        assembly_in_file_len = len(assembly_in_file_result)
        assembly_inline_len = len(assembly_inline_result)
        build_script_len = len(build_script_result)
        builtin_len = len(builtin_result)
        conditional_len = len(conditional_result)
        intrinsic_len = len(intrinsic_result)
        syscall_len = len(syscall_result)

        from settings import SAVE_PATH_1
        with open(SAVE_PATH_1, 'a', encoding='utf-8') as f:
            f.write(
                f"{self.repo_name},{assembly_in_file_len},{assembly_inline_len},"
                f"{build_script_len},{builtin_len},{conditional_len},{intrinsic_len},{syscall_len}\n")

    def convert_to_str(self, mlist: list):
        result_str = ""
        for i in mlist:
            result_str += str(i)
        return result_str

    def save_as_text(self, assembly_inline_result: str,
                     assembly_in_file_result: str,
                     build_script_result: list,
                     builtin_result: list,
                     conditional_result: list,
                     intrinsic_result: list,
                     syscall_result: list):
        # 把所有list的入参做成字符串
        build_script_result = self.convert_to_str(build_script_result)
        builtin_result = self.convert_to_str(builtin_result)
        conditional_result = self.convert_to_str(conditional_result)
        intrinsic_result = self.convert_to_str(intrinsic_result)
        syscall_result = self.convert_to_str(syscall_result)

        # 写入文本
        # 这里直接写入可能会有问题，导致训练的文本很大，这一步需要配合文本预处理步骤做比较好，但是现在其他步骤没到位，先这样处理
        from settings import SAVE_PATH_2
        with open(SAVE_PATH_2, 'a', encoding='utf-8') as f:
            # 因为文本里可能就带了逗号，再用逗号来存向量会出错，这里直接用中文分割
            f.write(
                f"{self.repo_name}逗号{assembly_inline_result}逗号{assembly_in_file_result}逗号"
                f"{build_script_result}逗号{builtin_result}逗号"
                f"{conditional_result}逗号{intrinsic_result}逗号{syscall_result}\n")


if __name__ == '__main__':
    from missions import sub_dirs, root_dir
    repo_paths = [root_dir + sub_dir + '/' for sub_dir in sub_dirs]

    for repo_path in repo_paths:
        repo_name = repo_path.split('/')[-2]
        scanner = Scanner(repo_path, repo_name)
        result=scanner.scan()
        # if result == "0":
        #   print(f"工程{repo_name}移植复杂度：低")
        # elif result == "1":
        #   print(f"工程{repo_name}移植复杂度：中")
        # elif result == "2":
        #   print(f"工程{repo_name}移植复杂度：高")

    # repo_path, repo_name = sys.argv[1], sys.argv[2]
    # scanner = Scanner(repo_path, repo_name)
    # result = scanner.scan()
    #
    # if result == "0":
    #     print(f"工程{repo_name}移植复杂度：低")
    # elif result == "1":
    #     print(f"工程{repo_name}移植复杂度：中")
    # elif result == "2":
    #     print(f"工程{repo_name}移植复杂度：高")
