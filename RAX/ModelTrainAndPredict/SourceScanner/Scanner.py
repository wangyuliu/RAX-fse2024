import os

import numpy as np

# from parsers.AssemblyParser import AssemblyParser
# from parsers.BuiltinParser import BuiltinParser
# from parsers.ConditionalCompilingParser import ConditionalCompilingParser
# from parsers.IntrinsicParser import IntrinsicParser


#
# from SourceScanner.parsers.AssemblyParser import AssemblyParser
# from SourceScanner.parsers.BuildScriptParser import BuildScriptParser
# from SourceScanner.parsers.BuiltinParser import BuiltinParser
# from SourceScanner.parsers.ConditionalCompilingParser import ConditionalCompilingParser
# from SourceScanner.parsers.IntrinsicParser import IntrinsicParser
# from SourceScanner.parsers.SyscallParser import SyscallParser


class Scanner:

    def __init__(self, repo_path, repo_name):
        self.repo_path = repo_path
        self.repo_name = repo_name
        # 初始化各类Parser
        # from SourceScanner.parsers.AssemblyParser import AssemblyParser
        from SourceScanner.parsers.AssemblyParser import AssemblyParser
        self.assemblyParser = AssemblyParser(self.repo_path)
        # from parsers.BuildScriptParser import BuildScriptParser
        from SourceScanner.parsers.BuildScriptParser import BuildScriptParser
        self.buildScriptParser = BuildScriptParser(self.repo_path)
        # from SourceScanner.parsers.BuiltinParser import BuiltinParser
        from SourceScanner.parsers.BuiltinParser import BuiltinParser
        self.builtinParser = BuiltinParser(self.repo_path)
        # from SourceScanner.parsers.ConditionalCompilingParser import ConditionalCompilingParser
        from SourceScanner.parsers.ConditionalCompilingParser import ConditionalCompilingParser
        self.conditionalCompilingParser = ConditionalCompilingParser(self.repo_path)
        # from SourceScanner.parsers.IntrinsicParser import IntrinsicParser
        from SourceScanner.parsers.IntrinsicParser import IntrinsicParser
        self.intrinsicParser = IntrinsicParser(self.repo_path)
        # from parsers.SyscallParser import SyscallParser
        from SourceScanner.parsers.SyscallParser import SyscallParser
        self.syscallParser = SyscallParser(self.repo_path)

    def scan(self):
        # 扫描工程的汇编
        print("[汇编扫描]")
        assembly_inline_result = self.assemblyParser.scan_asm_inline()  # 返回字符串
        assembly_in_file_result = self.assemblyParser.scan_asm_file()  # 返回字符串
        print("[汇编扫描] OK")

        # 扫描构建脚本
        print("[构建脚本扫描]")
        build_script_result = self.buildScriptParser.run()  # 返回数组
        print("[构建脚本扫描] OK")

        # 扫描工程的Builtin
        print("[Builtin扫描]")
        builtin_result = self.builtinParser.run()  # 返回数组
        print("[Builtin扫描] OK")

        # 扫描条件编译语句
        print("[条件编译语句扫描]")
        conditional_result = self.conditionalCompilingParser.run()  # 返回数组
        print("[条件编译语句扫描] OK")

        # 扫描工程的Intrinsic
        print("[Intrinsic扫描]")
        intrinsic_result = self.intrinsicParser.run()  # 返回数组
        print("[Intrinsic扫描] OK")

        # 扫描工程的Syscall
        print("[Syscall扫描]")
        syscall_result = self.syscallParser.run()  # 返回数组
        print("[Syscall扫描] OK")

        vec = self.format_result(assembly_inline_result,
                                 assembly_in_file_result,
                                 build_script_result,
                                 builtin_result,
                                 conditional_result,
                                 intrinsic_result,
                                 syscall_result)

        return vec

    def format_result(self, assembly_inline_result,
                      assembly_in_file_result,
                      build_script_result,
                      builtin_result,
                      conditional_result,
                      intrinsic_result,
                      syscall_result):
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

        # 整理成list
        lengths = [assembly_in_file_len, assembly_inline_len, build_script_len,
                   builtin_len, conditional_len, intrinsic_len, syscall_len]
        # 将列表转换成numpy向量
        vector = np.array(lengths)

        return vector

    def filter_empty_in_list(self, item: str):
        if len(item) == 0:
            return False
        return True
