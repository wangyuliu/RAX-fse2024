import codecs
import json
import os
import re

# from SourceScanner.settings import BUILTIN_PATH, X86_MACROS


def extract_intrinsic_names(intrinsics, arch):
    x86_intrinsic_root = intrinsics[arch]
    x86_intrinsic = set()
    for k in x86_intrinsic_root:
        inner_intrins = x86_intrinsic_root[k]

        for i in inner_intrins:
            x86_intrinsic.add(i['Name'])

    return x86_intrinsic


def build_script_filter(file_path: str):
    ends = ['configure', '.sub', 'configure.ac', '.in', '.m4']
    flag = False
    for end in ends:
        if file_path.endswith(end):
            flag = True

    return flag


def read_intrinsic_name(json_path, arch):
    intrin_json = read_json_file(json_path)
    intrinsic_names = extract_intrinsic_names(intrin_json, arch)
    return intrinsic_names


def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def normal_file_filter(file_path: str):
    ends = ['.c', '.cpp', '.cxx', '.cc', '.h', '.hpp']
    flag = False
    for end in ends:
        if file_path.endswith(end):
            flag = True

    return flag


def extract_asm_blocks(regex, content):
    asm_blocks = ""
    # regex = r"[__]*?asm[__]*?\s*[__]*[volatile]*[__]*\s*[(|{]"

    regex = re.compile(regex, re.IGNORECASE)
    while re.search(regex, content) != None:
        search_res = re.search(regex, content)
        start = search_res.start()
        end = search_res.end()
        bracket = "(" if content[start:end][-1] == "(" else "{"
        end_bracket = ")" if bracket == "(" else "}"

        match_str = ""
        try:
            bracket_stack = [bracket]
            while len(bracket_stack) != 0:
                end += 1
                char = content[end]
                match_str += char
                if char == bracket:
                    bracket_stack.append(bracket)
                if char == end_bracket:
                    bracket_stack.pop()

            matched_content = content[start: end + 1]
            # print(matched_content)
            content = content.replace(matched_content, "", 1)

            asm_blocks += matched_content.strip() + "\n"
        except IndexError as ie:
            # print(ie)
            # 若括号匹配不成功，就把开头3个字符砍掉
            # content = content.replace(content[start:start + 3], "", 1)
            # maybe_error = content[start: end - 1].split("asm")[1]
            # content = content.replace(maybe_error, "", 1)

            # 暂时无法解决，选择跳过本文件
            return asm_blocks, ""

    if content == None:
        print(1111)

    return asm_blocks, content


def asm_file_filter(file_path: str):
    if file_path.lower().endswith(".s") or file_path.lower().endswith(".asm"):
        return True
    return False


def read_macros():
    macros = []
    from settings import X86_MACROS
    with open(X86_MACROS, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    macros += [l.strip() for l in lines]
    return macros


def divide_list(lst: list, step):
    return [lst[i:i + step] for i in range(0, len(lst), step)]


def read_builtin_names():
    from settings import BUILTIN_PATH
    with open(BUILTIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines]
    return lines


def get_all_files(path):
    file_list = []
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            file_list.append(os.path.join(dirpath, filename))
    return file_list


def extract_asm_lines(regex, content):
    asm_lines = ""
    regex = re.compile(regex, re.IGNORECASE)

    for line in content.split("\n"):
        if re.search(regex, line) != None:
            asm_lines += line.strip() + "\n"

    return asm_lines.strip()


def read_file(file_path):
    try:
        with codecs.open(file_path, 'r', encoding='utf-8') as file:
            file_contents = remove_comments(file.read())
            return file_contents
    except UnicodeDecodeError:
        with codecs.open(file_path, 'r', encoding='ISO-8859-1') as file:
            file_contents = remove_comments(file.read())
            return file_contents
    except Exception as e:
        print("Error: ", str(e))
        return ""


def remove_comments(text):
    # 删除单行注释
    text = re.sub(r'//.*?\n', '', text)
    # 删除多行注释
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)
    return text


if __name__ == '__main__':
    file_path = "../data/intrinsics.json"
    res = read_intrinsic_name(file_path, "X86")
    print(res)
