import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# from utils.FileUtil import get_all_files


# from SourceScanner.settings import INTRINSIC_PATH
# from SourceScanner.utils.FileUtil import read_intrinsic_name, normal_file_filter, get_all_files, read_file


class IntrinsicParser:

    def __init__(self, repo_path):
        # self.intrinsic_names = read_intrinsic_name(INTRINSIC_PATH, 'X86')
        # from utils.FileUtil import normal_file_filter
        # from utils.FileUtil import get_all_files
        from SourceScanner.utils.FileUtil import normal_file_filter
        from SourceScanner.utils.FileUtil import get_all_files
        self.file_paths = list(filter(normal_file_filter, get_all_files(repo_path)))

    # 查找源代码文本字符串中调用intrinsic的位置
    def find_intrinsics(self, code):
        # 匹配_mm_和__m128等字符串（这里会漏一点，大概200+）
        intrinsic_regex = re.compile(r'\b_mm_\w+\b|\b__m\d{2,4}\b|\b_mm\d{2,4}_\w+\b|\b_m_\w+\b|\b_tile_\w+\b',
                                     re.IGNORECASE)

        # 查找所有匹配项的位置（这里会有重复项）
        matches = re.findall(intrinsic_regex, code)

        # 补充查找前面漏的200+个
        single_regex = {'_xbegin', '_kor_mask64', '_setssbsy', '_cvtmask8_u32', '_xrstor', '_addcarry_u64', '_rotwr',
                        '_bnd_store_ptr_bounds', '_kand_mask64', '_rdrand16_step', '_kshiftri_mask16',
                        '_directstoreu_u32', '_bnd_chk_ptr_lbounds', '_kandn_mask16', '_kortestz_mask8_u8', '_xsaves',
                        '_BitScanReverse64', '_fxsave64', '_writefsbase_u64', '_addcarryx_u64', '_xsetbv',
                        '_kshiftri_mask8', '_castf64_u64', '_xrstor64', '_popcnt32', '_directstoreu_u64',
                        '_rdseed64_step', '_storebe_i16', '_clui', '_enclv_u32', '_senduipi', '_readfsbase_u32',
                        '_kortestc_mask8_u8', '_rdseed16_step', '_rotr64', '_hreset', '_ptwrite64', '_xsavec',
                        '_rdrand64_step', '_cvtu32_mask32', '_kshiftri_mask32', '_kandn_mask64', '_ktestz_mask32_u8',
                        '_xend', '_kortest_mask16_u8', '_enqcmd', '_bit_scan_forward', '_cvtss_sh', '_loadbe_i64',
                        '_ktestc_mask16_u8', '_allow_cpu_features', '_incsspq', '_castf32_u32', '_kxor_mask8',
                        '_blsr_u64', '_clrssbsy', '_rdtsc', '_xsave', '_cvtu32_mask16', '_xsavec64',
                        '_kortest_mask32_u8', '_kand_mask8', '_bnd_init_ptr_bounds', '_lrotl', '_may_i_use_cpu_feature',
                        '_kxor_mask32', '_writegsbase_u32', '_tzcnt_u64', '_andn_u64', '_ktestz_mask64_u8',
                        '_cvtmask32_u32', '_store_mask16', '_kortestc_mask16_u8', '_mulx_u64', '_load_mask32',
                        '_kshiftli_mask8', '_kandn_mask8', '_bittestandset', '_rdpmc', '_ktest_mask64_u8',
                        '_kadd_mask32', '_blsi_u64', '_xsaveopt', '_ktest_mask32_u8', '_rdpid_u32', '_wbnoinvd',
                        '_subborrow_u32', '_BitScanForward64', '_blsmsk_u64', '_storebe_i32', '_wrussq', '_xrstors64',
                        '_serialize', '_kxor_mask64', '_kortestz_mask32_u8', '_writefsbase_u32', '_xrstors', '_rotl64',
                        '_incsspd', '_xgetbv', '_inc_ssp', '_bzhi_u32', '_knot_mask64', '_knot_mask16',
                        '_bit_scan_reverse', '_popcnt64', '_storebe_i64', '_kand_mask32', '_lzcnt_u64', '_kadd_mask16',
                        '_xsusldtrk', '_knot_mask32', '_kand_mask16', '_rdsspq_i64', '_enqcmds', '_xtest',
                        '_store_mask32', '_wbinvd', '_invpcid', '_kxnor_mask16', '_bittest64', '_fxsave',
                        '_rdseed32_step', '_addcarry_u32', '_load_mask16', '_encls_u32', '_kortestz_mask64_u8',
                        '_bextr2_u64', '_BitScanForward', '_loadbe_i32', '_bnd_copy_ptr_bounds', '_get_ssp', '__rdtscp',
                        '_mulx_u32', '_kxnor_mask64', '_bswap64', '_kadd_mask64', '_BitScanReverse', '_wrussd',
                        '_kshiftri_mask64', '_rdrand32_step', '_xresldtrk', '_bextr2_u32', '_subborrow_u64',
                        '_bittestandreset', '_ktestz_mask16_u8', '_knot_mask8', '_kandn_mask32', '_bnd_set_ptr_bounds',
                        '_bnd_get_ptr_ubound', '_ktest_mask8_u8', '_kshiftli_mask32', '_readgsbase_u32', '_andn_u32',
                        '_kor_mask32', '_ktestc_mask8_u8', '_ktestc_mask64_u8', '_bnd_chk_ptr_ubounds', '_lrotr',
                        '_store_mask8', '_kortest_mask8_u8', '_ktestz_mask8_u8', '_kor_mask8', '_load_mask64',
                        '_bzhi_u64', '_testui', '_cvtu32_mask8', '_tpause', '_blsr_u32', '_wrssd', '_bittest',
                        '_bittestandreset64', '_bnd_get_ptr_lbound', '_stui', '_xsaveopt64', '_saveprevssp',
                        '_xsaves64', '_kshiftli_mask16', '_pext_u32', '_castu32_f32', '_pconfig_u32',
                        '_bnd_chk_ptr_bounds', '_kxor_mask16', '_ptwrite32', '_loadbe_i16', '_bextr_u32',
                        '_kortestc_mask64_u8', '_blsmsk_u32', '_bextr_u64', '_bnd_narrow_ptr_bounds', '_rstorssp',
                        '_readfsbase_u64', '_blsi_u32', '_bittestandset64', '_pdep_u32', '_rotl', '_wrssq', '_fxrstor',
                        '_bswap', '_movdir64b', '_ktest_mask16_u8', '_castu64_f64', '_store_mask64', '_kshiftli_mask64',
                        '_kortestc_mask32_u8', '_cvtu64_mask64', '_cvtmask16_u32', '_rdsspd_i32', '_fxrstor64',
                        '_pdep_u64', '_rotr', '_addcarryx_u32', '_tzcnt_u32', '_rotwl', '_lzcnt_u32', '_kadd_mask8',
                        '_writegsbase_u64', '_xsave64', '_cvtsh_ss', '_umonitor', '_load_mask8', '_cvtmask64_u64',
                        '_bittestandcomplement', '_kor_mask16', '_enclu_u32', '_ktestc_mask32_u8', '_xabort', '_umwait',
                        '_tzcnt_u16', '_kortestz_mask16_u8', '_kxnor_mask32', '_pext_u64', '_kxnor_mask8',
                        '_readgsbase_u64', '_kortest_mask64_u8', '_bittestandcomplement64'}
        intrinsic_regex = re.compile('|'.join(map(re.escape, single_regex)))
        single_matches = re.findall(intrinsic_regex, code)
        matches = list(matches + single_matches)

        # single_regex = {'_xsaves64', '_xsave64', '_bittestandcomplement64', '_bittestandset', '_BitScanForward64',
        #                 '_bittestandcomplement', '_enqcmds', '_xsaveopt', '_xsaves', '_xsavec64', '_bittest64',
        #                 '_bittestandset64', '_bittestandreset', '_bittestandreset64', '_xsavec', '_xrstors64',
        #                 '_rotl64'}
        # intrinsic_regex = re.compile('|'.join(map(re.escape, single_regex)))
        # single_matches = re.findall(intrinsic_regex, code)
        # matches = list(matches + single_matches)
        #
        # single_regex = {'_bittestandreset64', '_xsaveopt64', '_BitScanReverse64', '_xsaves64'}
        # intrinsic_regex = re.compile('|'.join(map(re.escape, single_regex)))
        # single_matches = re.findall(intrinsic_regex, code)
        # matches = list(matches + single_matches)

        return matches

    def process_file(self, file_path):
        # print(f"[IntrinsicParser] processing {file_path}")
        # 读文件
        # from utils.FileUtil import read_file
        from SourceScanner.utils.FileUtil import read_file
        file_content = read_file(file_path)

        # 找Intrinsic函数
        matched = list(self.find_intrinsics(file_content))

        return matched

    def run(self):
        matched_inrtins = []

        with ThreadPoolExecutor() as executor:
            futures = []
            # submit a task for each file
            for file_path in self.file_paths:
                future = executor.submit(self.process_file, file_path)
                futures.append(future)

            # collect the results as they become available
            for future in as_completed(futures):
                matched = future.result()
                matched_inrtins.extend(matched)

        return matched_inrtins


if __name__ == '__main__':
    repo_path = "/Users/jimto/PycharmProjects/repos/coreutils"
    parser = IntrinsicParser(repo_path)
    res = parser.run()

    print(res)
    print(len(res))
