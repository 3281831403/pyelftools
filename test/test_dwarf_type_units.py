#-------------------------------------------------------------------------------
# elftools tests
#
# Tests for type-signature lookups: DWARFv4 type units in .debug_types and
# DWARFv5 type units (DW_UT_type) in .debug_info.
#
# This code is in the public domain
#-------------------------------------------------------------------------------
import io
import os
import unittest

from elftools.common.exceptions import DWARFError
from elftools.common.utils import bytes2str
from elftools.elf.elffile import ELFFile

CUv5 = os.path.join('test', 'testfiles_for_readelf', 'cuv5_x86-64_gcc.so.elf')
TYPESv4 = os.path.join('test', 'testfiles_for_readelf', 'dwarf_debug_types.elf')

# From `readelf --debug-dump=info` on cuv5_x86-64_gcc.so.elf (gcc 8.3.1 -gdwarf-5
# -fdebug-types-section): the DW_UT_type unit at offset 0 describes type unit
# "ball_t" and the DW_TAG_formal_parameter at 0xab references it by signature.
CUv5_SIG = 0xa907c670544738f5
CUv5_TYPE_OFFSET = 0x1e
CUv5_TYPE_DIE = 0x1e

TYPESv4_SIG = 6012688284197268420


class TestTypeSignatureLookup(unittest.TestCase):
    def test_dwarfv5_get_DIE_by_sig8(self):
        with open(CUv5, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()
            self.assertFalse(dwarf.has_debug_types())

            die = dwarf.get_DIE_by_sig8(CUv5_SIG)
            self.assertEqual(die.offset, CUv5_TYPE_DIE)
            self.assertEqual(die.tag, 'DW_TAG_structure_type')
            self.assertEqual(bytes2str(die.attributes['DW_AT_name'].value), 'ball_t')

            # The type DIE's members are reachable from the returned DIE.
            members = list(die.iter_children())
            self.assertEqual([d.tag for d in members], ['DW_TAG_member'] * 2)
            self.assertEqual(
                [bytes2str(d.attributes['DW_AT_name'].value) for d in members],
                ['i', 'y'])

    def test_dwarfv5_get_TU_by_sig8(self):
        with open(CUv5, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()
            tu = dwarf.get_TU_by_sig8(CUv5_SIG)
            self.assertEqual(tu.tu_offset, 0)
            self.assertEqual(tu.header['unit_type'], 'DW_UT_type')
            self.assertEqual(tu.header['type_signature'], CUv5_SIG)
            self.assertEqual(tu.header['type_offset'], CUv5_TYPE_OFFSET)
            self.assertIs(tu.get_top_DIE().cu, tu)
            self.assertEqual(tu.get_DIE_from_refaddr(CUv5_TYPE_DIE).tag,
                             'DW_TAG_structure_type')

    def test_dwarfv5_DIE_from_ref_sig8_attribute(self):
        with open(CUv5, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()

            refs = []
            for cu in dwarf.iter_CUs():
                for die in cu.iter_DIEs():
                    if die.tag and die.attributes.get('DW_AT_type'):
                        attr = die.attributes['DW_AT_type']
                        if attr.form == 'DW_FORM_ref_sig8':
                            refs.append(die.get_DIE_from_attribute('DW_AT_type'))

            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].offset, CUv5_TYPE_DIE)
            self.assertEqual(refs[0].tag, 'DW_TAG_structure_type')
            self.assertEqual(bytes2str(refs[0].attributes['DW_AT_name'].value),
                             'ball_t')
            self.assertEqual(len(list(refs[0].iter_children())), 2)

    def test_repeated_lookup_does_not_rescan(self):
        with open(CUv5, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()

            calls = []
            orig = dwarf._parse_CU_at_offset
            orig_tu = dwarf._parse_TU_at_offset

            def counting_cu(offset):
                calls.append(offset)
                return orig(offset)

            def counting_tu(offset):
                calls.append(offset)
                return orig_tu(offset)

            dwarf._parse_CU_at_offset = counting_cu  # type: ignore[method-assign]
            dwarf._parse_TU_at_offset = counting_tu  # type: ignore[method-assign]

            first = dwarf.get_DIE_by_sig8(CUv5_SIG)
            scans = len(calls)
            self.assertGreater(scans, 0)

            for _ in range(5):
                self.assertIs(dwarf.get_DIE_by_sig8(CUv5_SIG), first)
                self.assertIs(dwarf.get_TU_by_sig8(CUv5_SIG),
                              dwarf.get_TU_by_sig8(CUv5_SIG))
            # No unit header was re-parsed for the later lookups.
            self.assertEqual(len(calls), scans)

    def test_dwarfv4_debug_types_unchanged(self):
        with open(TYPESv4, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()
            self.assertTrue(dwarf.has_debug_types())

            tu = dwarf.get_TU_by_sig8(TYPESv4_SIG)
            self.assertEqual(tu.header['version'], 4)
            self.assertEqual(tu.header['signature'], TYPESv4_SIG)

            die = dwarf.get_DIE_by_sig8(TYPESv4_SIG)
            self.assertEqual(die.offset, tu.tu_offset + tu.header['type_offset'])
            self.assertEqual(die.tag, 'DW_TAG_enumeration_type')
            self.assertEqual(len(list(die.iter_children())), 26)

            # iter_CUs still walks .debug_info only, in section order.
            cu_offsets = [cu.cu_offset for cu in dwarf.iter_CUs()]
            self.assertEqual(cu_offsets, sorted(cu_offsets))

    def test_dwarfv4_lookup_does_not_scan_debug_info(self):
        with open(TYPESv4, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()

            parsed = []
            orig = dwarf._parse_CU_at_offset

            def counting_cu(offset):
                parsed.append(offset)
                return orig(offset)

            dwarf._parse_CU_at_offset = counting_cu  # type: ignore[method-assign]

            # DWARFv4 signatures are answered from .debug_types alone.
            dwarf.get_TU_by_sig8(TYPESv4_SIG)
            dwarf.get_DIE_by_sig8(TYPESv4_SIG)
            self.assertEqual(parsed, [])

    def test_missing_signature_raises_KeyError(self):
        with open(CUv5, 'rb') as f:
            dwarf = ELFFile(f).get_dwarf_info()
            self.assertRaises(KeyError, dwarf.get_DIE_by_sig8, 0xdeadbeef)
            self.assertRaises(KeyError, dwarf.get_TU_by_sig8, 0xdeadbeef)

    def test_corrupt_type_offset_raises_DWARFError(self):
        with open(CUv5, 'rb') as f:
            elf = ELFFile(f)
            sec = elf.get_section_by_name('.debug_info')
            data = bytearray(sec.data())
            # Patch the DWARFv5 type unit's type_offset (unit header starts at
            # offset 0): unit_length(4) version(2) unit_type(1) address_size(1)
            # debug_abbrev_offset(4) type_signature(8) type_offset(4).
            sec_off = sec['sh_offset']
            field = 4 + 2 + 1 + 1 + 4 + 8
            data[field:field + 4] = (0xFFFFFFF0).to_bytes(4, 'little')

            # Patch the whole file so that ELF section offsets stay consistent,
            # then let ELFFile read .debug_info from it.
            f.seek(0)
            whole = bytearray(f.read())
            off = sec_off
            whole[off:off + len(data)] = data
            patched = io.BytesIO(bytes(whole))

            dwarf = ELFFile(patched).get_dwarf_info()
            # The TU itself is still resolvable by signature...
            self.assertEqual(dwarf.get_TU_by_sig8(CUv5_SIG).tu_offset, 0)
            # ...but the type DIE it points at is out of the unit's range,
            # so resolving the DIE must fail loudly instead of returning
            # some other DIE that happens to sit at that offset.
            with self.assertRaises(DWARFError):
                dwarf.get_DIE_by_sig8(CUv5_SIG)

    def test_dwarfv4_corrupt_type_offset_raises_DWARFError(self):
        with open(TYPESv4, 'rb') as f:
            elf = ELFFile(f)
            sec = elf.get_section_by_name('.debug_types')
            sec_off = sec['sh_offset']
            # DWARFv4 TU header: unit_length(4) version(2) abbrev_offset(4)
            # address_size(1) signature(8) type_offset(4).
            field = 4 + 2 + 4 + 1 + 8
            f.seek(0)
            whole = bytearray(f.read())
            whole[sec_off + field:sec_off + field + 4] = (
                0xFFFFFFF0).to_bytes(4, 'little')

            dwarf = ELFFile(io.BytesIO(bytes(whole))).get_dwarf_info()
            with self.assertRaises(DWARFError):
                dwarf.get_DIE_by_sig8(TYPESv4_SIG)


if __name__ == '__main__':
    unittest.main()
