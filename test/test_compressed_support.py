#-------------------------------------------------------------------------------
# Test handling for compressed debug sections
#
# Pierre-Marie de Rodat (pmderodat@kawie.fr)
# This code is in the public domain
#-------------------------------------------------------------------------------

import os
import struct
import unittest
import zlib
from contextlib import contextmanager
from io import BytesIO

from elftools.common.exceptions import ELFCompressionError
from elftools.elf.constants import SH_FLAGS
from elftools.elf.descriptions import describe_sh_flags
from elftools.elf.elffile import ELFFile
from elftools.dwarf.dwarfinfo import DebugSectionDescriptor


class TestCompressedSupport(unittest.TestCase):

    def test_compressed_32(self):
        with self.elffile('32') as elf:
            section = elf.get_section_by_name('.debug_info')
            self.assertTrue(section.compressed)
            self.assertEqual(describe_sh_flags(section['sh_flags']), 'C')
            self.assertEqual(section.data_size, 0x330)
            self.assertEqual(section.data_alignment, 1)

            self.assertEqual(self.get_cus_info(elf), ['CU 0x0: 0xb-0x322'])

    def test_compressed_64(self):
        with self.elffile('64') as elf:
            section = elf.get_section_by_name('.debug_info')
            self.assertTrue(section.compressed)
            self.assertEqual(describe_sh_flags(section['sh_flags']), 'C')
            self.assertEqual(section.data_size, 0x327)
            self.assertEqual(section.data_alignment, 1)
            self.assertEqual(self.get_cus_info(elf), ['CU 0x0: 0xb-0x319'])

    def test_compressed_and_excluded_flags(self):
        flags = SH_FLAGS.SHF_COMPRESSED | SH_FLAGS.SHF_EXCLUDE
        self.assertEqual(describe_sh_flags(flags), 'CE')

    def test_compressed_unknown_type(self):
        with self.elffile('unknown_type') as elf:
            section = elf.get_section_by_name('.debug_info')
            try:
                section.data()
            except ELFCompressionError as exc:
                self.assertEqual(
                    str(exc), 'Unknown compression type: 0x7ffffffe'
                )
            else:
                self.fail('An exception was exected')

    def test_compressed_bad_size(self):
        with self.elffile('bad_size') as elf:
            section = elf.get_section_by_name('.debug_info')
            try:
                section.data()
            except ELFCompressionError as exc:
                self.assertEqual(
                    str(exc),
                    'Decompressed data is 807 bytes long, should be 808 bytes'
                    ' long'
                )
            else:
                self.fail('An exception was exected')

    # Test helpers

    @contextmanager
    def elffile(self, name):
        """ Context manager to open and parse an ELF file.
        """
        with open(os.path.join('test', 'testfiles_for_unittests',
                               f'compressed_{name}.o'), 'rb') as f:
            yield ELFFile(f)

    def get_cus_info(self, elffile):
        """ Return basic info about the compile units in `elffile`.

        This is used as a basic sanity check for decompressed DWARF data.
        """
        result = []

        dwarf = elffile.get_dwarf_info()
        for cu in dwarf.iter_CUs():
            dies = []

            def traverse(die, dies=dies):
                dies.append(die.offset)
                for child in die.iter_children():
                    traverse(child)

            traverse(cu.get_top_DIE())
            result.append(f'CU {cu.cu_offset:#0x}: {dies[0]:#0x}-{dies[-1]:#0x}')

        return result


class TestGNUZDebugDecompression(unittest.TestCase):
    """ Malformed GNU .zdebug sections must raise ELFCompressionError rather
        than AssertionError (which -O strips) or raw zlib errors, and must
        never expand past the declared uncompressed size.
    """
    def descriptor(self, payload):
        return DebugSectionDescriptor(
            stream=BytesIO(payload), name='.zdebug_info',
            global_offset=0, size=len(payload), address=0)

    def zdebug(self, declared_size, compressed_payload, magic=b'ZLIB'):
        return magic + struct.pack('>Q', declared_size) + compressed_payload

    def test_bad_magic(self):
        payload = self.zdebug(4, zlib.compress(b'data'), magic=b'XYZB')
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_header_too_small(self):
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(b'ZLIB\x00'))

    def test_declared_size_mismatch(self):
        payload = self.zdebug(99, zlib.compress(b'data'))
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_truncated_zlib_stream(self):
        full = zlib.compress(b'A' * 1000)
        payload = self.zdebug(1000, full[:len(full) // 2])
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_truncated_stream_at_declared_size(self):
        # The truncated stream still produces exactly the declared number of
        # bytes: it must be rejected instead of silently accepted.
        full = zlib.compress(b'A' * 100)
        payload = self.zdebug(100, full[:-3])
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_garbage_payload(self):
        payload = self.zdebug(4, b'not-a-zlib-stream')
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_oversize_stream_capped_and_rejected(self):
        # The header claims 100 bytes but the stream expands to 50 KiB.
        inflated = b'B' * (50 * 1024)
        payload = self.zdebug(100, zlib.compress(inflated))
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))

    def test_valid_roundtrip(self):
        data = b'hello compressed section' * 64
        payload = self.zdebug(len(data), zlib.compress(data))
        result = ELFFile._decompress_dwarf_section(
            self.descriptor(payload))
        result.stream.seek(0)
        self.assertEqual(result.stream.read(), data)
        self.assertEqual(result.size, len(data))

    def test_valid_empty_section(self):
        payload = self.zdebug(0, zlib.compress(b''))
        result = ELFFile._decompress_dwarf_section(
            self.descriptor(payload))
        self.assertEqual(result.size, 0)
        result.stream.seek(0)
        self.assertEqual(result.stream.read(), b'')

    def test_zero_declared_size_with_data_rejected(self):
        # The loop must still feed zlib when the declared size is zero,
        # instead of skipping decompression and accepting the section.
        payload = self.zdebug(0, zlib.compress(b'x'))
        with self.assertRaises(ELFCompressionError):
            ELFFile._decompress_dwarf_section(self.descriptor(payload))
