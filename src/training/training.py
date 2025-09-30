import idb
import sys
import os
import random

from pathlib import Path

from model import BidirectionalRNNClassifier

RANDOM_PADDING = False

def main(idb_dir_path):
    idb_data = []
    idb_func_property = []

    for root, dirs, files in os.walk(idb_dir_path):
        for filename in files:

            file_path = Path(root) / filename
            if file_path.suffix.lower() not in ('.idb', '.i64'):
                continue

            print(f"Processing {file_path} ...")

            with idb.from_file(file_path) as db:
                api = idb.IDAPython(db)

                # Creating byte sequences with "start func" property
                for seg_start in api.idautils.Segments():
                    seg_end = api.idc.SegEnd(seg_start)

                    print(f"Segment start: {hex(seg_start)}")
                    print(f"Segment end: {hex(seg_end)}")
                    print(api.idc.GetSegmentAttr(seg_start, api.idc.SEGATTR_TYPE))

                    property_seq = bytearray([0] * (seg_end - seg_start))

                    # Creating set from func start addresses
                    start_func_addresses = api.idautils.Functions(seg_start, seg_end - 1)

                    for addr in start_func_addresses:
                        property_seq[addr - seg_start] = 1

                    byte_seq = api.idaapi.get_bytes(seg_start, seg_end - seg_start)

                    # Without Random padding between functions
                    if not RANDOM_PADDING:
                        idb_data.append(byte_seq)
                        idb_func_property.append(property_seq)
                        continue

                    # With Random padding between functions
                    bytearray_seq = bytearray(byte_seq)

                    if api.idc.GetSegmentAttr(seg_start, api.idc.SEGATTR_TYPE) == 2:
                        print(bytes(bytearray_seq).count(0))

                        for index in range(len(bytearray_seq)):
                            if bytearray_seq[index] == 0x00 and (api.ida_funcs.get_func(seg_start + iter) is None):
                                bytearray_seq[index] = random.randint(0, 255)

                    idb_data.append(bytes(bytearray_seq))
                    idb_func_property.append(property_seq)

    model = BidirectionalRNNClassifier()

    print(f"Total functions number: {sum(sum(seq) for seq in idb_func_property)}")
    print(f"Total bytes number: {sum(len(seq) for seq in idb_func_property)}")

    x_data, y_data = model.split_byte_sequences(idb_data, idb_func_property)
    print(model.summary())
    model.train(x_data, y_data)

    model.save_model("find_func_start_model.h5")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Path to the database directory is required!")
        print("Usage: python training.py <path_to_idb_dir>")
        sys.exit(1)

    main(sys.argv[1])
