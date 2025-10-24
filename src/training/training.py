import idb
import sys
import os
import random

from pathlib import Path

from model import BidirectionalRNNClassifier

# Random padding between functions FLAG
RANDOM_PADDING_ON_TRAIN = True
RANDOM_PADDING_ON_VALUE = True

def main(idb_dir_path):
    idb_data = []
    idb_func_property = []

    for root, dirs, files in os.walk(idb_dir_path):
        for filename in files:

            # Searching for IDA pro bases
            file_path = Path(root) / filename
            if file_path.suffix.lower() not in ('.idb', '.i64'):
                continue

            print(f"Processing {file_path} ...")

            with idb.from_file(file_path) as db:
                api = idb.IDAPython(db)

                for seg_start in api.idautils.Segments():
                    seg_end = api.idc.SegEnd(seg_start)

                    print(f"Segment start: {hex(seg_start)}")
                    print(f"Segment end: {hex(seg_end)}")
                    print(api.idc.GetSegmentAttr(seg_start, api.idc.SEGATTR_TYPE))

                    property_seq = bytearray([0] * (seg_end - seg_start))

                    # Creating property sequence from func start addresses
                    start_func_addresses = api.idautils.Functions(seg_start, seg_end - 1)

                    for addr in start_func_addresses:
                        property_seq[addr - seg_start] = 1

                    # Creating byte sequence with get_bytes()
                    byte_seq = api.idaapi.get_bytes(seg_start, seg_end - seg_start)
                    idb_data.append(byte_seq)
                    idb_func_property.append(property_seq)

                    # # Saving sequences
                    # if not RANDOM_PADDING:
                    #     idb_data.append(byte_seq)
                    #     idb_func_property.append(property_seq)
                    #     continue
                    #
                    # # Replacing zeros between functions with random paddings and saving sequences
                    # bytearray_seq = bytearray(byte_seq)
                    #
                    # if api.idc.GetSegmentAttr(seg_start, api.idc.SEGATTR_TYPE) == 2:
                    #     print(bytes(bytearray_seq).count(0))
                    #
                    #     for index in range(len(bytearray_seq)):
                    #         if bytearray_seq[index] == 0x00 and (api.ida_funcs.get_func(seg_start + index) is None):
                    #             bytearray_seq[index] = random.randint(0, 255)
                    #
                    # idb_data.append(bytes(bytearray_seq))
                    # idb_func_property.append(property_seq)

    model = BidirectionalRNNClassifier(random_padding_on_train=RANDOM_PADDING_ON_TRAIN, random_padding_on_val=RANDOM_PADDING_ON_VALUE)

    print(f"Total functions number: {sum(sum(seq) for seq in idb_func_property)}")
    print(f"Total bytes number: {sum(len(seq) for seq in idb_func_property)}")

    x_data, y_data = model.split_byte_sequences(idb_data, idb_func_property)
    print(model.summary())
    model.train(x_data, y_data)

    saved_model_file_name = "find_func_start_model.h5"

    if RANDOM_PADDING_ON_TRAIN:
        saved_model_file_name = "train_rand_pad_" + saved_model_file_name

    if RANDOM_PADDING_ON_VALUE:
        saved_model_file_name = "value_rand_pad_" + saved_model_file_name

    model.save_model(saved_model_file_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Path to the database directory is required!")
        print("Usage: python training.py <path_to_idb_dir>")
        sys.exit(1)

    main(sys.argv[1])
