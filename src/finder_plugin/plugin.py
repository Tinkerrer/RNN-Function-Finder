import ida_idaapi
import ida_funcs
import idautils
import idc
import idaapi
import os

from pluginform import FunctionStartFinderForm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEQUENCE_LENGTH = 1000


class FunctionStartFinderPlugmod(ida_idaapi.plugmod_t):
    def __del__(self):
        print(">>> FunctionStartFinderPlugmod: destructor called.")

    def split_byte_sequence(self, byte_sequence):
        split_sequences = []

        for j in range(0, len(byte_sequence), SEQUENCE_LENGTH):
            split_sequences.append(list(byte_sequence[j:j + SEQUENCE_LENGTH]))

        tail_length = len(byte_sequence) % SEQUENCE_LENGTH

        if tail_length != 0:
            split_sequences.append(
                list(byte_sequence[len(byte_sequence) - tail_length:len(byte_sequence)]
                     )
            )

        return split_sequences

    def preprocess_data(self, sequences_list):
        return pad_sequences(
            sequences_list,
            maxlen=SEQUENCE_LENGTH,
            padding='post',  # 'post' — нули в конец, 'pre' — в начало
            dtype='int32',
            value=0
        )

    def prepare_byte_seq(self, byte_sequence):
        return self.preprocess_data(self.split_byte_sequence(byte_sequence))

    def predict_function_start(self, start_addr, end_addr, model_filepath, soft_predict=False):

        loaded_model = load_model(model_filepath)

        prepared_byte_seq = self.prepare_byte_seq(idaapi.get_bytes(start_addr, end_addr - start_addr))

        predicted = loaded_model.predict(prepared_byte_seq)

        found_ctr = 0
        error_create_ctr = 0

        if soft_predict:
            for addr in range(start_addr, end_addr):
                if predicted[(addr - start_addr) // SEQUENCE_LENGTH][(addr - start_addr) % SEQUENCE_LENGTH] > 0.5:
                    ida_funcs.set_func_cmt(addr, "Possible Func Start !!!", 1)
                    found_ctr += 1

        else:
            for addr in range(start_addr, end_addr):
                if predicted[(addr - start_addr) // SEQUENCE_LENGTH][(addr - start_addr) % SEQUENCE_LENGTH] > 0.5:

                    if ida_funcs.add_func(addr):
                        found_ctr += 1
                        continue

                    print(f"Can't create func on {hex(addr)}")
                    error_create_ctr += 1

            print(f">>> Number of functions that could not be created: {error_create_ctr}.")

        print(f">>> Total found function number : {found_ctr}.")


    def run(self, arg):

        print(f">>> FunctionStartFinderPlugmod.run() is invoked with argument value: {arg}.")

        form = FunctionStartFinderForm()

        ret_tuple = form.exec_()

        if ret_tuple is None:
            return

        start_addr, end_addr, model_file_name, soft_find = ret_tuple

        if not os.path.exists(model_file_name):
            print("Model file doesn't exist, try again(")
            return

        if not (isinstance(start_addr, int) and isinstance(end_addr, int)):
            print(f"Something wrong with addresses")
            return

        # Checks
        self.predict_function_start(start_addr, end_addr, model_file_name, soft_find)


class FunctionStartFinderPlugin(ida_idaapi.plugin_t):
    flags = ida_idaapi.PLUGIN_UNL | ida_idaapi.PLUGIN_MULTI | ida_idaapi.PLUGIN_MOD
    comment = "This is a function start finder. It is based on RNN"
    help = "Visit github repo: <URL>"
    wanted_name = "Function start finder"
    wanted_hotkey = "Shift-P"

    def init(self):
        print(">>>FunctionStartFinderPlugin: Init called.")
        return FunctionStartFinderPlugmod()


def PLUGIN_ENTRY():
    return FunctionStartFinderPlugin()
