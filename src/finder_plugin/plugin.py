import ida_idaapi
import ida_funcs
import idautils

from pluginform import FunctionStartFinderForm


class FunctionStartFinderPlugmod(ida_idaapi.plugmod_t):
    def __del__(self):
        print(">>> FunctionStartFinderPlugmod: destructor called.")
    
    def run(self, arg):
        print(f">>> FunctionStartFinderPlugmod.run() is invoked with argument value: {arg}.")

        form = FunctionStartFinderForm()
        form.exec_()


class FunctionStartFinderPlugin(ida_idaapi.plugin_t):
    flags = ida_idaapi.PLUGIN_UNL | ida_idaapi.PLUGIN_MULTI | ida_idaapi.PLUGIN_MOD
    comment = "This is a function start finder. It is based on RNN"
    help = "Visit github repo: "
    wanted_name = "Function start finder"
    wanted_hotkey = "Shift-P"

    def init(self):
        print(">>>FunctionStartFinderPlugin: Init called.")
        return FunctionStartFinderPlugmod()


def PLUGIN_ENTRY():
    return FunctionStartFinderPlugin()