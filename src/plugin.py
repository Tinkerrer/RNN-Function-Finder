import ida_ida
import ida_funcs
import ida_idaapi
import ida_name
import ida_loader
import sys


def analyze_functions(idb_path):
    # Загружаем IDB-файл
    if not ida_loader.load_database(idb_path):
        print(f"Ошибка: не удалось загрузить файл {idb_path}")
        return False

    # Ждём завершения автоматического анализа
    ida_idaapi.auto_wait()

    # Получаем общее количество функций
    func_count = ida_funcs.get_func_qty()
    print(f"Всего функций в базе: {func_count}")

    # Итерируемся по всем функциям
    for i in range(func_count):
        func = ida_funcs.getn_func(i)
        if not func:
            continue

        start_ea = func.start_ea
        end_ea = func.end_ea
        func_name = ida_name.get_ea_name(start_ea)

        print(f"Функция {i + 1}:")
        print(f"  Имя: {func_name}")
        print(f"  Адрес начала: 0x{start_ea:X}")
        print(f"  Адрес конца: 0x{end_ea:X}")
        print(f"  Размер: {end_ea - start_ea} байт")

    return True


# if __name__ == "__main__":
#     print("Helo!!")
#     if len(sys.argv) < 2:
#         print("Использование: idat64 -A -Sscript.py <путь_к_idb>")
#         sys.exit(1)
#
#     idb_path = sys.argv[1]
#     analyze_functions(idb_path)

analyze_functions("C:/Users/ashaykhanov/Documents/Projects/TERMINAL/FIRMWARE/idb/first/reader-1.50.7567_FIRST_FIRM.P_NO_IRF_HEADER_20250626163647.idb")