from typing import List

class TableCleaner:
    def __init__(self, tables: List[List[List[str]]]):
        self.tables = tables
        self.filtered_tables: List[List[List[str]]] = []
        self.check_invalid: bool = False

    def remove_unwanted_rows(self, remove_keys: List[str]):
        self.filtered_tables = [
            [row for row in table if row[0] not in remove_keys]
            for table in self.tables
        ]
        return self.filtered_tables

    def print_invalid_tables(self, expected_row_count: int = 8):
        print("⚠️⚠️⚠️ Please record the table and row index in the table (start with 0)... ⚠️⚠️⚠️\n")
        for i, table in enumerate(self.filtered_tables):
            if len(table) != expected_row_count:
                self.check_invalid = True
                print(f"⚠️ Table {i} (Total Rows: {len(table)}) is invalid. (Expected total {expected_row_count})")
                for row in table:
                    print(row)
                print("\n" + "-" * 50 + "\n")

    def delete_row(self, table_idx: int, row_idx: int):
        if 0 <= table_idx < len(self.filtered_tables):
            table = self.filtered_tables[table_idx]
            if 0 <= row_idx < len(table):
                del table[row_idx]

    def batch_delete_by_input(self):
        try:
            raw_input_str = input(
                "\nEnter according to the following format (table_index, num_deleted_rows, row_index)."
                "\nExample: (5, 2, [5, 9]), or (6, 1, 5):\n> "
            )

            user_input = eval(raw_input_str)

            if not isinstance(user_input, tuple) or len(user_input) != 3:
                print("❌ Invalid format. Please enter a tuple with 3 elements.")
                return

            table_idx, num_deleted_rows, row_indices = user_input

            if not isinstance(table_idx, int) or not isinstance(num_deleted_rows, int):
                print("❌ table_index and num_deleted_rows must be integers.")
                return

            if table_idx < 0 or table_idx >= len(self.filtered_tables):
                print("❌ Invalid table_index.")
                return

            table = self.filtered_tables[table_idx]

            if num_deleted_rows <= 0 or num_deleted_rows >= len(table):
                print("❌ Invalid number of rows to delete.")
                return

            if num_deleted_rows == 1:
                if not isinstance(row_indices, int):
                    print("❌ With num_deleted_rows = 1, row_index must be an integer.")
                    return
                row_indices = [row_indices]
            else:
                if not isinstance(row_indices, list) or len(row_indices) != num_deleted_rows:
                    print(f"❌ With num_deleted_rows = {num_deleted_rows}, row_index must be a list with {num_deleted_rows} elements.")
                    return

            if any(idx < 0 or idx >= len(table) for idx in row_indices):
                print("❌ There is an invalid row index.")
                return

            print("\n--- Confirm information ---")
            print(f"- Deleted table: {table_idx}")
            print(f"- Number of deleted rows: {num_deleted_rows}")
            print(f"- Row index: {row_indices}")
            confirm = input("Are you sure you want to delete? (y/n): ").strip().lower()
            if confirm != 'y':
                print("❎ Cancel the delete operation.")
                return

            row_indices.sort()
            deleted_count = 0
            for original_idx in row_indices:
                adjusted_idx = original_idx - deleted_count
                self.delete_row(table_idx, adjusted_idx)
                deleted_count += 1

            print(f"✅ Successfully deleted {deleted_count} row(s) from table {table_idx}.")

        except Exception as e:
            print(f"❌ Error: {e}")

    def run_deletion_loop(self):
        print("=" * 50 + " Deleting Section " + "=" * 50)
        while self.check_invalid:
            self.batch_delete_by_input()
            cont = input("Continue? (y/n): ").strip().lower()
            if cont != 'y':
                break