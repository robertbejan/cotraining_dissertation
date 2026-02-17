import csv
import pandas as pd
from io import StringIO

# The content of your file as a string
file_content = """
small_80_start5_rgb0.95_fft0.9,small_80,80,5,0.95,0.9,0.8685483870967742,0.7931451612903225,0.8612903225806452,"65,29,0,0,29,19,0,612,0,0,5,1,0,27,155,0,25,1,0,1,0,324,0,0,7,72,33,5,692,34,1,17,1,0,19,306","8,42,1,1,32,58,2,555,0,1,40,20,4,36,115,0,28,25,0,1,0,322,1,1,20,75,19,5,645,79,0,10,1,1,10,322","55,30,0,0,31,26,0,610,0,0,7,1,0,25,156,0,24,3,0,1,0,324,0,0,10,76,30,6,681,40,0,18,1,0,15,310",6,3465,4401,7371,2172,3108,2025-12-12 00:25:50
organized_50_start5_rgb0.95_fft0.9,organized_50,50,5,0.95,0.9,0.9306451612903226,0.8282258064516129,0.930241935483871,"118,3,0,0,17,4,2,612,0,0,3,1,0,2,165,0,41,0,1,0,0,321,3,0,14,15,18,3,769,24,1,3,0,0,17,323","64,31,2,0,30,15,10,580,2,0,21,5,10,26,141,0,26,5,0,0,1,319,4,1,49,66,29,4,663,32,29,14,1,0,13,287","120,4,0,0,15,3,1,613,0,0,4,0,0,5,165,0,38,0,1,0,0,322,2,0,14,18,20,3,760,28,1,3,0,0,13,327",6,5548,6097,4866,1333,1882,2025-12-12 21:38:42
large_20_start5_rgb0.95_fft0.9,large_20,20,5,0.95,0.9,0.9403225806451613,0.8302419354838709,0.9439516129032258,"127,1,1,0,10,3,1,614,1,0,2,0,0,0,191,0,17,0,1,0,0,322,2,0,11,11,42,1,756,22,0,2,0,0,20,322","46,16,2,0,65,13,11,533,8,1,63,2,1,9,152,0,44,2,1,0,0,321,3,0,24,21,28,3,742,25,11,3,4,0,61,265","124,1,0,0,13,4,0,614,1,0,3,0,0,0,191,0,17,0,1,0,0,323,1,0,11,12,35,1,764,20,0,2,0,0,17,325",6,7074,6929,1946,330,185,2025-12-14 19:45:50
"""

# --- Helper Function ---
## Function to convert a flat list of 36 ints into a matrix string
def make_matrix_string(values):
    """Converts a comma-separated string of 36 integers into a 6x6 formatted matrix string."""
    try:
        # Check if values is a string and convert to a list of ints
        if isinstance(values, str):
            value_list = list(map(int, values.split(",")))
        # If it's already a list (in case it was pre-processed), use it directly
        elif isinstance(values, list):
            value_list = list(map(int, values))
        else:
            raise ValueError("Input must be a string of comma-separated integers.")

        assert len(value_list) == 36, f"Confusion matrix must have exactly 36 values, but got {len(value_list)}."
        rows = []
        for i in range(6):
            row_vals = value_list[i * 6:(i + 1) * 6]
            # Use f-string formatting to ensure each number takes 4 characters (e.g., '  65')
            row_str = " ".join(f"{v:4d}" for v in row_vals)
            rows.append(row_str)
        return "\n" + "\n".join(rows) + "\n"
    except Exception as e:
        # Handle cases where the input is malformed, returning the original string/value
        print(f"Error processing matrix: {e}. Returning original value.")
        return str(values)


# --- MODIFIED LOGIC ---

# 1. Define Column Headers
# The sample data has 18 columns in total (9 standard + 3 quoted matrices + 6 standard).
# I'll provide descriptive names for the columns based on their presumed content.
header = [
    'full_name', 'name_part1', 'name_part2', 'start', 'rgb', 'fft',
    'metric1', 'metric2', 'metric3',
    'matrix_str_1', 'matrix_str_2', 'matrix_str_3',
    'val1', 'val2', 'val3', 'val4', 'val5', 'val6', # <--- ADDED 'val6'
    'timestamp'
]

# Indices of the columns containing the confusion matrix strings (0-indexed)
# Based on the sample, these are columns 9, 10, and 11.
MATRIX_INDICES = [9, 10, 11]

# 2. Use StringIO to treat the string content as a file and csv.reader to parse
# This correctly handles the quoted fields.
file_stream = StringIO(file_content.strip())
reader = csv.reader(file_stream)

processed_rows = []

for parsed_row in reader:
    # Basic check for column count consistency
    if len(parsed_row) != len(header):
        print(f"Skipping row: Expected {len(header)} columns, but found {len(parsed_row)}.")
        continue

    current_row = list(parsed_row)

    # Convert the confusion matrix columns
    for idx in MATRIX_INDICES:
        # The csv.reader already strips quotes, so we just check if the content is a string
        if isinstance(current_row[idx], str):
            current_row[idx] = make_matrix_string(current_row[idx])

    processed_rows.append(current_row)

# 3. Create the DataFrame and export
if processed_rows:
    df = pd.DataFrame(processed_rows, columns=header)

    # Print the DataFrame to verify the matrix formatting (optional)
    # print(df[['name_part1', 'matrix_str_1']].head())

    # Export to Excel
    df.to_excel("output80_GOOD.xlsx", index=False)
    print("\n✅ output80_GOOD.xlsx generated successfully.")
    print(f"Dataframe shape: {df.shape}")
else:
    print("❌ No valid data found to create DataFrame.")