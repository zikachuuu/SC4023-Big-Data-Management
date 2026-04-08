# SC4023 Big Data Management - HDB Resale Data Analysis

## Updates
- **2026-02-15:** Implemented the column store database class
- **2026-02-20:** Implemented querying
- **2026-02-21:** New storage logic for `month` and `floor_area` column. Corresponding querying logic updated.
- **2026-03-20:** Bug fixes regarding `target_end_month`. Validated correctness of query with PostgreSQL.
- **2026-03-21:** Further bug fixes regarding `target_end_month`. Improved and generalised correctness query script with SQLite.
- **2026-03-26:** Implemented disk-based storage for the columnar database.
- **2026-04-08:** Implemented clustered b-tree, sorted the data by month. Added vectorised processing to the query. Verified correctness w.r.t. our matric numbers.

## To Start
Simply install the necessary dependencies from `requirements.txt` and run `main.py`. You will be prompted to enter a matriculation number, which will be parsed to retrieve the correct queries. A copy of the database (since its non persistent) as well as the logs (`run_<matriculation_num>.log`) will be saved in `Logs` folder. The output `ScanResult_<matriculation_num>.csv` will be saved in `Results` folder.

## Database Design
We implemented a columnar database by creating a ColumnStoreDB class in Python. The data structure consists of independent 1-dimensional NumPy memory-mapped (memmap) arrays, where these arrays are stored as binary files on disk. Each column in the database corresponds to a NumPy memmap array. This simulates a disk-based storage scheme for our project.

During initialization, the entire CSV is first physically sorted (clustered) chronologically by month. The data is then parsed in horizontal chunks, vertically shredded into independent columns, and encoded before being written directly to separate `.bin` files on the disk.

The column store database is made up of the following data structures:
- `col_names`: A dictionary that maps each column name to an index.

- `val_code_mapper`: A list of dictionaries which, for each column, maps each unique value (string or integer) to a 32-bit integer code. The integer code preserves the ordering of the values.
    - For the `month` column, each string value (e.g., Jan-20, Dec-19) is converted to a 4-digit code (e.g., 2001, 1912). This preserves chronological order natively (e.g., 1912 < 2001).
    - For the `floor_area_sqm` column, float values are converted to an integer code by multiplying by 10, preserving 1 decimal place of precision while enabling fast integer math.
    - For remaining columns, we sort them lexicographically by increasing order and map each value to an integer code starting from 0.

- `code_val_mapper`: The reverse mapping of integer codes to unique values for each column, used when writing to the final CSV.

- `columns`: The main Column Store Database. IThis is a list of NumPy memory-mapped arrays (`np.memmap`). These dynamically link to the `.bin` files on disk, allowing the OS to page chunks of data into memory only when strictly needed.

- `month_btree`: An in-memory Clustered B-Tree (`IOBTree` from the `BTrees` library). Because the data is physically sorted by month on disk, all identical months sit in contiguous rows. The B-Tree maps each integer `month_code` to its exact physical row boundaries `(start_row_idx, end_row_idx)`.

- `num_chunks`: ceil (`row_count` / `CHUNK_SIZE`)

- `zone_maps`: To skip irrelevant data blocks, we store metadata for chunks of 1,000 rows. This is a nested list where we track:
    - column `town`: A unique set of towns that appeared in the chunk.
    - column `floor_area`: `[min floor area, max floor area]` for the chunk.
    - column `resale_price`: `[min price, max price]` for the chunk.

Lastly, we log out the column store data structure summary as `database_state.json` to the `Logs` folder.

## Query

### 1. Set up
`queries` are stored in a column-store like fashion. In this list of lists:
- The first list stores the `(start_month_code, end_month_code)` of each query.
- The second list stores the `minimum_floor_area` (i.e., y) of each query.
- The last list stores the metadata `(x, y)` of each query for tracking.

The `target_town_names` are stored as a separate list of strings as they apply to all queries.

### 2. Filtering
Instead of scanning column-by-column using loops, we utilise a variety of techniques, such as the B-Tree, Zone Maps, and NumPy vectorization.

**Level 1: Clustered Index (B-Tree)**
For a given query, the system first queries the in-memory B-Tree `month_btree` with `min=target_start_month` and `max=target_end_month`. This operates in logarithmic time and returns the `min_row_idx` and `max_row_idx` boundaries for that date range.

**Level 2: Zone Map Data Skipping**
The query calculates which 1,000-row chunks overlap with the B-Tree's row boundaries. For each overlapping chunk, it checks the `town` and `floor_area` zone maps. If a chunk's max floor area is smaller than the target `y`, or if the chunk does not contain any of the target towns, the entire 1,000-row chunk is skipped without performing any disk I/Os.

**Level 3: NumPy Vectorization**
For the chunks that pass the Zone Map checks, the program extracts the overlapping data slices from the memory-mapped arrays. Instead of iterating row-by-row, it uses NumPy vectorized operations to evaluate the entire data slice simultaneously in C. 
It creates a `valid_town_mask` and a `valid_area_mask`, and combines them using a bitwise AND operation. The resulting `True` indices are translated to global row indices and added to the candidate list.

### 3. PSM Tie-Breaking and Late Materialization
Finally, for the matching candidate rows, the system calculates the Price Per Square Meter (PSM). It tracks the row with the absolute minimum PSM (bounded by `MAX_PSM`). If multiple rows tie, it applies tie-breaking (earliest month -> alphabetical town -> alphabetical block). 

Once the final row index is determined, the system fetches the encoded values from the separate memory-mapped columns, decoding them via `code_val_mapper`, and writing the final human-readable line to the output CSV.