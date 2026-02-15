# SC4023 Big Data Management - HDB Resale Data Analysis

## Updates
- **2026-02-15:** Implemented the column store database class

## Database Design
The entire csv is loaded into memory (its only 20.6MB) as a Dataframe. We then parse this temporary Dataframe column by column, and populate our column store database.

The column store database is made up of the following data structures.
- `col_names`: A dictionary that maps each column name to a index.

- `val_code_mapper`: For each column, we retrieve the unique values and sort them by increasing order. All columns beside 'month' can be sorted lexicographically. For 'month' since its a string in 'MMM-YY' format, we need to convert to a (year, month idx) int tuple for sorting. This is a list of dictionary, the outer list is the column index, while the inner dictionary map the increasingly sorted unique values to their unique index.
    - Note that the index for `town` here is different from the index for querying in Table 1.

- `columns`: A list of numpy arrays, where the outer list is the column index, while the inner numpy arrays store the values (as their index) of each column, following the same order from csv.

- `row_count` and `column_count`: Self explainatory

- `num_chunks`: ceil (`row_count` / `CHUNK_SIZE`)

- `zone_maps`: For selected columns only, we store a list of metadata for each chunk. This is a list of list of list of int, where the outer list is the column index, the middle list is the per chunk, and the inner list is the metadata of each chunk. The metadata are of course stored as the value index instead of the actual values

    - column `month`: [earliest month, latest month] for each chunk
    - column `town`: [towns that appeared in the chunk] for each chunk
    - column `floor_area`: [min floor area, max floor area] for each chunk
    - column `resale_price`: [min price, max price] for each chunk
    - all other columns: empty array

Lastly, the temporary Dataframe is cleared, and we log out the column store data structure as a json file to `Logs` folder.

## TODO

- Querying (currently its a non workable code by Gemini)

- Enhance column stores: the lecture presented 5 (or 6 depending on how you count it) techniques to optimize column store. We should at least try to implement all of them here, if have time then can add more.

    - ✅ **Compression:** Already implemented by `val_code_mapper`

    - ❌ **Shared Scan:** Implement together with querying

    - ✅ **Zone Map:** Already implemented by `zone_maps`

    - ❌ **Enhanced with sorting:** Need change the csv loading logic of column store DB. But functionality wise overlaps with Zone Map, not really necessary to implement?

    - ❌ **Enhanced with Index:** Literally skipped by Prof, but we can try to implement if you guys want to brush up ur B+ tree.

     - ❌ **Vectorized Processing:** For cache friendliness, but we literally load the entire csv to memory though. So I dont think need to implement?
