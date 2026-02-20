# SC4023 Big Data Management - HDB Resale Data Analysis

## Updates
- **2026-02-15:** Implemented the column store database class
- **2026-02-20:** Implemented querying

## To Start
Simply install the necessary dependencies from `requirements.txt` and run `main.py`. You will be prompted to enter a matriculation number, which will be parsed to retrieve the correct queries. A copy of the database (since its non persistent) as well as the logs (`run_<matriculation_num>.log`) will be saved in `Logs` folder. The output `ScanResult_<matriculation_num>.csv` will be saved in `Results` folder.

## Database Design
The entire csv is loaded into memory (its only 20.6MB) as a Dataframe. We then parse this temporary Dataframe column by column, and populate our column store database.

The column store database is made up of the following data structures.
- `col_names`: A dictionary that maps each column name to a index.

    - In the original csv, column `month` contains strings of format "Mon-YY" (eg Jan-15, Apr-16). We preprocess this by splitting this into 2 seperate columns, (new) `month` column, which contain the strings "Jan", "Feb", etc; and `year` column, which contains the 2 digits year in the form of int.

- `val_code_mapper`: For each column, we retrieve the unique values and sort them by increasing order. All columns beside (new) `month` can be sorted lexicographically, and we use a dictionary to map each value to a integer code starting from 0. For `month`, we simply use the mapping `MONTH_MAP_DIGIT` in `constants.py` (the integer code here starts from 1).
    
    - Note that the code for `town` here is different from the digit for querying in Table 1.

- `code_val_mapper`: The reverse mapping of integer code to unique values for each column. This can be done as integer code and unique values are one-to-one.

- `columns`: Main Column Store Database. A list of numpy arrays, where each numpy array corresponds to a column, and stores the values from `ResalePricesSingapore.csv` in terms of their codes.

- `row_count` and `column_count`: Self explainatory

- `num_chunks`: ceil (`row_count` / `CHUNK_SIZE`)

- `zone_maps`: For selected columns only, we store a list of metadata for each chunk. This is a list of list of list of int, where the outer list corresponds to the columns, the middle list corresponds to a chunk for each column, and the inner list is the metadata of each chunk. The contents of the inner list differs for each column: (all stored as codes)

    - column `month`: [earliest month, latest month] for each chunk
    - column `town`: [towns that appeared in the chunk] for each chunk
    - column `floor_area`: [min floor area, max floor area] for each chunk
    - column `resale_price`: [min price, max price] for each chunk
    - all other columns: empty array [] for each chunk

Lastly, the temporary Dataframe is cleared, and we log out the column store data structure as `database_state.json` to `Logs` folder.

## Query

### 1. Set up

`queries` are stored in a column-store like fashion. In this list of list, the first list stores the `(start_month, end_month)` of each query; the second list stores the `(start_year, end_year)` of each query; the third list stores the `minimum_floor_area` (i.e. y) of each query; the last list stores the metadata `(x, y)` of each query. The index of each element in the inner list corresponds to the query index. This design allow us to scan each column in the database no more than once, and for each column, we can check the condition for all queries at once.

The `target_town_names` are stored as a seperate list of strings as it applies for all queries.

Note that 1) `start_month` is not necessarily smaller than `end_month` if the query spans for more than one calender year; 2) All values here are in their original values and yet to be encoded as codes in the database.

We keep a list of queues `valid_rows` to store for each query, the current rows index that are valid. They are initially empty, and will be populated once we have parsed the first column. For subsequent columns, only the row indexes in the queues will be checked, and if they do not satify the conditions, they will be dequeued.


### 2. Column by column scanning

We currently scan in a deterministic manner: `month` -> `year` -> `town` -> `floor_area`.

For column `month`, for each query, we only iterate each chunk. we use `zone_maps` to check if the chunk contains the target months, and we will skip the chunk entirely if it doesn't. We split into the following cases:

- If `start_month` <= `end_month` (i.e. `start_year` = `end_year` since `x` <= 8), then we skip the chunk iff 

    ``` chunk_max_month < start_month OR end_month < chunk_min_month```
    
    e.g. chunk is from May to August, and we have end month = April OR start month = September.

- If `start_month` > `end_month` (i.e. `start_year` < `end_year`), then we skip the chunk iff

    ```end_month < chunk_min_month AND chunk_max_month < start_month```

    e.g. chunk is from May to August, and we have start_month = September AND end month = April.

If we did not skip the chunk, then it may have valid rows, so we have to scan row by row for this chunk. Again, we have to split into the following cases:
    
- If `start_month` <= `end_month`, then we take the row iff

    ```start_month <= row_month <= end_month```

- If `start_month` > `end_month`, then we take the row iff

    ```row_month <= end_month OR  start_month <= row_month```

For column `year` onwards, for each query, we no longer have to scan chunk by chunk, we just have to query the valid row indices. We keep the row if the `start_year` <= `row_year` <= `end_year`. However:

- If `start_year` = `row_year`, we keep the row iff `row_month` >= `start_month`
- If `end_year` = `row_year`, we keep the row iff `row_month` <= `end_month`

For column `town`, we keep the row iff `row_town` is in `target_town_names`.

For column `floor_area`, we keep the row iff `row_floor_area` >= `minimum_floor_area`.

We have found the queue of valid row indices for each query, and for each query, we will find the row with the minimum psm, and append it to our output csv.


## TODO

- Only the `month` column use `zone_map` as it is the first column to be scanned. We should let the user to choose the order of the four columns to parse so that we can compare the performance.

- As you may have noticed, for `month` column (or whichever column thats scanned first), it still scan the entire column for each query. Change the logic abit so that we only scan the column once.

- Load the csv data to an actual database to validate our results.

- Consider the case when the entire data cannot be fully loaded into the memory at once. Then maybe we have to load chunk by chunk. And for querying, we will also load the column store database from storage chunk by chunk.

- I divided `month` column into `month` and `year` cos i thought it will be easier to parse. Now its making things much more difficult. The code now will fail if `x` >= 12. Can fix but not required.

- Enhance column stores: the lecture presented 5 (or 6 depending on how you count it) techniques to optimize column store. We should at least try to implement all of them here, if have time then can add more.

    - ✅ **Compression:** Already implemented by `val_code_mapper`

    - ❌ **Shared Scan:** Implemented in querying but require some fixing

    - ✅ **Zone Map:** Already implemented by `zone_maps`

    - ❌ **Enhanced with sorting:** Need change the csv loading logic of column store DB. But functionality wise overlaps with Zone Map, not really necessary to implement?

    - ❌ **Enhanced with Index:** Literally skipped by Prof, but we can try to implement if you guys want to brush up ur B+ tree.

     - ❌ **Vectorized Processing:** For cache friendliness, but we literally load the entire csv to memory though. So I dont think need to implement?
