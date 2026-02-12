# code likely dont work!

## My prompt:

I am doing a big data mangement course project for my uni. For this project we are given a csv of 19.6 Mb with 10 columns and 259238 rows (including heading). The columns are in the photo attached. For this project, we are to use column stored learned to store the data in a column manner, optimize it, and do some queiries. 

The queiries are for all x (month) in [1,8], and for all y (floor area) in [80, 150], the program should find all entries from a certain month, year (chosen by user, the earlist month in csv is jan 2015, latest is dec 2025) to start month, year + x -1 (so eg if start is2021 august, and x is 2, then search up to 2021 sep; if x is 8, then search up to march 2022), and user chosen towns (out of 10), and floor area >= y. The program find all valid entry and return the entry with min price per square meter (= resale price / floor area). Save all entry to a csv file, so there should be x*y entries in output csv result. Details are in   picture attached.

rec is java, but i wan use python. Write a python program that does the following. 
1) load the csv entirely into memory, can fit i think. use a numpy array to store each column. 
2) optimization: we learnt 5 methods to optimize: vectorized scan, compression, shared scan, zone map, and sorting. 
2a) for compression, we can map value of each column to a binary value eg 000, 001,002,... we also need a lookup table for each column. numpy array is fine. also make sure the mapping is in increasing order of values. sort between mapping, dont assume its sorted
2b) we can do sorting. sort by increasing order of month, then group towns together, sort by increasing order of floor area, then sort by increasing order of resale price. then we can use binary search later. 
2c) we can do zone map. maybe break them up in chunks of equal size, then have a dict or smt to keep track for each chunk, the max and min value. We only need do 2b and 2c for column town, month, floor area, and resale price, only these 4 used in filtering, other 6 is just noise.
2d) i havent think of ways to implement vectorized scan and shared scan.

3) wait for response from user. The user need choose start month and year, and town(s) to filter. Alterntaively, he can key in his matric number. the matric number is a 9 char string, first and last is a upper case letter, middle 7 are int between 0 and 9. Last int shows the year: 0 -> 2020, 1: 2022, ... 4: 2024, 5: 2015, 6: 2016, ... 9: 2019. we do not start query from 2025. second last int shows month (0 -> oct, 1-> jan, ... 9-> sep) these 2 give the starrt month and year. towns are the set of int that appeared in all 7 int. see picture. 

4) for each x in [1, 8], y in [80, 150], do the filtering. scan month col first, get all matched pos stored in queue. then pop as scanning town col, if match then push back if town matched, then do for floor area. at the end the queue has only matched index.

5) save all matched idx and corresponding rows in a log file for each x, y. (for personal reference and debug only). then find the row with min price per sqm and save in output csv file.

6) at the end, clear all memories used. just need output result csv and log file.

implement it as i described.  its okay if not most efficient. only change if the logic i desribed is logically incorrect, but let me know. the code should be clean and clear and understandable with comments. after you have implemented the code. you can put in below section the optimization improvement. for eg vectorized scan and shared scan. you can rec improvements outside of the lec. but dont add them in the main code. i want to get the base code working first, then add improvement one at a time. oh and also rephrase my request cleaned, so i can show the prompt to my teammate.

---

## Gemini Response:

Here is the refined project description and the complete Python implementation.

### 1. Rephrased Project Request (for your teammate)

**Project Title:** Column-Oriented HDB Resale Data Analysis

**Objective:**
Develop a Python program to analyze Singapore HDB resale data (2015–2025). The program must implement a **column-oriented storage** architecture (simulated using NumPy arrays) and specific query optimization techniques (Dictionary Encoding, Sorting, and Zone Maps) to efficiently find the minimum resale price per square meter based on user constraints.

**Key Requirements:**

1. **Data Loading:** Load `ResalePricesSingapore.csv` entirely into memory, storing each column as a separate NumPy array.

2. **Optimizations (Pre-processing):**
* **Dictionary Encoding:** Compress string columns (e.g., Town, Flat_Model) into integers using sorted lookup tables.
* **Sorting:** Sort the entire dataset by **Month  Town  Floor Area  Resale Price** to enable efficient scanning.
* **Zone Maps:** Divide data into chunks (e.g., 1024 rows) and store min/max statistics for filter columns (Month, Town, Floor Area) to skip irrelevant chunks.

3. **User Input (Matriculation Number):**
* Parse a matriculation number (e.g., `A1234567B`) to determine the query parameters:
* **Target Year:** Derived from the last digit (e.g., ).
* **Start Month:** Derived from the 2nd last digit ().
* **Towns:** Filter based on all digits present in the matriculation number (mapped via the provided Town Table).

4. **Query Execution:**
* Iterate through a duration  (1 to 8 months) and minimum floor area  ( to ).
* For each  pair, filter records matching the date range, town list, and area requirement.
* Identify the record with the **Minimum Price Per Square Meter**.


5. **Output:**
* Generate a CSV file `ScanResult_<MatricNum>.csv` containing the best record for each valid  pair.
* Include a log file for debugging.


### 3. Optimization Explanation (for your Report/Improvement section)

You mentioned you need to describe improvements. Since the code above handles the "Base" requirements (Column Store + Dict Encoding + Zone Maps), here is how you can describe **Vectorized Scan** and **Shared Scan** as further improvements:

1. **Vectorized Scan (SIMD):**
* *Concept:* Instead of checking rows one by one (tuple-at-a-time), we process data in blocks (vectors) to utilize CPU cache and SIMD (Single Instruction, Multiple Data) instructions.
* *Python Context:* NumPy is inherently vectorized. In the code above, `mask = (chunk_months >= start_code) & (chunk_months <= end_code)` is a vectorized operation. It compares the entire chunk array at once in C-level loops rather than a Python for-loop. This is already implemented in my solution implicitly via NumPy. You can highlight this in your report as "utilizing NumPy's vectorization for column scanning."


2. **Shared Scan (Query Batching):**
* *Concept:* Since the project asks to loop through  and , many queries overlap. For example, the scan for  (8 months duration) covers the same data as  plus 7 more months.
* *Implementation Idea:* Instead of running distinct queries for every loop, you could scan the table once. As you iterate through rows, check which of the active  queries this row satisfies and update their intermediate states (e.g., maintaining a "current minimum" for each query bucket). This dramatically reduces memory access because you read the table once rather than  times.

### 4. Implementation Notes

* **Matric Year Logic:** I followed the PDF source strictly. If the matric ends in `5`, the year is 2015. If it ends in `0`, it is 2020.
* **Rounding:** I used standard `round()` and cast to `int` as per the example output format.
* **File Handling:** The code assumes `ResalePricesSingapore.csv` is in the same folder.
* **Memory Management:** The `del db` and `gc.collect()` at the end ensure memory is freed as requested.
