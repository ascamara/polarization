# polarization
Code for "Identifying and Quantifying Polarization in Categorized Corpora"

To run: py driver.py
Note the directory stem, sources, CONTEXT_SIZE, CORPUS_WIDE_MEASUREMENT_TERMS variables to configure the pipeline.
Go to the main function on line 209 to change which combination of the sources you'd like to run.

Note for the data:
Summary is a CSV file with the fields: term, controversy score, tfidf (joint significance), log odds (marginal significance). NaN if field empty.
Report is a detailed text file with the terms, scores, and contexts.
