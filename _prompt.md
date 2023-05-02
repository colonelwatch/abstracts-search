### Instruction
Given the query, infer the motivation behind the query. Motivations include but is not limited to: solving a problem, getting basic understanding, or finding deeper concepts. From this, give the title and abstract of an academic paper that answers the query and the motivation.
### Example
query:: How to separate speech signals from mixture in real-time
motivation:: Solving the problem caused by crosstalk between signals
title:: An On-line Algorithm for Blind Source Separation on Speech Signals
text:: In this article, we propose an on-line algorithm for Blind Source Separation of speech signals, which is recorded in a real environment. This on-line algorithm makes it possible to trace the changing environment. The idea is to apply some on-line algorithm in the time-frequency domain. We show some results of experiments.
### Example
query:: what is the performance of inner joining many tables vs making many subqueries
motivation:: Getting an deeper understanding of how SQL servers work
title:: Joins versus Subqueries: Which Is Faster?
text:: I won't leave you in suspense, between Joins and Subqueries, joins tend to execute faster. In fact, query retrieval time using joins will almost always outperform one that employs a subquery. The reason is that joins mitigate the processing burden on the database by replacing multiple queries with one join query. This in turn makes better use of the database's ability to search through, filter, and sort records. Having said that, as you add more joins to a query, the database server has to do more work, which translates to slower data retrieval times. While joins are a necessary part of data retrieval from a normalized database, it is important that joins be written correctly, as improper joins can result in serious performance degradation and inaccurate query results. There are also some cases where a subquery can replace complex joins and unions with only minimal performance degradation, if any.
### Response
query:: [REP]
motivation::