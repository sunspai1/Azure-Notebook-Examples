from octopy.db import Presto
from fastcore.all import L, merge
import pickle
import numpy as np

sql = Presto(username='hamelsmu')

test_query="""
select 
    repository_id as id,
    concat(repository_owner, '/', repository_name) as name,
    substr(FROM_UTF8(content), 1, 8000) as content
from hive.suez.readme_v2 where day = (select max(day) from hive.suez."readme_v2$partitions") 
and LOWER("path") = 'readme.md' 
and content_size > 75 
limit 10000
"""

df = sql.execute(test_query, params=None, return_type='df')

df.to_parquet("./outputs/data.parquet")