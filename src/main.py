import nasdaq as nq 

# From https://data.nasdaq.com/
dataset = nq.getDataset('LBMA/SILVER', start_date='2015-12-31', end_date='2020-12-31')
print(dataset)
